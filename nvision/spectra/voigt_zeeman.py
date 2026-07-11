"""Voigt-broadened NV center model with Zeeman splitting."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.numba_kernels import (
    nv_center_zeeman_pseudo_voigt_eval,
    nv_center_zeeman_pseudo_voigt_vectorized_many,
    nv_center_zeeman_pseudo_voigt_vectorized_many_fast,
    nv_center_zeeman_pseudo_voigt_vectorized_one_serial,
)
from nvision.spectra.signal import SignalModel
from nvision.spectra.spec import GenericParamSpec


@dataclass(frozen=True)
class VoigtZeemanSpectrum:
    frequency: float
    fwhm_total: float
    lorentz_frac: float
    zeeman_split: float
    split: float
    k_np: float
    c_total: float
    background: float


@dataclass(frozen=True)
class VoigtZeemanSpectrumSamples:
    frequency: np.ndarray
    fwhm_total: np.ndarray
    lorentz_frac: np.ndarray
    zeeman_split: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    c_total: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class VoigtZeemanSpectrumUncertainty:
    frequency: float
    fwhm_total: float
    lorentz_frac: float
    zeeman_split: float
    split: float
    k_np: float
    c_total: float
    background: float


class _VoigtZeemanSpec(
    GenericParamSpec[
        VoigtZeemanSpectrum,
        VoigtZeemanSpectrumSamples,
        VoigtZeemanSpectrumUncertainty,
    ]
):
    params_cls = VoigtZeemanSpectrum
    samples_cls = VoigtZeemanSpectrumSamples
    uncertainty_cls = VoigtZeemanSpectrumUncertainty


class VoigtZeemanModel(SignalModel[VoigtZeemanSpectrum, VoigtZeemanSpectrumSamples, VoigtZeemanSpectrumUncertainty]):
    """Voigt-broadened NV center model with Zeeman splitting.

    Uses a two-width pseudo-Voigt approximation
    (:func:`~nvision.spectra.numba_kernels.nv_center_zeeman_pseudo_voigt_eval`), not a true Voigt
    profile (no ``wofz``/error-function evaluation). The Lorentzian and Gaussian components are
    evaluated at their own split widths rather than the combined Voigt FWHM the standard
    Thompson-Cox-Hastings mixing weight assumes, so the approximation error vs. a true Voigt is
    uncontrolled by that calibration; see ``tests/spectra/test_pseudo_voigt_accuracy.py``.

    Models an NV center as two Zeeman-split groups (ms=+1/-1), each group a hyperfine
    triplet of Voigt profile dips. Each Lorentzian dip is convolved with a Gaussian, which
    accounts for both homogeneous (Lorentzian) and inhomogeneous (Gaussian) broadening.

    Parameters
    ----------
    frequency : float
        Central (zero-field) frequency (f_B)
    fwhm_total : float
        Total effective linewidth (Lorentzian + Gaussian)
    lorentz_frac : float
        Lorentzian share of broadening in [0, 1]
    zeeman_split : float
        Half-separation between the two Zeeman groups
    split : float
        Hyperfine splitting (delta_f_HF) within each Zeeman group
    k_np : float
        Non-polarization factor (amplitude ratio between hyperfine peaks)
    c_total : float
        Population-normalized total contrast, split across both Zeeman groups
        and the hyperfine triplet within each via ``k_np``.
    background : float
        Background level
    """

    def compute_voigt_zeeman_model(
        self,
        x: float,
        frequency: float,
        fwhm_total: float,
        lorentz_frac: float,
        zeeman_split: float,
        split: float,
        k_np: float,
        c_total: float,
        background: float,
    ) -> float:
        """Zeeman + hyperfine pseudo-Voigt NV model; parameter order matches :meth:`parameter_names`."""
        return nv_center_zeeman_pseudo_voigt_eval(
            float(x),
            float(frequency),
            float(fwhm_total),
            float(lorentz_frac),
            float(zeeman_split),
            float(split),
            float(k_np),
            float(c_total),
            float(background),
        )

    _SPEC = _VoigtZeemanSpec()

    @property
    def spec(self) -> _VoigtZeemanSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("fwhm_total", "c_total")

    def expected_dip_count(self) -> int:
        """Zeeman splitting produces 2 resolvable groups (each an unresolved hyperfine triplet)."""
        return 2

    def compute(self, x: float, params: VoigtZeemanSpectrum) -> float:
        return self.compute_voigt_zeeman_model(
            float(x),
            params.frequency,
            params.fwhm_total,
            params.lorentz_frac,
            params.zeeman_split,
            params.split,
            params.k_np,
            params.c_total,
            params.background,
        )

    def compute_vectorized_samples(self, x: float, samples: VoigtZeemanSpectrumSamples) -> np.ndarray:
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        n = freq.shape[0]
        out = np.empty(n, dtype=FLOAT_DTYPE)
        nv_center_zeeman_pseudo_voigt_vectorized_one_serial(
            float(x),
            freq,
            np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE),
            np.asarray(samples.zeeman_split, dtype=FLOAT_DTYPE),
            np.asarray(samples.split, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples.c_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
            out,
        )
        return out

    def compute_vectorized_many(self, x_array: Sequence[float], samples: VoigtZeemanSpectrumSamples) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")

        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        out = np.empty((xs.shape[0], freq.shape[0]), dtype=FLOAT_DTYPE)
        nv_center_zeeman_pseudo_voigt_vectorized_many(
            xs,
            freq,
            np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE),
            np.asarray(samples.zeeman_split, dtype=FLOAT_DTYPE),
            np.asarray(samples.split, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples.c_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
            out,
        )
        return out.astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many_fast(
        self, x_array: Sequence[float], samples: VoigtZeemanSpectrumSamples
    ) -> np.ndarray:
        """Acquisition-only fast variant: uses the fastmath Zeeman pseudo-Voigt kernel."""
        if not hasattr(samples, "frequency"):
            return super().compute_vectorized_many_fast(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        out = np.empty((xs.shape[0], freq.shape[0]), dtype=FLOAT_DTYPE)
        nv_center_zeeman_pseudo_voigt_vectorized_many_fast(
            xs,
            freq,
            np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE),
            np.asarray(samples.zeeman_split, dtype=FLOAT_DTYPE),
            np.asarray(samples.split, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples.c_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
            out,
        )
        return out.astype(FLOAT_DTYPE, copy=False)

    def sample_params(self, rng: random.Random) -> VoigtZeemanSpectrum:
        """Sample parameters that keep the signal within [0, 1].

        ``zeeman_split`` is drawn strictly larger than ``split`` (as ``split + gap``) so the two
        hyperfine sub-triplets never interleave past the shared center frequency. ``fwhm_total``
        is then drawn as a multiple of ``2 * zeeman_split`` (the group-to-group separation), so
        the ratio between linewidth and group separation is what varies — spanning cleanly
        separated groups (ratio << 1), groups that visibly overlap but are still two distinct
        dips (ratio ~ 1), and a single fully merged dip (ratio >> 1). It's also frequently well
        beyond ``split``, so strong inhomogeneous broadening regularly washes out the fine
        (hyperfine) structure into a single dip per Zeeman group on top of that.
        """
        lorentz_frac = rng.uniform(0.23, 0.89)
        split = rng.uniform(0.05, 0.12)
        zeeman_split = split + rng.uniform(0.05, 0.12)
        fwhm_ratio = rng.uniform(0.15, 1.1)  # fwhm_total / (2 * zeeman_split)
        fwhm_total = max(fwhm_ratio * 2.0 * zeeman_split, 0.03)
        k_np = rng.uniform(2.0, 4.0)
        margin = zeeman_split + split + 0.08
        frequency = rng.uniform(margin, 1.0 - margin)
        background = 1.0

        # Estimate c_total (population-normalized amplitude) using a coarse grid.
        # We use compute_vectorized_many with background=0 and c_total=1.
        # The max dip depth is then 1.0 / max_dip_observed.
        xs = np.linspace(frequency - margin, frequency + margin, 400)
        samples = VoigtZeemanSpectrumSamples(
            frequency=np.array([frequency], dtype=FLOAT_DTYPE),
            fwhm_total=np.array([fwhm_total], dtype=FLOAT_DTYPE),
            lorentz_frac=np.array([lorentz_frac], dtype=FLOAT_DTYPE),
            zeeman_split=np.array([zeeman_split], dtype=FLOAT_DTYPE),
            split=np.array([split], dtype=FLOAT_DTYPE),
            k_np=np.array([k_np], dtype=FLOAT_DTYPE),
            c_total=np.array([1.0], dtype=FLOAT_DTYPE),
            background=np.array([0.0], dtype=FLOAT_DTYPE),
        )
        # compute_vectorized_many returns bg - dips.
        # Here bg=0, so it returns -dips.
        res = self.compute_vectorized_many(xs, samples)
        max_dip = -float(res.min())
        c_total = 1.0 / max_dip if max_dip > 1e-6 else 1.0

        return VoigtZeemanSpectrum(
            frequency=frequency,
            fwhm_total=fwhm_total,
            lorentz_frac=lorentz_frac,
            zeeman_split=zeeman_split,
            split=split,
            k_np=k_np,
            c_total=c_total,
            background=background,
        )
