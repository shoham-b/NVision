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
from nvision.spectra.nv_center import hyperfine_geometry
from nvision.spectra.signal import SignalModel
from nvision.spectra.spec import GenericParamSpec


@dataclass(frozen=True)
class VoigtZeemanSpectrum:
    zero_field_freq: float
    fwhm_total: float
    lorentz_frac: float
    zeeman_shift: float
    hyperfine_shift: float
    k_np: float
    contrast_total: float
    background: float


@dataclass(frozen=True)
class VoigtZeemanSpectrumSamples:
    zero_field_freq: np.ndarray
    fwhm_total: np.ndarray
    lorentz_frac: np.ndarray
    zeeman_shift: np.ndarray
    hyperfine_shift: np.ndarray
    k_np: np.ndarray
    contrast_total: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class VoigtZeemanSpectrumUncertainty:
    zero_field_freq: float
    fwhm_total: float
    lorentz_frac: float
    zeeman_shift: float
    hyperfine_shift: float
    k_np: float
    contrast_total: float
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

    Uses a pseudo-Voigt approximation
    (:func:`~nvision.spectra.numba_kernels.nv_center_zeeman_pseudo_voigt_eval`), not a true Voigt
    profile (no ``wofz``/error-function evaluation). The Lorentzian and Gaussian components are
    evaluated at the shared combined ``fwhm_total``, matching the standard Thompson-Cox-Hastings
    mixing weight's calibration; see ``tests/spectra/test_pseudo_voigt_accuracy.py`` for the
    residual approximation error.

    Models an NV center as two Zeeman-split groups (ms=+1/-1), each group a hyperfine
    multiplet of Voigt profile dips. Each Lorentzian dip is convolved with a Gaussian, which
    accounts for both homogeneous (Lorentzian) and inhomogeneous (Gaussian) broadening.

    Unlike the models in :mod:`nvision.spectra.nv_center`, ``hyperfine_shift`` is always a
    free parameter here, so this model always *has* hyperfine structure -- the
    ``hyperfine`` constructor argument only chooses its shape (``"n14"``, the
    default, for the triplet; ``"n15"`` for the doublet with no central line).

    Parameters
    ----------
    zero_field_freq : float
        Zero-field dip centre ``zero_field_freq`` (f_B)
    fwhm_total : float
        Total effective hwhm (Lorentzian + Gaussian)
    lorentz_frac : float
        Lorentzian share of broadening in [0, 1]
    zeeman_shift : float
        Half-separation between the two Zeeman groups
    hyperfine_shift : float
        Hyperfine splitting (delta_f_HF) within each Zeeman group
    k_np : float
        Non-polarization factor (amplitude ratio between hyperfine peaks)
    contrast_total : float
        Population-normalized total contrast, hyperfine_shift across both Zeeman groups
        and the hyperfine lines within each via ``k_np``.
    background : float
        Background level
    """

    def __init__(self, hyperfine: str = "n14") -> None:
        if hyperfine == "unresolved":
            raise ValueError("VoigtZeemanModel always has hyperfine structure; use 'n14' or 'n15'")
        _, self._w_center, self._hf_lines = hyperfine_geometry(hyperfine)
        self._hyperfine = hyperfine

    def compute_voigt_zeeman_model(
        self,
        x: float,
        zero_field_freq: float,
        fwhm_total: float,
        lorentz_frac: float,
        zeeman_shift: float,
        hyperfine_shift: float,
        k_np: float,
        contrast_total: float,
        background: float,
    ) -> float:
        """Zeeman + hyperfine pseudo-Voigt NV model; parameter order matches :meth:`parameter_names`."""
        return nv_center_zeeman_pseudo_voigt_eval(
            float(x),
            float(zero_field_freq),
            float(fwhm_total),
            float(lorentz_frac),
            float(zeeman_shift),
            float(hyperfine_shift),
            float(k_np),
            float(self._w_center),
            float(contrast_total),
            float(background),
        )

    _SPEC = _VoigtZeemanSpec()

    @property
    def spec(self) -> _VoigtZeemanSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("fwhm_total", "contrast_total")

    def expected_dip_count(self) -> int:
        """Hyperfine lines per group x the 2 Zeeman groups."""
        return 2 * self._hf_lines

    def compute(self, x: float, params: VoigtZeemanSpectrum) -> float:
        return self.compute_voigt_zeeman_model(
            float(x),
            params.zero_field_freq,
            params.fwhm_total,
            params.lorentz_frac,
            params.zeeman_shift,
            params.hyperfine_shift,
            params.k_np,
            params.contrast_total,
            params.background,
        )

    def compute_vectorized_samples(self, x: float, samples: VoigtZeemanSpectrumSamples) -> np.ndarray:
        zero_field_freq = np.asarray(samples.zero_field_freq, dtype=FLOAT_DTYPE)
        n = zero_field_freq.shape[0]
        out = np.empty(n, dtype=FLOAT_DTYPE)
        nv_center_zeeman_pseudo_voigt_vectorized_one_serial(
            float(x),
            zero_field_freq,
            np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE),
            np.asarray(samples.zeeman_shift, dtype=FLOAT_DTYPE),
            np.asarray(samples.hyperfine_shift, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            self._w_center,
            np.asarray(samples.contrast_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
            out,
        )
        return out

    def compute_vectorized_many(self, x_array: Sequence[float], samples: VoigtZeemanSpectrumSamples) -> np.ndarray:
        if not hasattr(samples, "zero_field_freq"):
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")

        zero_field_freq = np.asarray(samples.zero_field_freq, dtype=FLOAT_DTYPE)
        out = np.empty((xs.shape[0], zero_field_freq.shape[0]), dtype=FLOAT_DTYPE)
        nv_center_zeeman_pseudo_voigt_vectorized_many(
            xs,
            zero_field_freq,
            np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE),
            np.asarray(samples.zeeman_shift, dtype=FLOAT_DTYPE),
            np.asarray(samples.hyperfine_shift, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            self._w_center,
            np.asarray(samples.contrast_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
            out,
        )
        return out.astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many_fast(self, x_array: Sequence[float], samples: VoigtZeemanSpectrumSamples) -> np.ndarray:
        """Acquisition-only fast variant: uses the fastmath Zeeman pseudo-Voigt kernel."""
        if not hasattr(samples, "zero_field_freq"):
            return super().compute_vectorized_many_fast(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        zero_field_freq = np.asarray(samples.zero_field_freq, dtype=FLOAT_DTYPE)
        out = np.empty((xs.shape[0], zero_field_freq.shape[0]), dtype=FLOAT_DTYPE)
        nv_center_zeeman_pseudo_voigt_vectorized_many_fast(
            xs,
            zero_field_freq,
            np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE),
            np.asarray(samples.zeeman_shift, dtype=FLOAT_DTYPE),
            np.asarray(samples.hyperfine_shift, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            self._w_center,
            np.asarray(samples.contrast_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
            out,
        )
        return out.astype(FLOAT_DTYPE, copy=False)

    def sample_params(self, rng: random.Random) -> VoigtZeemanSpectrum:
        """Sample parameters that keep the signal within [0, 1].

        ``zeeman_shift`` is drawn strictly larger than ``hyperfine_shift`` (as ``hyperfine_shift + gap``) so the two
        hyperfine sub-triplets never interleave past the shared ``zero_field_freq``. ``fwhm_total``
        is then drawn as a multiple of ``2 * zeeman_shift`` (the group-to-group separation), so
        the ratio between hwhm and group separation is what varies — spanning cleanly
        separated groups (ratio << 1), groups that visibly overlap but are still two distinct
        dips (ratio ~ 1), and a single fully merged dip (ratio >> 1). It's also frequently well
        beyond ``hyperfine_shift``, so strong inhomogeneous broadening regularly washes out the fine
        (hyperfine) structure into a single dip per Zeeman group on top of that.
        """
        lorentz_frac = rng.uniform(0.23, 0.89)
        hyperfine_shift = rng.uniform(0.05, 0.12)
        zeeman_shift = hyperfine_shift + rng.uniform(0.05, 0.12)
        fwhm_ratio = rng.uniform(0.15, 1.1)  # fwhm_total / (2 * zeeman_shift)
        fwhm_total = max(fwhm_ratio * 2.0 * zeeman_shift, 0.03)
        k_np = rng.uniform(2.0, 4.0)
        margin = zeeman_shift + hyperfine_shift + 0.08
        zero_field_freq = rng.uniform(margin, 1.0 - margin)
        background = 1.0

        # Estimate contrast_total (population-normalized amplitude) using a coarse grid.
        # We use compute_vectorized_many with background=0 and contrast_total=1.
        # The max dip depth is then 1.0 / max_dip_observed.
        xs = np.linspace(zero_field_freq - margin, zero_field_freq + margin, 400)
        samples = VoigtZeemanSpectrumSamples(
            zero_field_freq=np.array([zero_field_freq], dtype=FLOAT_DTYPE),
            fwhm_total=np.array([fwhm_total], dtype=FLOAT_DTYPE),
            lorentz_frac=np.array([lorentz_frac], dtype=FLOAT_DTYPE),
            zeeman_shift=np.array([zeeman_shift], dtype=FLOAT_DTYPE),
            hyperfine_shift=np.array([hyperfine_shift], dtype=FLOAT_DTYPE),
            k_np=np.array([k_np], dtype=FLOAT_DTYPE),
            contrast_total=np.array([1.0], dtype=FLOAT_DTYPE),
            background=np.array([0.0], dtype=FLOAT_DTYPE),
        )
        # compute_vectorized_many returns bg - dips.
        # Here bg=0, so it returns -dips.
        res = self.compute_vectorized_many(xs, samples)
        max_dip = -float(res.min())
        contrast_total = 1.0 / max_dip if max_dip > 1e-6 else 1.0

        return VoigtZeemanSpectrum(
            zero_field_freq=zero_field_freq,
            fwhm_total=fwhm_total,
            lorentz_frac=lorentz_frac,
            zeeman_shift=zeeman_shift,
            hyperfine_shift=hyperfine_shift,
            k_np=k_np,
            contrast_total=contrast_total,
            background=background,
        )
