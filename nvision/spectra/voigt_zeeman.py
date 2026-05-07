"""Voigt-broadened NV center model with Zeeman splitting."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.jax_kernels import nv_center_pseudo_voigt_jax
from nvision.spectra.numba_kernels import (
    nv_center_pseudo_voigt_eval,
    nv_center_pseudo_voigt_vectorized_many,
)
from nvision.spectra.signal import SignalModel
from nvision.spectra.spec import GenericParamSpec


@dataclass(frozen=True)
class VoigtZeemanSpectrum:
    frequency: float
    fwhm_total: float
    lorentz_frac: float
    split: float
    k_np: float
    dip_depth: float
    background: float


@dataclass(frozen=True)
class VoigtZeemanSpectrumSamples:
    frequency: np.ndarray
    fwhm_total: np.ndarray
    lorentz_frac: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class VoigtZeemanSpectrumUncertainty:
    frequency: float
    fwhm_total: float
    lorentz_frac: float
    split: float
    k_np: float
    dip_depth: float
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

    Not njit-accelerated: evaluation uses SciPy/JAX ``wofz`` or a pseudo-Voigt fallback.

    Models an NV center with three Voigt profile dips (hyperfine splitting),
    where each Lorentzian dip is convolved with a Gaussian. This accounts for
    both homogeneous (Lorentzian) and inhomogeneous (Gaussian) broadening.

    Parameters
    ----------
    frequency : float
        Central frequency (f_B)
    fwhm_total : float
        Total effective linewidth (Lorentzian + Gaussian)
    lorentz_frac : float
        Lorentzian share of broadening in [0, 1]
    split : float
        Hyperfine splitting (delta_f_HF)
    k_np : float
        Non-polarization factor (amplitude ratio between peaks)
    dip_depth : float
        Right (deepest) peak depth in [0, 1]. Center depth = dip_depth / k_np.
    background : float
        Background level
    """

    def compute_voigt_zeeman_model(
        self,
        x: float,
        frequency: float,
        fwhm_total: float,
        lorentz_frac: float,
        split: float,
        k_np: float,
        dip_depth: float,
        background: float,
    ) -> float:
        """Triple Voigt NV model; parameter order matches :meth:`parameter_names`."""
        return nv_center_pseudo_voigt_eval(
            float(x),
            float(frequency),
            float(fwhm_total),
            float(lorentz_frac),
            float(split),
            float(k_np),
            float(dip_depth),
            float(background),
        )


    _SPEC = _VoigtZeemanSpec()

    @property
    def spec(self) -> _VoigtZeemanSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("fwhm_total", "dip_depth")

    def expected_dip_count(self) -> int:
        """Zeeman splitting produces 3 dips: ms=-1, 0, +1 transitions."""
        return 3

    def compute(self, x: float, params: VoigtZeemanSpectrum) -> float:
        return self.compute_voigt_zeeman_model(
            float(x),
            params.frequency,
            params.fwhm_total,
            params.lorentz_frac,
            params.split,
            params.k_np,
            params.dip_depth,
            params.background,
        )

    def compute_vectorized_samples(self, x: float, samples: VoigtZeemanSpectrumSamples) -> np.ndarray:
        return self.compute_vectorized_many([x], samples)[0]

    def compute_vectorized_many(self, x_array: Sequence[float], samples: VoigtZeemanSpectrumSamples) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")

        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        out = np.empty((xs.shape[0], freq.shape[0]), dtype=FLOAT_DTYPE)
        nv_center_pseudo_voigt_vectorized_many(
            xs,
            freq,
            np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE),
            np.asarray(samples.split, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
            out,
        )
        return out.astype(FLOAT_DTYPE, copy=False)

    def compute_jax(self, x: float, params: VoigtZeemanSpectrum) -> Any:
        return nv_center_pseudo_voigt_jax(
            x,
            params.frequency,
            params.fwhm_total,
            params.lorentz_frac,
            params.split,
            params.k_np,
            params.dip_depth,
            params.background,
        )

    def sample_params(self, rng: random.Random) -> VoigtZeemanSpectrum:
        """Sample parameters that keep the signal within [0, 1]."""
        fwhm_total = rng.uniform(0.04, 0.13)
        lorentz_frac = rng.uniform(0.23, 0.89)
        split = rng.uniform(0.05, 0.12)
        k_np = rng.uniform(2.0, 4.0)
        frequency = rng.uniform(split + 0.1, 1.0 - split - 0.1)
        background = 1.0

        # Estimate dip_depth (right peak depth) using a coarse grid.
        # We use compute_vectorized_many with background=0 and dip_depth=1.
        # The max dip depth is then 1.0 / max_dip_observed.
        xs = np.linspace(frequency - split - 0.1, frequency + split + 0.1, 200)
        samples = VoigtZeemanSpectrumSamples(
            frequency=np.array([frequency], dtype=FLOAT_DTYPE),
            fwhm_total=np.array([fwhm_total], dtype=FLOAT_DTYPE),
            lorentz_frac=np.array([lorentz_frac], dtype=FLOAT_DTYPE),
            split=np.array([split], dtype=FLOAT_DTYPE),
            k_np=np.array([k_np], dtype=FLOAT_DTYPE),
            dip_depth=np.array([1.0], dtype=FLOAT_DTYPE),
            background=np.array([0.0], dtype=FLOAT_DTYPE),
        )
        # compute_vectorized_many returns bg - dips.
        # Here bg=0, so it returns -dips.
        res = self.compute_vectorized_many(xs, samples)
        max_dip = -float(res.min())
        dip_depth = 1.0 / max_dip if max_dip > 1e-6 else 1.0

        return VoigtZeemanSpectrum(
            frequency=frequency,
            fwhm_total=fwhm_total,
            lorentz_frac=lorentz_frac,
            split=split,
            k_np=k_np,
            dip_depth=dip_depth,
            background=background,
        )
