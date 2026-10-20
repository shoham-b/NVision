"""Single Gaussian peak model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.numba_kernels import gaussian_peak_value
from nvision.spectra.signal import SignalModel
from nvision.spectra.spec import GenericParamSpec


@dataclass(frozen=True)
class GaussianSpectrum:
    zero_field_freq: float
    sigma: float
    contrast: float
    background: float


@dataclass(frozen=True)
class GaussianSpectrumSamples:
    zero_field_freq: np.ndarray
    sigma: np.ndarray
    contrast: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class GaussianSpectrumUncertainty:
    zero_field_freq: float
    sigma: float
    contrast: float
    background: float


class _GaussianSpec(GenericParamSpec[GaussianSpectrum, GaussianSpectrumSamples, GaussianSpectrumUncertainty]):
    params_cls = GaussianSpectrum
    samples_cls = GaussianSpectrumSamples
    uncertainty_cls = GaussianSpectrumUncertainty


class GaussianModel(SignalModel[GaussianSpectrum, GaussianSpectrumSamples, GaussianSpectrumUncertainty]):
    """Single Gaussian peak model.

    Signal form:
        f(x) = background + contrast * exp(-0.5 * ((x - zero_field_freq) / sigma)^2)

    Parameters
    ----------
    zero_field_freq : float
        Peak center
    sigma : float
        Standard deviation
    contrast : float
        Peak contrast
    background : float
        Background level
    """

    _SPEC = _GaussianSpec()

    @property
    def spec(self) -> _GaussianSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("sigma", "contrast")

    def compute(self, x: float, params: GaussianSpectrum) -> float:
        return float(
            gaussian_peak_value(
                x,
                params.zero_field_freq,
                params.sigma,
                params.contrast,
                params.background,
            )
        )

    def compute_vectorized_samples(self, x: float, samples: GaussianSpectrumSamples) -> np.ndarray:
        x_f = float(x)
        zero_field_freq = np.asarray(samples.zero_field_freq, dtype=FLOAT_DTYPE)
        sig = np.asarray(samples.sigma, dtype=FLOAT_DTYPE)
        amp = np.asarray(samples.contrast, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        z = (x_f - zero_field_freq) / sig
        return (bg + amp * np.exp(-0.5 * z * z)).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: GaussianSpectrumSamples) -> np.ndarray:
        if not hasattr(samples, "zero_field_freq"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        zero_field_freq = np.asarray(samples.zero_field_freq, dtype=FLOAT_DTYPE)
        sig = np.asarray(samples.sigma, dtype=FLOAT_DTYPE)
        amp = np.asarray(samples.contrast, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        z = (xs[:, None] - zero_field_freq[None, :]) / sig[None, :]
        return (bg[None, :] + amp[None, :] * np.exp(-0.5 * z * z)).astype(FLOAT_DTYPE, copy=False)
