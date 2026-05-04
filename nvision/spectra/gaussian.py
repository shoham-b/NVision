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
    frequency: float
    sigma: float
    dip_depth: float
    background: float


@dataclass(frozen=True)
class GaussianSpectrumSamples:
    frequency: np.ndarray
    sigma: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class GaussianSpectrumUncertainty:
    frequency: float
    sigma: float
    dip_depth: float
    background: float


class _GaussianSpec(GenericParamSpec[GaussianSpectrum, GaussianSpectrumSamples, GaussianSpectrumUncertainty]):
    params_cls = GaussianSpectrum
    samples_cls = GaussianSpectrumSamples
    uncertainty_cls = GaussianSpectrumUncertainty


class GaussianModel(SignalModel[GaussianSpectrum, GaussianSpectrumSamples, GaussianSpectrumUncertainty]):
    """Single Gaussian peak model.

    Signal form:
        f(x) = background + dip_depth * exp(-0.5 * ((x - frequency) / sigma)^2)

    Parameters
    ----------
    frequency : float
        Peak center
    sigma : float
        Standard deviation
    dip_depth : float
        Peak dip_depth
    background : float
        Background level
    """

    _SPEC = _GaussianSpec()

    @property
    def spec(self) -> _GaussianSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("sigma", "dip_depth")

    def compute(self, x: float, params: GaussianSpectrum) -> float:
        return float(
            gaussian_peak_value(
                x,
                params.frequency,
                params.sigma,
                params.dip_depth,
                params.background,
            )
        )

    def compute_vectorized_samples(self, x: float, samples: GaussianSpectrumSamples) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        sig = np.asarray(samples.sigma, dtype=FLOAT_DTYPE)
        amp = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        z = (x_f - freq) / sig
        return (bg + amp * np.exp(-0.5 * z * z)).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: GaussianSpectrumSamples) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        sig = np.asarray(samples.sigma, dtype=FLOAT_DTYPE)
        amp = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        z = (xs[:, None] - freq[None, :]) / sig[None, :]
        return (bg[None, :] + amp[None, :] * np.exp(-0.5 * z * z)).astype(FLOAT_DTYPE, copy=False)
