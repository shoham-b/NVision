"""Single Gaussian peak model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.signal.dtypes import FLOAT_DTYPE
from nvision.signal.numba_kernels import gaussian_peak_value
from nvision.signal.signal import ParamSpec, SignalModel


@dataclass(frozen=True)
class GaussianParams:
    frequency: float
    sigma: float
    amplitude: float
    background: float


@dataclass(frozen=True)
class GaussianSampleParams:
    frequency: np.ndarray
    sigma: np.ndarray
    amplitude: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class GaussianUncertaintyParams:
    frequency: float
    sigma: float
    amplitude: float
    background: float


class _GaussianSpec(ParamSpec[GaussianParams, GaussianSampleParams, GaussianUncertaintyParams]):
    @property
    def names(self) -> tuple[str, ...]:
        return ("frequency", "sigma", "amplitude", "background")

    @property
    def dim(self) -> int:
        return 4

    def unpack_params(self, values) -> GaussianParams:
        f, s, a, b = values
        return GaussianParams(float(f), float(s), float(a), float(b))

    def pack_params(self, params: GaussianParams) -> tuple[float, ...]:
        return (float(params.frequency), float(params.sigma), float(params.amplitude), float(params.background))

    def unpack_uncertainty(self, values) -> GaussianUncertaintyParams:
        f, s, a, b = values
        return GaussianUncertaintyParams(float(f), float(s), float(a), float(b))

    def pack_uncertainty(self, u: GaussianUncertaintyParams) -> tuple[float, ...]:
        return (float(u.frequency), float(u.sigma), float(u.amplitude), float(u.background))

    def unpack_samples(self, arrays_in_order) -> GaussianSampleParams:
        f, s, a, b = arrays_in_order
        return GaussianSampleParams(
            frequency=np.asarray(f, dtype=FLOAT_DTYPE),
            sigma=np.asarray(s, dtype=FLOAT_DTYPE),
            amplitude=np.asarray(a, dtype=FLOAT_DTYPE),
            background=np.asarray(b, dtype=FLOAT_DTYPE),
        )

    def pack_samples(self, samples: GaussianSampleParams) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(samples.frequency, dtype=FLOAT_DTYPE),
            np.asarray(samples.sigma, dtype=FLOAT_DTYPE),
            np.asarray(samples.amplitude, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
        )


class GaussianModel(SignalModel[GaussianParams, GaussianSampleParams, GaussianUncertaintyParams]):
    """Single Gaussian peak model.

    Signal form:
        f(x) = background + amplitude * exp(-0.5 * ((x - frequency) / sigma)^2)

    Parameters
    ----------
    frequency : float
        Peak center
    sigma : float
        Standard deviation
    amplitude : float
        Peak amplitude
    background : float
        Background level
    """

    _SPEC = _GaussianSpec()

    @property
    def spec(self) -> _GaussianSpec:
        return self._SPEC

    def compute(self, x: float, params: GaussianParams) -> float:
        return float(
            gaussian_peak_value(
                x,
                params.frequency,
                params.sigma,
                params.amplitude,
                params.background,
            )
        )

    def compute_vectorized_samples(self, x: float, samples: GaussianSampleParams) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        sig = np.asarray(samples.sigma, dtype=FLOAT_DTYPE)
        amp = np.asarray(samples.amplitude, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        z = (x_f - freq) / sig
        return (bg + amp * np.exp(-0.5 * z * z)).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: GaussianSampleParams) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        sig = np.asarray(samples.sigma, dtype=FLOAT_DTYPE)
        amp = np.asarray(samples.amplitude, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        z = (xs[:, None] - freq[None, :]) / sig[None, :]
        return (bg[None, :] + amp[None, :] * np.exp(-0.5 * z * z)).astype(FLOAT_DTYPE, copy=False)
