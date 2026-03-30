"""Exponential decay model (typed generic signal)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.signal.dtypes import FLOAT_DTYPE
from nvision.signal.signal import ParamSpec, SignalModel


@dataclass(frozen=True)
class ExponentialDecayParams:
    decay_rate: float
    amplitude: float
    background: float


@dataclass(frozen=True)
class ExponentialDecaySampleParams:
    decay_rate: np.ndarray
    amplitude: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class ExponentialDecayUncertaintyParams:
    decay_rate: float
    amplitude: float
    background: float


class _ExponentialDecaySpec(
    ParamSpec[ExponentialDecayParams, ExponentialDecaySampleParams, ExponentialDecayUncertaintyParams]
):
    @property
    def names(self) -> tuple[str, ...]:
        return ("decay_rate", "amplitude", "background")

    @property
    def dim(self) -> int:
        return 3

    def unpack_params(self, values) -> ExponentialDecayParams:
        r, a, b = values
        return ExponentialDecayParams(float(r), float(a), float(b))

    def pack_params(self, params: ExponentialDecayParams) -> tuple[float, ...]:
        return (float(params.decay_rate), float(params.amplitude), float(params.background))

    def unpack_uncertainty(self, values) -> ExponentialDecayUncertaintyParams:
        r, a, b = values
        return ExponentialDecayUncertaintyParams(float(r), float(a), float(b))

    def pack_uncertainty(self, u: ExponentialDecayUncertaintyParams) -> tuple[float, ...]:
        return (float(u.decay_rate), float(u.amplitude), float(u.background))

    def unpack_samples(self, arrays_in_order) -> ExponentialDecaySampleParams:
        r, a, b = arrays_in_order
        return ExponentialDecaySampleParams(
            decay_rate=np.asarray(r, dtype=FLOAT_DTYPE),
            amplitude=np.asarray(a, dtype=FLOAT_DTYPE),
            background=np.asarray(b, dtype=FLOAT_DTYPE),
        )

    def pack_samples(self, samples: ExponentialDecaySampleParams) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(samples.decay_rate, dtype=FLOAT_DTYPE),
            np.asarray(samples.amplitude, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
        )


class ExponentialDecayModel(
    SignalModel[ExponentialDecayParams, ExponentialDecaySampleParams, ExponentialDecayUncertaintyParams]
):
    """Exponential decay model.

    Signal form:
        f(x) = background + amplitude * exp(-x / decay_rate)

    Parameters
    ----------
    decay_rate : float
        Decay rate constant
    amplitude : float
        Peak amplitude
    background : float
        Background level
    """

    _SPEC = _ExponentialDecaySpec()

    @property
    def spec(self) -> _ExponentialDecaySpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("decay_rate", "amplitude")

    def compute(self, x: float, params: ExponentialDecayParams) -> float:
        decay = max(float(params.decay_rate), 1e-12)
        return float(params.background + params.amplitude * np.exp(-float(x) / decay))

    def compute_vectorized_samples(self, x: float, samples: ExponentialDecaySampleParams) -> np.ndarray:
        r = np.asarray(samples.decay_rate, dtype=FLOAT_DTYPE)
        a = np.asarray(samples.amplitude, dtype=FLOAT_DTYPE)
        b = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        return (b + a * np.exp(-float(x) / np.maximum(r, 1e-12))).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: ExponentialDecaySampleParams) -> np.ndarray:
        if not hasattr(samples, "decay_rate"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        r = np.asarray(samples.decay_rate, dtype=FLOAT_DTYPE)
        a = np.asarray(samples.amplitude, dtype=FLOAT_DTYPE)
        b = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        decay = np.maximum(r, 1e-12)
        return (b[None, :] + a[None, :] * np.exp(-xs[:, None] / decay[None, :])).astype(FLOAT_DTYPE, copy=False)
