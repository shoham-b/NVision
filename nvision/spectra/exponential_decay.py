"""Exponential decay model (typed generic signal)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.signal import ParamSpec, SignalModel


@dataclass(frozen=True)
class ExponentialDecaySpectrum:
    decay_rate: float
    dip_depth: float
    background: float


@dataclass(frozen=True)
class ExponentialDecaySpectrumSamples:
    decay_rate: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class ExponentialDecaySpectrumUncertainty:
    decay_rate: float
    dip_depth: float
    background: float


class _ExponentialDecaySpec(
    ParamSpec[ExponentialDecaySpectrum, ExponentialDecaySpectrumSamples, ExponentialDecaySpectrumUncertainty]
):
    @property
    def names(self) -> tuple[str, ...]:
        return ("decay_rate", "dip_depth", "background")

    @property
    def dim(self) -> int:
        return 3

    def unpack_params(self, values) -> ExponentialDecaySpectrum:
        r, a, b = values
        return ExponentialDecaySpectrum(float(r), float(a), float(b))

    def pack_params(self, params: ExponentialDecaySpectrum) -> tuple[float, ...]:
        return (float(params.decay_rate), float(params.dip_depth), float(params.background))

    def unpack_uncertainty(self, values) -> ExponentialDecaySpectrumUncertainty:
        r, a, b = values
        return ExponentialDecaySpectrumUncertainty(float(r), float(a), float(b))

    def pack_uncertainty(self, u: ExponentialDecaySpectrumUncertainty) -> tuple[float, ...]:
        return (float(u.decay_rate), float(u.dip_depth), float(u.background))

    def unpack_samples(self, arrays_in_order) -> ExponentialDecaySpectrumSamples:
        r, a, b = arrays_in_order
        return ExponentialDecaySpectrumSamples(
            decay_rate=np.asarray(r, dtype=FLOAT_DTYPE),
            dip_depth=np.asarray(a, dtype=FLOAT_DTYPE),
            background=np.asarray(b, dtype=FLOAT_DTYPE),
        )

    def pack_samples(self, samples: ExponentialDecaySpectrumSamples) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(samples.decay_rate, dtype=FLOAT_DTYPE),
            np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
        )


class ExponentialDecayModel(
    SignalModel[ExponentialDecaySpectrum, ExponentialDecaySpectrumSamples, ExponentialDecaySpectrumUncertainty]
):
    """Exponential decay model.

    Signal form:
        f(x) = background + dip_depth * exp(-x / decay_rate)

    Parameters
    ----------
    decay_rate : float
        Decay rate constant
    dip_depth : float
        Peak dip_depth
    background : float
        Background level
    """

    _SPEC = _ExponentialDecaySpec()

    @property
    def spec(self) -> _ExponentialDecaySpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("decay_rate", "dip_depth")

    def compute(self, x: float, params: ExponentialDecaySpectrum) -> float:
        decay = max(float(params.decay_rate), 1e-12)
        return float(params.background + params.dip_depth * np.exp(-float(x) / decay))

    def compute_vectorized_samples(self, x: float, samples: ExponentialDecaySpectrumSamples) -> np.ndarray:
        r = np.asarray(samples.decay_rate, dtype=FLOAT_DTYPE)
        a = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        b = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        return (b + a * np.exp(-float(x) / np.maximum(r, 1e-12))).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: ExponentialDecaySpectrumSamples) -> np.ndarray:
        if not hasattr(samples, "decay_rate"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        r = np.asarray(samples.decay_rate, dtype=FLOAT_DTYPE)
        a = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        b = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        decay = np.maximum(r, 1e-12)
        return (b[None, :] + a[None, :] * np.exp(-xs[:, None] / decay[None, :])).astype(FLOAT_DTYPE, copy=False)
