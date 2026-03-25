"""Single Lorentzian peak model."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from nvision.signal.dtypes import FLOAT_DTYPE
from nvision.signal.numba_kernels import lorentzian_peak_value
from nvision.signal.signal import ParamSpec, SignalModel


@dataclass(frozen=True)
class LorentzianParams:
    frequency: float
    linewidth: float
    amplitude: float
    background: float


@dataclass(frozen=True)
class LorentzianSampleParams:
    frequency: np.ndarray
    linewidth: np.ndarray
    amplitude: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class LorentzianUncertaintyParams:
    frequency: float
    linewidth: float
    amplitude: float
    background: float


class _LorentzianSpec(ParamSpec[LorentzianParams, LorentzianSampleParams, LorentzianUncertaintyParams]):
    @property
    def names(self) -> tuple[str, ...]:
        return ("frequency", "linewidth", "amplitude", "background")

    @property
    def dim(self) -> int:
        return 4

    def unpack_params(self, values) -> LorentzianParams:
        f, w, a, b = values
        return LorentzianParams(float(f), float(w), float(a), float(b))

    def pack_params(self, params: LorentzianParams) -> tuple[float, ...]:
        return (float(params.frequency), float(params.linewidth), float(params.amplitude), float(params.background))

    def unpack_uncertainty(self, values) -> LorentzianUncertaintyParams:
        f, w, a, b = values
        return LorentzianUncertaintyParams(float(f), float(w), float(a), float(b))

    def pack_uncertainty(self, u: LorentzianUncertaintyParams) -> tuple[float, ...]:
        return (float(u.frequency), float(u.linewidth), float(u.amplitude), float(u.background))

    def unpack_samples(self, arrays_in_order) -> LorentzianSampleParams:
        f, w, a, b = arrays_in_order
        return LorentzianSampleParams(
            frequency=np.asarray(f, dtype=FLOAT_DTYPE),
            linewidth=np.asarray(w, dtype=FLOAT_DTYPE),
            amplitude=np.asarray(a, dtype=FLOAT_DTYPE),
            background=np.asarray(b, dtype=FLOAT_DTYPE),
        )

    def pack_samples(self, samples: LorentzianSampleParams) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(samples.frequency, dtype=FLOAT_DTYPE),
            np.asarray(samples.linewidth, dtype=FLOAT_DTYPE),
            np.asarray(samples.amplitude, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
        )


class LorentzianModel(SignalModel[LorentzianParams, LorentzianSampleParams, LorentzianUncertaintyParams]):
    """Single Lorentzian peak model.

    Signal form:
        f(x) = background - amplitude / ((x - frequency)^2 + linewidth^2)

    The amplitude parameter has units of [signal * frequency^2] so that
    dip depth = amplitude / linewidth².  Use ``sample_params`` to get a set
    of parameters that keeps the signal in [0, 1].

    Parameters
    ----------
    frequency : float
        Peak center, in [0, 1] normalized units
    linewidth : float
        Half-width at half-maximum (HWHM)
    amplitude : float
        = dip_depth * linewidth^2 (not dip depth directly)
    background : float
        Baseline level (max signal)
    """

    _SPEC = _LorentzianSpec()

    @property
    def spec(self) -> _LorentzianSpec:
        return self._SPEC

    def compute(self, x: float, params: LorentzianParams) -> float:
        return float(
            lorentzian_peak_value(
                x,
                params.frequency,
                params.linewidth,
                params.amplitude,
                params.background,
            )
        )

    def compute_vectorized_samples(self, x: float, samples: LorentzianSampleParams) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.linewidth, dtype=FLOAT_DTYPE)
        amp = np.asarray(samples.amplitude, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        denom = (x_f - freq) * (x_f - freq) + lw * lw
        return (bg - amp / denom).astype(FLOAT_DTYPE, copy=False)

    def sample_params(self, rng: random.Random) -> LorentzianParams:
        """Sample parameters that keep the signal within [0, 1]."""
        frequency = rng.uniform(0.1, 0.9)
        linewidth = rng.uniform(0.03, 0.12)
        depth = rng.uniform(0.3, 0.85)
        amplitude = depth * linewidth**2
        background = 1.0
        return LorentzianParams(frequency=frequency, linewidth=linewidth, amplitude=amplitude, background=background)
