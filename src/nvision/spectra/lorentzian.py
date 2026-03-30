"""Single Lorentzian peak model."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.numba_kernels import lorentzian_peak_value
from nvision.spectra.signal import ParamSpec, SignalModel


@dataclass(frozen=True)
class LorentzianParams:
    frequency: float
    linewidth: float
    dip_depth: float
    background: float

    @property
    def physical_amplitude(self) -> float:
        """Physical Hz² amplitude (numerator): dip_depth * linewidth²."""
        return self.dip_depth * self.linewidth**2


@dataclass(frozen=True)
class LorentzianSampleParams:
    frequency: np.ndarray
    linewidth: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class LorentzianUncertaintyParams:
    frequency: float
    linewidth: float
    dip_depth: float
    background: float


class _LorentzianSpec(ParamSpec[LorentzianParams, LorentzianSampleParams, LorentzianUncertaintyParams]):
    @property
    def names(self) -> tuple[str, ...]:
        return ("frequency", "linewidth", "dip_depth", "background")

    @property
    def dim(self) -> int:
        return 4

    def unpack_params(self, values) -> LorentzianParams:
        f, w, a, b = values
        return LorentzianParams(float(f), float(w), float(a), float(b))

    def pack_params(self, params: LorentzianParams) -> tuple[float, ...]:
        return (float(params.frequency), float(params.linewidth), float(params.dip_depth), float(params.background))

    def unpack_uncertainty(self, values) -> LorentzianUncertaintyParams:
        f, w, a, b = values
        return LorentzianUncertaintyParams(float(f), float(w), float(a), float(b))

    def pack_uncertainty(self, u: LorentzianUncertaintyParams) -> tuple[float, ...]:
        return (float(u.frequency), float(u.linewidth), float(u.dip_depth), float(u.background))

    def unpack_samples(self, arrays_in_order) -> LorentzianSampleParams:
        f, w, a, b = arrays_in_order
        return LorentzianSampleParams(
            frequency=np.asarray(f, dtype=FLOAT_DTYPE),
            linewidth=np.asarray(w, dtype=FLOAT_DTYPE),
            dip_depth=np.asarray(a, dtype=FLOAT_DTYPE),
            background=np.asarray(b, dtype=FLOAT_DTYPE),
        )

    def pack_samples(self, samples: LorentzianSampleParams) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(samples.frequency, dtype=FLOAT_DTYPE),
            np.asarray(samples.linewidth, dtype=FLOAT_DTYPE),
            np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE),
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
    dip_depth : float
        Normalized peak depth (0 to 1). Peak height = dip_depth.
    background : float
        Baseline level (max signal)
    """

    _SPEC = _LorentzianSpec()

    @property
    def spec(self) -> _LorentzianSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("linewidth", "dip_depth")

    def compute(self, x: float, params: LorentzianParams) -> float:
        return float(
            lorentzian_peak_value(
                x,
                params.frequency,
                params.linewidth,
                params.dip_depth,
                params.background,
            )
        )

    def compute_vectorized_samples(self, x: float, samples: LorentzianSampleParams) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.linewidth, dtype=FLOAT_DTYPE)
        depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        denom = (x_f - freq) * (x_f - freq) + lw * lw
        return (bg - (depth * lw * lw) / denom).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: LorentzianSampleParams) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.linewidth, dtype=FLOAT_DTYPE)
        depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        denom = (xs[:, None] - freq[None, :]) ** 2 + lw[None, :] ** 2
        return (bg[None, :] - (depth[None, :] * lw[None, :] ** 2) / denom).astype(FLOAT_DTYPE, copy=False)

    def sample_params(self, rng: random.Random) -> LorentzianParams:
        """Sample parameters that keep the signal within [0, 1]."""
        frequency = rng.uniform(0.1, 0.9)
        linewidth = rng.uniform(0.03, 0.12)
        dip_depth = rng.uniform(0.3, 0.85)
        background = 1.0
        return LorentzianParams(frequency=frequency, linewidth=linewidth, dip_depth=dip_depth, background=background)
