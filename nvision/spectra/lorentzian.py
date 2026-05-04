"""Single Lorentzian peak model."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.numba_kernels import lorentzian_peak_value
from nvision.spectra.spec import GenericParamSpec
from nvision.spectra.signal import SignalModel


@dataclass(frozen=True)
class LorentzianSpectrum:
    frequency: float
    linewidth: float
    dip_depth: float
    background: float

    @property
    def physical_amplitude(self) -> float:
        """Physical Hz² amplitude (numerator): dip_depth * linewidth²."""
        return self.dip_depth * self.linewidth**2


@dataclass(frozen=True)
class LorentzianSpectrumSamples:
    frequency: np.ndarray
    linewidth: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class LorentzianSpectrumUncertainty:
    frequency: float
    linewidth: float
    dip_depth: float
    background: float


class _LorentzianSpec(GenericParamSpec[LorentzianSpectrum, LorentzianSpectrumSamples, LorentzianSpectrumUncertainty]):
    params_cls = LorentzianSpectrum
    samples_cls = LorentzianSpectrumSamples
    uncertainty_cls = LorentzianSpectrumUncertainty


class LorentzianModel(SignalModel[LorentzianSpectrum, LorentzianSpectrumSamples, LorentzianSpectrumUncertainty]):
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

    def compute(self, x: float, params: LorentzianSpectrum) -> float:
        return float(
            lorentzian_peak_value(
                x,
                params.frequency,
                params.linewidth,
                params.dip_depth,
                params.background,
            )
        )

    def compute_vectorized_samples(self, x: float, samples: LorentzianSpectrumSamples) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.linewidth, dtype=FLOAT_DTYPE)
        depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        denom = (x_f - freq) * (x_f - freq) + lw * lw
        return (bg - (depth * lw * lw) / denom).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: LorentzianSpectrumSamples) -> np.ndarray:
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

    def sample_params(self, rng: random.Random) -> LorentzianSpectrum:
        """Sample parameters that keep the signal within [0, 1]."""
        frequency = rng.uniform(0.1, 0.9)
        linewidth = rng.uniform(0.03, 0.12)
        dip_depth = rng.uniform(0.3, 0.85)
        background = 1.0
        return LorentzianSpectrum(frequency=frequency, linewidth=linewidth, dip_depth=dip_depth, background=background)
