"""Single Lorentzian peak model."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.numba_kernels import lorentzian_peak_value
from nvision.spectra.signal import SignalModel
from nvision.spectra.spec import GenericParamSpec


@dataclass(frozen=True)
class LorentzianSpectrum:
    zero_field_freq: float
    hwhm: float
    contrast: float
    background: float

    @property
    def physical_amplitude(self) -> float:
        """Physical Hz² amplitude (numerator): contrast * hwhm²."""
        return self.contrast * self.hwhm**2


@dataclass(frozen=True)
class LorentzianSpectrumSamples:
    zero_field_freq: np.ndarray
    hwhm: np.ndarray
    contrast: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class LorentzianSpectrumUncertainty:
    zero_field_freq: float
    hwhm: float
    contrast: float
    background: float


class _LorentzianSpec(GenericParamSpec[LorentzianSpectrum, LorentzianSpectrumSamples, LorentzianSpectrumUncertainty]):
    params_cls = LorentzianSpectrum
    samples_cls = LorentzianSpectrumSamples
    uncertainty_cls = LorentzianSpectrumUncertainty


class LorentzianModel(SignalModel[LorentzianSpectrum, LorentzianSpectrumSamples, LorentzianSpectrumUncertainty]):
    """Single Lorentzian peak model.

    Signal form:
        f(x) = background - amplitude / ((x - zero_field_freq)^2 + hwhm^2)

    The amplitude parameter has units of [signal * frequency^2] so that
    dip depth = amplitude / hwhm².  Use ``sample_params`` to get a set
    of parameters that keeps the signal in [0, 1].

    Parameters
    ----------
    zero_field_freq : float
        Peak center, in [0, 1] normalized units
    hwhm : float
        Half-width at half-maximum (HWHM)
    contrast : float
        Normalized peak depth (0 to 1). Peak height = contrast.
    background : float
        Baseline level (max signal)
    """

    _SPEC = _LorentzianSpec()

    @property
    def spec(self) -> _LorentzianSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("hwhm", "contrast")

    def compute(self, x: float, params: LorentzianSpectrum) -> float:
        return float(
            lorentzian_peak_value(
                x,
                params.zero_field_freq,
                params.hwhm,
                params.contrast,
                params.background,
            )
        )

    def compute_vectorized_samples(self, x: float, samples: LorentzianSpectrumSamples) -> np.ndarray:
        x_f = float(x)
        zero_field_freq = np.asarray(samples.zero_field_freq, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.hwhm, dtype=FLOAT_DTYPE)
        depth = np.asarray(samples.contrast, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        denom = (x_f - zero_field_freq) * (x_f - zero_field_freq) + lw * lw
        return (bg - (depth * lw * lw) / denom).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: LorentzianSpectrumSamples) -> np.ndarray:
        if not hasattr(samples, "zero_field_freq"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        zero_field_freq = np.asarray(samples.zero_field_freq, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.hwhm, dtype=FLOAT_DTYPE)
        depth = np.asarray(samples.contrast, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        denom = (xs[:, None] - zero_field_freq[None, :]) ** 2 + lw[None, :] ** 2
        return (bg[None, :] - (depth[None, :] * lw[None, :] ** 2) / denom).astype(FLOAT_DTYPE, copy=False)

    def sample_params(self, rng: random.Random) -> LorentzianSpectrum:
        """Sample parameters that keep the signal within [0, 1]."""
        zero_field_freq = rng.uniform(0.1, 0.9)
        hwhm = rng.uniform(0.03, 0.12)
        contrast = rng.uniform(0.3, 0.85)
        background = 1.0
        return LorentzianSpectrum(zero_field_freq=zero_field_freq, hwhm=hwhm, contrast=contrast, background=background)
