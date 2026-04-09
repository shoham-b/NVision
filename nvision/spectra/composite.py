"""Composite peak model combining multiple independent signals (typed params)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.signal import ParamSpec, SignalModel


@dataclass(frozen=True)
class CompositeSpectrum:
    peaks: tuple[object, ...]


@dataclass(frozen=True)
class CompositeSpectrumSamples:
    peaks: tuple[object, ...]


@dataclass(frozen=True)
class CompositeSpectrumUncertainty:
    peaks: tuple[object, ...]


class _CompositeSpec(ParamSpec[CompositeSpectrum, CompositeSpectrumSamples, CompositeSpectrumUncertainty]):
    def __init__(self, peak_models: tuple[tuple[str, SignalModel], ...]) -> None:
        self._peak_models = peak_models
        self._models = tuple(m for _, m in peak_models)
        self._prefixes = tuple(prefix for prefix, _ in peak_models)
        self._dims = tuple(int(m.spec.dim) for m in self._models)
        self._names = tuple(
            f"{prefix}_{name}" for prefix, m in zip(self._prefixes, self._models, strict=True) for name in m.spec.names
        )

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def dim(self) -> int:
        return int(sum(self._dims))

    def unpack_params(self, values) -> CompositeSpectrum:
        peaks: list[object] = []
        off = 0
        for m, d in zip(self._models, self._dims, strict=True):
            peaks.append(m.spec.unpack_params(values[off : off + d]))
            off += d
        return CompositeSpectrum(peaks=tuple(peaks))

    def pack_params(self, params: CompositeSpectrum) -> tuple[float, ...]:
        out: list[float] = []
        for m, p in zip(self._models, params.peaks, strict=True):
            out.extend(m.spec.pack_params(p))
        return tuple(out)

    def unpack_uncertainty(self, values) -> CompositeSpectrumUncertainty:
        peaks: list[object] = []
        off = 0
        for m, d in zip(self._models, self._dims, strict=True):
            peaks.append(m.spec.unpack_uncertainty(values[off : off + d]))
            off += d
        return CompositeSpectrumUncertainty(peaks=tuple(peaks))

    def pack_uncertainty(self, u: CompositeSpectrumUncertainty) -> tuple[float, ...]:
        out: list[float] = []
        for m, pu in zip(self._models, u.peaks, strict=True):
            out.extend(m.spec.pack_uncertainty(pu))
        return tuple(out)

    def unpack_samples(self, arrays_in_order) -> CompositeSpectrumSamples:
        peaks: list[object] = []
        off = 0
        for m, d in zip(self._models, self._dims, strict=True):
            peaks.append(m.spec.unpack_samples(arrays_in_order[off : off + d]))
            off += d
        return CompositeSpectrumSamples(peaks=tuple(peaks))

    def pack_samples(self, samples: CompositeSpectrumSamples) -> tuple[np.ndarray, ...]:
        out: list[np.ndarray] = []
        for m, s in zip(self._models, samples.peaks, strict=True):
            out.extend(m.spec.pack_samples(s))
        return tuple(out)


class CompositePeakModel(SignalModel[CompositeSpectrum, CompositeSpectrumSamples, CompositeSpectrumUncertainty]):
    """Sum of multiple independent peak models with nested parameter bundles."""

    def __init__(self, peak_models: list[tuple[str, SignalModel]]):
        self.peak_models = tuple(peak_models)
        self._spec = _CompositeSpec(self.peak_models)

    @property
    def spec(self) -> _CompositeSpec:
        return self._spec

    def is_scale_parameter(self, name: str) -> bool:
        for prefix, model in self.peak_models:
            if name.startswith(prefix + "_"):
                sub_name = name[len(prefix) + 1 :]
                return model.is_scale_parameter(sub_name)
        return False

    def compute(self, x: float, params: CompositeSpectrum) -> float:
        total = 0.0
        for (_, model), p in zip(self.peak_models, params.peaks, strict=True):
            total += float(model.compute(float(x), p))
        return float(total)

    def compute_vectorized_samples(self, x: float, samples: CompositeSpectrumSamples) -> np.ndarray:
        out = None
        for m, s in zip(self.peak_models, samples.peaks, strict=True):
            _, model = m
            pred = model.compute_vectorized(float(x), s)
            out = pred if out is None else (out + pred)
        return out if out is not None else np.empty(0, dtype=FLOAT_DTYPE)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: CompositeSpectrumSamples) -> np.ndarray:
        if not hasattr(samples, "peaks"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        out = None
        for (_, model), s in zip(self.peak_models, samples.peaks, strict=True):
            pred = model.compute_vectorized_many(xs, s)
            out = pred if out is None else (out + pred)
        if out is None:
            return np.empty((len(xs), 0), dtype=FLOAT_DTYPE)
        return np.asarray(out, dtype=FLOAT_DTYPE)
