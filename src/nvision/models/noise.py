"""Composite noise containers used by CoreExperiment and ScalarMeasure."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from nvision.sim.batch import DataBatch, OverFrequencyNoise, OverProbeNoise


class CompositeOverFrequencyNoise(OverFrequencyNoise):
    """Applies multiple over-frequency noise models in sequence."""

    def __init__(self, parts: Sequence[OverFrequencyNoise] | None = None):
        self._parts: list[OverFrequencyNoise] = list(parts or [])

    def add(self, model: OverFrequencyNoise) -> None:
        self._parts.append(model)

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        out = data
        for part in self._parts:
            out = part.apply(out, rng)
        return out

    def noise_std(self) -> float:
        """Return the combined RMS noise standard deviation for all over-frequency components."""
        rss = sum(p.noise_std() ** 2 for p in self._parts)
        std = rss**0.5
        return std


class CompositeOverProbeNoise(OverProbeNoise):
    """Applies multiple over-probe noise models in sequence."""

    def __init__(self, parts: Sequence[OverProbeNoise] | None = None):
        self._parts: list[OverProbeNoise] = list(parts or [])

    def add(self, model: OverProbeNoise) -> None:
        self._parts.append(model)

    def apply(self, signal_value: float, rng: random.Random, locator: object = None) -> float:
        out = signal_value
        for part in self._parts:
            out = part.apply(out, rng, locator)
        return out


@dataclass(frozen=True, slots=True)
class CompositeNoise:
    """Container for both over-frequency and over-probe noise; applies both in sequence."""

    over_frequency_noise: CompositeOverFrequencyNoise | None = None
    over_probe_noise: CompositeOverProbeNoise | None = None

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        """Apply over-frequency noise (over-probe noise is not applicable to batch data)."""
        if self.over_frequency_noise is not None:
            return self.over_frequency_noise.apply(data, rng)
        return data

    def estimated_noise_std(self) -> float:
        """Return the combined RMS noise standard deviation for all over-frequency components.

        Returns
        -------
        float
            Square-root of summed squared stds from each noise component.
            Falls back to 0.05 if no noise components or all return 0.
        """
        if self.over_frequency_noise is None:
            return 0.05
        parts = getattr(self.over_frequency_noise, "_parts", [])
        rss = sum(p.noise_std() ** 2 for p in parts)
        std = rss**0.5
        return std if std > 1e-12 else 0.05
