from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core import DataBatch, OverTimeNoise, OverVoltageNoise


@dataclass
class OverVoltageNoises:
    """Group of per-measurement (intrinsic) noises applied in sequence.

    Example components: OverVoltageGaussianNoise, OverVoltagePoissonNoise, OverVoltageOutlierSpikes.
    """

    parts: Sequence[OverVoltageNoise] | None = None
    _parts: list[OverVoltageNoise] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.parts:
            self._parts = list(self.parts)

    def add(self, model: OverVoltageNoise) -> None:
        self._parts.append(model)

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        out = data
        for part in self._parts:
            out = part.apply(out, rng)
        return out


@dataclass
class OverTimeNoises:
    """Group of cumulative/system noises applied in sequence.

    Example components: OverTimeRandomWalkNoise, OverTimeDriftNoise.
    """

    parts: Sequence[OverTimeNoise] | None = None
    _parts: list[OverTimeNoise] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.parts:
            self._parts = list(self.parts)

    def add(self, model: OverTimeNoise) -> None:
        self._parts.append(model)

    def reset(self) -> None:
        """Reset internal state of parts that expose reset()."""
        for p in self._parts:
            reset = getattr(p, "reset", None)
            if callable(reset):
                reset()

    def apply(self, signal_value: float, rng: random.Random) -> float:
        out = signal_value
        for part in self._parts:
            out = part.apply(out, rng)
        return out
