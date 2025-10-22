from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

from .core import DataBatch


@dataclass
class RabiGenerator:
    """Generates a simple sinusoidal Rabi oscillation signal.

    signal_values(time_points) = offset + amplitude * sin(2π f time_points + phi)
    """

    n_points: int = 200
    duration: float = 5.0  # seconds (arbitrary units)
    amplitude: float | None = None  # randomized if None
    frequency: float | None = None  # Hz (cycles per duration unit), randomized if None
    phase: float | None = None  # radians
    offset: float | None = None  # baseline

    def generate(self, rng: random.Random) -> DataBatch:
        amp = self.amplitude if self.amplitude is not None else rng.uniform(0.2, 1.0)
        freq = self.frequency if self.frequency is not None else rng.uniform(0.5, 2.0)
        phi = self.phase if self.phase is not None else rng.uniform(0.0, 2 * math.pi)
        off = self.offset if self.offset is not None else rng.uniform(0.0, 1.0)
        dt = self.duration / max(self.n_points - 1, 1)
        t = [i * dt for i in range(self.n_points)]
        y = [off + amp * math.sin(2 * math.pi * freq * ti + phi) for ti in t]
        meta = {"amplitude": amp, "frequency": freq, "phase": phi, "offset": off}
        return DataBatch(time_points=t, signal_values=y, meta=meta)


@dataclass
class T1Generator:
    """Generates a simple exponential decay (T1-like) signal.

    signal_values(time_points) = offset + A * exp(-time_points / tau)
    """

    n_points: int = 200
    duration: float = 5.0
    A: float | None = None
    tau: float | None = None
    offset: float | None = None

    def generate(self, rng: random.Random) -> DataBatch:
        A = self.A if self.A is not None else rng.uniform(0.2, 1.2)
        tau = self.tau if self.tau is not None else rng.uniform(0.5, 3.0)
        off = self.offset if self.offset is not None else rng.uniform(0.0, 0.5)
        dt = self.duration / max(self.n_points - 1, 1)
        t = [i * dt for i in range(self.n_points)]
        y = [off + A * math.exp(-ti / tau) for ti in t]
        meta = {"A": A, "tau": tau, "offset": off}
        return DataBatch(time_points=t, signal_values=y, meta=meta)
