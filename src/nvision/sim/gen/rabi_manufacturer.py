from __future__ import annotations

import math
import random
from collections.abc import Callable

from ._protocols import PeakManufacturer, SeriesManufacturer


class RabiManufacturer(PeakManufacturer, SeriesManufacturer):
    """Rabi-shaped OnePeak manufacturer (randomized phase if unset)."""

    def __init__(
        self,
        amplitude: float = 1.0,
        sigma: float = 0.08,
        rabi_freq: float = 5.0,
        rabi_phase: float | None = None,
        **kwargs,
    ) -> None:
        if "A" in kwargs:
            amplitude = kwargs["A"]
        self.amplitude = amplitude
        self.sigma = sigma
        self.rabi_freq = rabi_freq
        self.rabi_phase = rabi_phase

    def build_peak(
        self,
        center: float,
        base: float,
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, float]]:
        phi = self.rabi_phase if self.rabi_phase is not None else rng.uniform(0.0, 2 * math.pi)

        def f(x: float) -> float:
            z = (x - center) / max(self.sigma, 1e-12)
            env = math.exp(-0.5 * z * z)
            osc = 0.5 * (1.0 + math.sin(2 * math.pi * self.rabi_freq * (x - center) + phi))
            return base + self.amplitude * env * osc

        return f, {
            "sigma": self.sigma,
            "amplitude": self.amplitude,
            "mode": "rabi",
            "rabi_freq": self.rabi_freq,
            "rabi_phase": float(phi),
        }

    def build_addition(
        self, time_points: list[float], center: float, base: float, rng: random.Random,
    ) -> tuple[list[float], dict[str, float]]:
        phi = self.rabi_phase if self.rabi_phase is not None else rng.uniform(0.0, 2 * math.pi)
        y = [
            self.amplitude
            * math.exp(-0.5 * ((t - center) / max(self.sigma, 1e-12)) ** 2)
            * 0.5
            * (1.0 + math.sin(2 * math.pi * self.rabi_freq * (t - center) + phi))
            for t in time_points
        ]
        return y, {
            "amplitude": self.amplitude,
            "sigma": self.sigma,
            "rabi_freq": self.rabi_freq,
            "rabi_phase": float(phi),
        }
