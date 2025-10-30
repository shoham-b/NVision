from __future__ import annotations

import math
import random
from collections.abc import Callable

from nvision.sim.gen._protocols import PeakManufacturer, SeriesManufacturer


class GaussianManufacturer(PeakManufacturer, SeriesManufacturer):
    """Gaussian-shaped OnePeak manufacturer."""

    def __init__(self, amplitude: float = 1.0, sigma: float = 0.08, **kwargs) -> None:
        if "A" in kwargs:
            amplitude = kwargs["A"]
        self.amplitude = amplitude
        self.sigma = sigma

    def build_peak(
        self,
        center: float,
        base: float,
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, float]]:
        def f(x: float) -> float:
            z = (x - center) / max(self.sigma, 1e-12)
            return base + self.amplitude * math.exp(-0.5 * z * z)

        return f, {"sigma": self.sigma, "amplitude": self.amplitude, "mode": "gaussian"}

    def build_addition(
        self,
        time_points: list[float],
        center: float,
        base: float,
        rng: random.Random,
    ) -> tuple[list[float], dict[str, float]]:
        if not time_points:
            return [], {"amplitude": self.amplitude, "sigma": self.sigma, "mode": "gaussian"}
        y = [
            self.amplitude * math.exp(-0.5 * ((t - center) / max(self.sigma, 1e-12)) ** 2)
            for t in time_points
        ]
        return y, {"amplitude": self.amplitude, "sigma": self.sigma, "mode": "gaussian"}
