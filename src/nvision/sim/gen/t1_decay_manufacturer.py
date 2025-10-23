from __future__ import annotations

import math
import random
from collections.abc import Callable

from ._protocols import PeakManufacturer, SeriesManufacturer


class T1DecayManufacturer(PeakManufacturer, SeriesManufacturer):
    """T1-decay-shaped OnePeak manufacturer."""

    def __init__(self, amplitude: float = 1.0, t1_tau: float | None = None, **kwargs) -> None:
        if "A" in kwargs:
            amplitude = kwargs["A"]
        self.amplitude = amplitude
        self.t1_tau = t1_tau

    def build_peak(
        self,
        center: float,
        base: float,
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, float]]:
        tau = self.t1_tau if self.t1_tau is not None else max(0.05 * (x_max - x_min), 1e-6)

        def f(x: float) -> float:
            return base + self.amplitude * math.exp(-abs(x - center) / max(tau, 1e-12))

        return f, {"tau": float(tau), "amplitude": self.amplitude, "mode": "t1_decay"}

    def build_addition(
        self,
        time_points: list[float],
        center: float,
        base: float,
        rng: random.Random,
    ) -> tuple[list[float], dict[str, float]]:
        if not time_points:
            return [], {
                "amplitude": self.amplitude,
                "mode": "t1_decay",
                "tau": float(self.t1_tau or 0.0),
            }
        tau = self.t1_tau
        if tau is None:
            span = max(time_points[-1] - time_points[0], 1e-6)
            tau = max(0.05 * span, 1e-6)
        y = [self.amplitude * math.exp(-abs(t - center) / max(tau, 1e-12)) for t in time_points]
        return y, {"amplitude": self.amplitude, "tau": float(tau), "mode": "t1_decay"}
