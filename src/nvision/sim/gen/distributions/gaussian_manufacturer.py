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

        sigma_min = max(self.sigma * 0.5, 1e-6)
        sigma_max = max(self.sigma * 2.0, sigma_min + 1e-6)
        amp_min = max(self.amplitude * 0.5, 1e-6)
        amp_max = max(self.amplitude * 1.5, amp_min + 1e-6)
        bg_min = float(base)
        bg_max = float(base + max(self.amplitude, 1e-6))

        inference_meta = {
            "model": "gaussian",
            "manufacturer": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "parameter_priors": {
                "center": (float(x_min), float(x_max)),
                "sigma": (float(sigma_min), float(sigma_max)),
                "amplitude": (float(amp_min), float(amp_max)),
                "background": (bg_min, bg_max),
            },
            "parameter_defaults": {
                "sigma": float(self.sigma),
                "amplitude": float(self.amplitude),
                "background": float(base),
            },
        }

        return f, {
            "sigma": self.sigma,
            "amplitude": self.amplitude,
            "mode": "gaussian",
            "inference": inference_meta,
        }

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
