from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List

from .locators import ScanBatch


@dataclass
class OnePeakGenerator:
    """Generates a single-peaked 1D signal around a hidden location x0.

    Modes:
      - gaussian (default): base + A * exp(-0.5 * ((x - x0)/sigma)^2)
      - rabi: base + A * exp(-0.5 * ((x - x0)/sigma)^2) * 0.5 * (1 + sin(2π f (x - x0) + phi))
      - t1_decay: base + A * exp(-|x - x0| / tau)

    The hidden position x0 is randomized within [x_min, x_max]. This lets the
    locator strategies search for the location while experiencing either a
    Rabi-like oscillatory envelope or a T1-like decay envelope.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    A: float = 1.0
    base: float = 0.0
    sigma: float = 0.08
    mode: str = "gaussian"  # "gaussian" | "rabi" | "t1_decay"
    # Rabi-specific params
    rabi_freq: float = 5.0  # cycles per x-unit
    rabi_phase: float | None = None  # randomized if None
    # T1-specific params
    t1_tau: float | None = None  # decay length-scale; randomized if None

    def generate(self, rng: random.Random) -> ScanBatch:
        width = self.x_max - self.x_min
        x0 = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)
        mode = (self.mode or "gaussian").lower()

        if mode == "rabi":
            phi = self.rabi_phase if self.rabi_phase is not None else rng.uniform(0.0, 2 * math.pi)

            def f(x: float) -> float:
                z = (x - x0) / max(self.sigma, 1e-12)
                env = math.exp(-0.5 * z * z)
                osc = 0.5 * (1.0 + math.sin(2 * math.pi * self.rabi_freq * (x - x0) + phi))
                return self.base + self.A * env * osc

            meta = {"A": self.A, "sigma": self.sigma, "base": self.base, "mode": "rabi", "rabi_freq": self.rabi_freq, "rabi_phase": float(phi)}
        elif mode == "t1_decay":
            tau = self.t1_tau if self.t1_tau is not None else max(0.05 * width, 1e-6)

            def f(x: float) -> float:
                return self.base + self.A * math.exp(-abs(x - x0) / max(tau, 1e-12))

            meta = {"A": self.A, "tau": float(tau), "base": self.base, "mode": "t1_decay"}
        else:
            # Gaussian peak (default)
            def f(x: float) -> float:
                z = (x - x0) / max(self.sigma, 1e-12)
                return self.base + self.A * math.exp(-0.5 * z * z)

            meta = {"A": self.A, "sigma": self.sigma, "base": self.base, "mode": "gaussian"}

        return ScanBatch(
            x_min=self.x_min,
            x_max=self.x_max,
            truth_positions=[x0],
            signal=f,
            meta=meta,
        )


@dataclass
class TwoPeakGenerator:
    """Generates a two-peak 1D signal as a sum of Gaussians at two random locations."""

    x_min: float = 0.0
    x_max: float = 1.0
    A1: float = 1.0
    A2: float = 0.8
    base: float = 0.0
    sigma1: float = 0.06
    sigma2: float = 0.06
    min_sep_frac: float = 0.1  # enforce minimum separation between peaks

    def generate(self, rng: random.Random) -> ScanBatch:
        width = self.x_max - self.x_min
        while True:
            x1 = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)
            x2 = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)
            if abs(x1 - x2) >= self.min_sep_frac * width:
                break

        def f(x: float) -> float:
            z1 = (x - x1) / max(self.sigma1, 1e-12)
            z2 = (x - x2) / max(self.sigma2, 1e-12)
            return self.base + self.A1 * math.exp(-0.5 * z1 * z1) + self.A2 * math.exp(-0.5 * z2 * z2)

        xs = sorted([x1, x2])
        return ScanBatch(
            x_min=self.x_min,
            x_max=self.x_max,
            truth_positions=xs,
            signal=f,
            meta={
                "A1": self.A1,
                "A2": self.A2,
                "sigma1": self.sigma1,
                "sigma2": self.sigma2,
                "base": self.base,
            },
        )
