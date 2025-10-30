from __future__ import annotations

import random
from dataclasses import dataclass

from nvision.sim.gen._protocols import PeakManufacturer
from nvision.sim.locs import ScanBatch


@dataclass
class SymmetricTwoPeakGenerator:
    x_min: float = 0.0
    x_max: float = 1.0
    center: float = 0.5
    sep_frac: float = 0.1
    base: float = 0.0
    manufacturers: tuple[PeakManufacturer, PeakManufacturer] | None = None

    def __post_init__(self) -> None:
        if self.manufacturers is None:
            raise ValueError("SymmetricTwoPeakGenerator requires manufacturers=(left, right)")
        if not (self.x_min <= self.center <= self.x_max):
            raise ValueError("center must lie within [x_min, x_max]")
        if self.sep_frac <= 0.0:
            raise ValueError("sep_frac must be positive")

    def generate(self, rng: random.Random) -> ScanBatch:
        width = self.x_max - self.x_min
        delta = 0.5 * self.sep_frac * width
        max_delta = min(self.center - self.x_min, self.x_max - self.center)
        if delta > max_delta:
            raise ValueError("sep_frac too large for given center and domain")
        x1 = self.center - delta
        x2 = self.center + delta
        manuf_left, manuf_right = self.manufacturers
        f1, _ = manuf_left.build_peak(x1, self.base, self.x_min, self.x_max, rng)
        f2, _ = manuf_right.build_peak(x2, self.base, self.x_min, self.x_max, rng)

        def f(x: float) -> float:
            return f1(x) + f2(x) - self.base

        return ScanBatch(
            x_min=self.x_min,
            x_max=self.x_max,
            truth_positions=[x1, x2],
            signal=f,
            meta={"center": self.center, "sep_frac": self.sep_frac, "base": self.base},
        )
