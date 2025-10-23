from __future__ import annotations

import random
from dataclasses import dataclass

from ..locators import ScanBatch
from ._protocols import PeakManufacturer


@dataclass
class OnePeakGenerator:
    x_min: float = 0.0
    x_max: float = 1.0
    base: float = 0.0
    manufacturer: PeakManufacturer | None = None

    def __post_init__(self) -> None:
        if self.manufacturer is None:
            raise ValueError("OnePeakGenerator requires a manufacturer (e.g., gaussian_peak)")

    def generate(self, rng: random.Random) -> ScanBatch:
        width = self.x_max - self.x_min
        x0 = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)
        f, extra_meta = self.manufacturer.build_peak(x0, self.base, self.x_min, self.x_max, rng)
        meta = {"base": self.base, **extra_meta}
        return ScanBatch(
            x_min=self.x_min,
            x_max=self.x_max,
            truth_positions=[x0],
            signal=f,
            meta=meta,
        )
