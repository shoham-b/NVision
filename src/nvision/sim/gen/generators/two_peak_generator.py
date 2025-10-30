from __future__ import annotations

import random
from dataclasses import dataclass

from nvision.sim.gen._protocols import PeakManufacturer
from nvision.sim.gen.generators.multi_peak_generator import MultiPeakGenerator
from nvision.sim.locs import ScanBatch


@dataclass
class TwoPeakGenerator:
    x_min: float = 0.0
    x_max: float = 1.0
    base: float = 0.0
    min_sep_frac: float = 0.1
    manufacturer_left: PeakManufacturer | None = None
    manufacturer_right: PeakManufacturer | None = None

    def generate(self, rng: random.Random) -> ScanBatch:
        if self.manufacturer_left is None or self.manufacturer_right is None:
            raise ValueError("TwoPeakGenerator requires manufacturer_left and manufacturer_right")
        mg = MultiPeakGenerator(
            x_min=self.x_min,
            x_max=self.x_max,
            count=2,
            base=self.base,
            min_sep_frac=self.min_sep_frac,
            manufacturers=[self.manufacturer_left, self.manufacturer_right],
        )
        return mg.generate(rng)
