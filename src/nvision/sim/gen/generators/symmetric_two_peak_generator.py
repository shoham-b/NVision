from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from nvision.sim.gen._protocols import PeakManufacturer
from nvision.sim.locs import ScanBatch


@dataclass
class SymmetricTwoPeakGenerator:
    """Generate a scan with two symmetric peaks around a centre point."""

    x_min: float = 0.0
    x_max: float = 1.0
    center: float = 0.5
    sep_frac: float = 0.2
    base: float = 0.0
    manufacturers: Sequence[PeakManufacturer] | None = None

    def __post_init__(self) -> None:
        if self.sep_frac <= 0.0:
            raise ValueError("sep_frac must be positive")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")

    def generate(self, rng: random.Random) -> ScanBatch:
        width = self.x_max - self.x_min
        delta = 0.5 * self.sep_frac * width
        left = self.center - delta
        right = self.center + delta

        if left < self.x_min or right > self.x_max:
            raise ValueError("sep_frac too large for given centre and domain")

        if self.manufacturers is None:
            from nvision.sim.gen.distributions.gaussian_manufacturer import (
                GaussianManufacturer,
            )

            manufacturers = (
                GaussianManufacturer(amplitude=1.0, sigma=0.06),
                GaussianManufacturer(amplitude=1.0, sigma=0.06),
            )
        elif len(self.manufacturers) != 2:
            raise ValueError("manufacturers must contain exactly two PeakManufacturer instances")
        else:
            manufacturers = self.manufacturers

        peaks = []
        for peak_center, manufacturer in zip((left, right), manufacturers, strict=True):
            fn, _ = manufacturer.build_peak(
                center=peak_center,
                base=self.base,
                x_min=self.x_min,
                x_max=self.x_max,
                rng=rng,
            )
            peaks.append(fn)

        def signal(x: float) -> float:
            return sum(fn(x) for fn in peaks) - self.base

        truth_positions = [left, right]
        return ScanBatch(
            x_min=self.x_min,
            x_max=self.x_max,
            truth_positions=truth_positions,
            signal=signal,
            meta={
                "center": self.center,
                "sep_frac": self.sep_frac,
                "base": self.base,
            },
        )
