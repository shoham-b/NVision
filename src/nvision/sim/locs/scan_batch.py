from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ScanBatch:
    """Describes a 1-D domain where we will query intensities."""

    x_min: float
    x_max: float
    truth_positions: list[float]
    signal: Callable[[float], float]
    meta: dict[str, float] | None = None

    def domain_width(self) -> float:
        return self.x_max - self.x_min
