from __future__ import annotations

import random
from collections.abc import Callable
from typing import Protocol


class PeakManufacturer(Protocol):
    def build_peak(
        self,
        center: float,
        base: float,
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, float]]: ...


class SeriesManufacturer(Protocol):
    def build_addition(
        self,
        time_points: list[float],
        center: float,
        base: float,
        rng: random.Random,
    ) -> tuple[list[float], dict[str, float]]: ...
