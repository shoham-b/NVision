from __future__ import annotations

import random
from collections.abc import Callable
from typing import Protocol


class PeakManufacturer(Protocol):
    """Factory of a single peak signal.

    The metadata dictionary may include an ``"inference"`` key containing
    structured priors for Bayesian locators. Implementations should document the
    schema they provide."""

    def build_peak(
        self,
        center: float,
        base: float,
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, object]]: ...


class SeriesManufacturer(Protocol):
    """Factory of a series of peak signals.

    The metadata dictionary may include an ``"inference"`` key containing
    structured priors for Bayesian locators. Implementations should document the
    schema they provide."""

    def build_addition(
        self,
        time_points: list[float],
        center: float,
        base: float,
        rng: random.Random,
    ) -> tuple[list[float], dict[str, object]]: ...
