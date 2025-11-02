from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class ScanBatch:
    """Represents a 1-D scan, defining the domain and the ideal signal function."""

    x_min: float
    x_max: float
    signal: Callable[[float], float]
    meta: dict[str, float]
    truth_positions: list[float]


class Locator(ABC):
    """Abstract base class for peak-finding strategies in a 1-D scan."""

    @abstractmethod
    def propose_next(self, history: pl.DataFrame, scan: ScanBatch) -> float:
        """Given the history of measurements, propose the next x-coordinate to sample."""
        raise NotImplementedError

    @abstractmethod
    def should_stop(self, history: pl.DataFrame, scan: ScanBatch) -> bool:
        """Determine whether the measurement process should terminate."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self, history: pl.DataFrame, scan: ScanBatch) -> dict[str, float]:
        """Post-process the complete history to return the final estimated parameters."""
        raise NotImplementedError
