"""Base locator interface for stateless architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl


@dataclass
class Observation:
    """Single measurement observation."""

    x: float
    signal_value: float


class Locator(ABC):
    """Stateless locator interface.

    The locator is fully stateless - same instance is safe to reuse across repeats.
    The locator knows the signal model shape (e.g. Lorentzian, Voigt-Zeeman)
    but not the parameters (where the peak is).

    The acquisition strategy (EIG, grid, golden section) and stopping rule
    are encapsulated inside the concrete locator implementation.

    All state is contained in the history DataFrame passed to each method.
    """

    @abstractmethod
    def next(self, history: pl.DataFrame) -> float:
        """Propose next measurement point.

        Parameters
        ----------
        history : pl.DataFrame
            DataFrame with columns ['x', 'signal_value'] containing all past
            measurements for this repeat. Empty on first call.

        Returns
        -------
        float
            Next x position to measure (normalized to [0, 1])
        """
        pass

    @abstractmethod
    def done(self, history: pl.DataFrame) -> bool:
        """Check if this repeat is complete.

        Parameters
        ----------
        history : pl.DataFrame
            DataFrame with columns ['x', 'signal_value'] containing all past
            measurements for this repeat.

        Returns
        -------
        bool
            True if no more measurements needed
        """
        pass

    @abstractmethod
    def result(self, history: pl.DataFrame) -> dict[str, float]:
        """Extract final results from completed repeat.

        Parameters
        ----------
        history : pl.DataFrame
            DataFrame with columns ['x', 'signal_value'] containing all
            measurements for this repeat.

        Returns
        -------
        dict[str, float]
            Final estimates (e.g., {'peak_x': 0.5, 'peak_signal': 0.95})
            Note: peak_x should be in normalized [0, 1] space
        """
        pass
