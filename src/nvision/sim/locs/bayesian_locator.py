from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .base import Locator, ScanBatch
from .models.priors import PriorModel
from .models.protocols import ObservationModel


@dataclass
class BayesianLocator(Locator):
    """A locator that uses a Bayesian approach to propose the next point to sample."""

    priors: PriorModel
    obs_model: ObservationModel
    min_x: float
    max_x: float
    num_x_bins: int = 1000
    min_uncertainty: float = 0.01
    max_steps: int = 100
    x_key: str = "x_hat"
    uncertainty_key: str = "uncertainty"
    posterior: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.posterior = self.priors.get_probabilities(
            self.min_x,
            self.max_x,
            self.num_x_bins,
        )

    def propose_next(self, history: pl.DataFrame, scan: ScanBatch) -> float:
        """Propose the x-coordinate with the highest uncertainty (variance)."""
        if history.is_empty():
            # For the first step, just pick the middle or a random point
            return self.min_x + (self.max_x - self.min_x) / 2

        # Update posterior with the latest measurement
        last_measurement = history.tail(1)
        x_measured = last_measurement["x"].item()
        y_measured = last_measurement["signal_values"].item()
        self.posterior = self.obs_model.update_posterior(
            self.posterior,
            x_measured,
            y_measured,
            self.min_x,
            self.max_x,
            self.num_x_bins,
        )

        uncertainty = self.obs_model.get_uncertainty(self.posterior)
        # Propose the point with the highest uncertainty
        max_uncertainty_idx = uncertainty.index(max(uncertainty))
        return self.min_x + max_uncertainty_idx * (self.max_x - self.min_x) / self.num_x_bins

    def should_stop(self, history: pl.DataFrame, scan: ScanBatch) -> bool:
        """Determine whether the measurement process should terminate."""
        if len(history) >= self.max_steps:
            return True
        uncertainty = self.obs_model.get_uncertainty(self.posterior)
        return max(uncertainty) < self.min_uncertainty

    def finalize(self, history: pl.DataFrame, scan: ScanBatch) -> dict[str, float]:
        """Post-process the complete history to return the final estimated parameters."""
        # Ensure posterior is updated with all history
        self.posterior = self.priors.get_probabilities(
            self.min_x,
            self.max_x,
            self.num_x_bins,
        )
        for row in history.iter_rows(named=True):
            self.posterior = self.obs_model.update_posterior(
                self.posterior,
                row["x"],
                row["signal_values"],
                self.min_x,
                self.max_x,
                self.num_x_bins,
            )

        best_x_arg = self.posterior.index(max(self.posterior))
        best_x = self.min_x + best_x_arg * (self.max_x - self.min_x) / self.num_x_bins
        uncertainty = self.obs_model.get_uncertainty(self.posterior)
        return {self.x_key: best_x, self.uncertainty_key: max(uncertainty)}
