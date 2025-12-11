from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from nvision.sim.locs.nv_center._jit_kernels import _calculate_utility_grid_jit

from .sequential_bayesian_locator import (
    NVCenterSequentialBayesianLocatorSingle,
)

log = logging.getLogger(__name__)


@dataclass
class ProjectBayesianLocator(NVCenterSequentialBayesianLocatorSingle):
    """
    Bayesian Locator based on the 'Project Book Submission'.

    Uses a variance-based utility function and probabilistic frequency selection.
    """

    pickiness: float = 5.0

    def _calculate_utility_grid(self) -> np.ndarray:
        """
        Calculate utility for all frequencies in the grid.

        Utility(f) = Var_Params(f) / Var_Noise
        Var_Params(f) = Variance of predicted signal at f given current posterior.
        """
        # Sample parameter sets from posterior
        # We vary f0 based on freq_posterior, keep others fixed
        n_samples = min(self.n_monte_carlo, 100)

        if np.all(self.freq_posterior == 0):
            # Fallback if posterior is invalid
            return np.ones_like(self.freq_grid)

        # Prepare current estimates array for JIT
        current_estimates_array = np.array(
            [
                self.current_estimates["linewidth"],
                self.current_estimates["amplitude"],
                self.current_estimates["background"],
            ],
            dtype=np.float64,
        )

        sigma = 0.05  # Placeholder, same as in base class

        utility = _calculate_utility_grid_jit(
            self.freq_grid, self.freq_posterior, n_samples, current_estimates_array, sigma
        )
        return utility

    def _optimize_acquisition(self, domain: tuple[float, float]) -> tuple[float, float]:
        """
        Select next frequency probabilistically based on utility.
        """
        utility_grid = self._calculate_utility_grid()

        # Avoid numerical issues with utility
        utility_grid = np.maximum(utility_grid, 1e-9)

        # Calculate selection probability: P(f) ~ Utility(f)^Pickiness
        # Use log space for stability
        log_prob = self.pickiness * np.log(utility_grid)
        log_prob -= logsumexp(log_prob)
        prob = np.exp(log_prob)

        # Sample from the grid
        idx = np.random.choice(self.grid_resolution, p=prob)
        selected_freq = self.freq_grid[idx]

        return selected_freq, utility_grid[idx]
