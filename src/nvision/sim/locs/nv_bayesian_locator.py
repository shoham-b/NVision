from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from nvision.models.locator import Locator
from nvision.signal.nv_center import NVCenterLorentzianModel
from nvision.signal.signal import BeliefSignal, ParameterWithPosterior


@dataclass(frozen=True)
class NVCenterBayesianConfig:
    max_steps: int = 150
    acquisition: str = "eig"  # "eig", "ucb", "random"
    n_grid_freq: int = 120
    n_grid_linewidth: int = 60
    n_grid_split: int = 60


class NVCenterBayesianLocator(Locator):
    """Bayesian locator for NV center ODMR signals using core architecture."""

    def __init__(
        self,
        belief: BeliefSignal,
        max_steps: int = 150,
        acquisition: str = "eig",
    ):
        super().__init__(belief)
        self.max_steps = max_steps
        self.acquisition = acquisition
        self.step_count = 0

    @classmethod
    def create(
        cls,
        max_steps: int = 150,
        acquisition: str = "eig",
        n_grid_freq: int = 120,
        n_grid_linewidth: int = 60,
        n_grid_split: int = 60,
        **_: object,
    ) -> NVCenterBayesianLocator:
        """Create NV center locator with priors over physical parameters."""
        model = NVCenterLorentzianModel()

        belief = BeliefSignal(
            model=model,
            parameters=[
                ParameterWithPosterior(
                    name="frequency",
                    bounds=(2.6e9, 3.1e9),
                    grid=np.linspace(2.6e9, 3.1e9, n_grid_freq),
                    posterior=np.ones(n_grid_freq) / n_grid_freq,
                ),
                ParameterWithPosterior(
                    name="linewidth",
                    bounds=(1e6, 50e6),
                    grid=np.linspace(1e6, 50e6, n_grid_linewidth),
                    posterior=np.ones(n_grid_linewidth) / n_grid_linewidth,
                ),
                ParameterWithPosterior(
                    name="split",
                    bounds=(1e6, 200e6),
                    grid=np.linspace(1e6, 200e6, n_grid_split),
                    posterior=np.ones(n_grid_split) / n_grid_split,
                ),
            ],
        )

        return cls(belief, max_steps=max_steps, acquisition=acquisition)

    def _expected_information_gain(self) -> float:
        """Placeholder EIG-style acquisition: sample by posterior variance proxy."""
        freq_param = self.belief.get_param("frequency")
        grid = freq_param.grid
        posterior = freq_param.posterior

        # Use a simple heuristic: prioritize mid-quantile points where posterior mass is concentrated.
        cdf = np.cumsum(posterior)
        target_quantiles = np.linspace(0.1, 0.9, 9)
        candidates = np.interp(target_quantiles, cdf, grid)
        return float(random.choice(candidates.tolist()))

    def _upper_confidence_bound(self) -> float:
        """Simple UCB-like heuristic over frequency grid."""
        freq_param = self.belief.get_param("frequency")
        grid = freq_param.grid
        posterior = freq_param.posterior

        # Treat posterior as "mean" proxy and add exploration bonus ~ sqrt(p*(1-p)).
        mean_like = posterior
        bonus = np.sqrt(posterior * (1.0 - posterior + 1e-12))
        ucb_score = mean_like + 2.0 * bonus
        idx = int(np.argmax(ucb_score))
        return float(grid[idx])

    def next(self) -> float:
        """Use acquisition function to propose next normalized measurement position."""
        self.step_count += 1

        if self.acquisition == "eig":
            freq = self._expected_information_gain()
        elif self.acquisition == "ucb":
            freq = self._upper_confidence_bound()
        else:
            freq = random.random() * (3.1e9 - 2.6e9) + 2.6e9

        # Convert physical frequency to normalized [0, 1] expected by core runner.
        f_min, f_max = 2.6e9, 3.1e9
        x_norm = (freq - f_min) / (f_max - f_min)
        return float(np.clip(x_norm, 0.0, 1.0))

    def done(self) -> bool:
        """Check convergence or max steps."""
        return self.step_count >= self.max_steps or self.belief.converged(threshold=0.01)

    def result(self) -> dict[str, float]:
        """Return final parameter estimates in physical units."""
        return self.belief.estimates()
