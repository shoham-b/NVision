"""Simple sweep locator using core architecture.

This demonstrates how to implement a locator using the new core architecture
with BeliefSignal and incremental Bayesian updates.
"""

from __future__ import annotations

import numpy as np

from nvision.models.locator import Locator
from nvision.signal.abstract_belief import AbstractBeliefDistribution
from nvision.signal.grid_belief import GridBeliefDistribution, GridParameter
from nvision.signal.signal import SignalModel


class BlackBoxSignalModel(SignalModel):
    """Signal model that treats parameters as a single peak position.

    Used for simple sweep locators that only estimate peak location.
    """

    @staticmethod
    def eval_black_box_signal_model(x: float, peak_x: float) -> float:
        """Placeholder (sweep measures hardware); order matches :meth:`parameter_names`."""
        return 0.0

    def compute(self, x: float, params: list) -> float:
        """Not used for black-box sweep — signal is measured, not computed."""
        v = self._param_floats_canonical(params)
        return self.eval_black_box_signal_model(x, v[0])

    def parameter_names(self) -> list[str]:
        """Return single parameter: peak position."""
        return ["peak_x"]


class SimpleSweepLocator(Locator):
    """Simple grid sweep locator using core architecture.

    Sweeps uniformly across [0, 1] normalized space and updates
    belief about peak location based on measured signal values.
    """

    def __init__(self, belief: AbstractBeliefDistribution, max_steps: int = 50):
        """Initialize sweep locator.

        Parameters
        ----------
        belief : AbstractBeliefDistribution
            Initial belief (uniform prior over peak location)
        max_steps : int
            Maximum number of sweep points
        """
        super().__init__(belief)
        self.max_steps = max_steps
        self.step_count = 0
        self.grid_positions = np.linspace(0.0, 1.0, max_steps)

        # Track best observation for peak estimation
        self.best_signal = -np.inf
        self.best_x = 0.5

    @classmethod
    def create(cls, max_steps: int = 50, n_grid: int = 100, **kwargs) -> SimpleSweepLocator:
        """Create fresh sweep locator with uniform prior.

        Parameters
        ----------
        max_steps : int
            Number of grid points to sweep
        n_grid : int
            Number of grid points for belief posterior
        **kwargs
            Additional configuration (ignored)

        Returns
        -------
        SimpleSweepLocator
            New locator with uniform prior
        """
        model = BlackBoxSignalModel()

        # Create uniform prior over [0, 1] for peak location
        belief = GridBeliefDistribution(
            model=model,
            parameters=[
                GridParameter(
                    name="peak_x",
                    bounds=(0.0, 1.0),
                    grid=np.linspace(0.0, 1.0, n_grid),
                    posterior=np.ones(n_grid) / n_grid,
                ),
            ],
        )

        return cls(belief, max_steps)

    def next(self) -> float:
        """Return next grid position to measure.

        Returns
        -------
        float
            Next position in [0, 1] normalized space
        """
        x = self.grid_positions[self.step_count]
        self.step_count += 1
        return float(x)

    def done(self) -> bool:
        """Check if sweep is complete.

        Returns
        -------
        bool
            True if all grid points have been measured
        """
        return self.step_count >= self.max_steps

    def result(self) -> dict[str, float]:
        """Extract peak position estimate.

        Returns
        -------
        dict[str, float]
            Estimated peak position in normalized space
        """
        # Return best observed position
        return {
            "peak_x": self.best_x,
            "peak_signal": self.best_signal,
        }

    def observe(self, obs) -> None:
        """Update belief with new observation.

        For sweep locator, we track the best signal value seen
        and update the peak position posterior.

        Parameters
        ----------
        obs : Observation
            New measurement
        """
        # Track best observation
        if obs.signal_value > self.best_signal:
            self.best_signal = obs.signal_value
            self.best_x = obs.x

        # Update belief about peak location
        # Use signal value as likelihood: higher signal = more likely peak location
        peak_param = self.belief.get_param("peak_x")

        # Compute likelihood based on distance from observation point
        # and signal strength
        likelihoods = np.exp(-0.5 * ((peak_param.grid - obs.x) / 0.1) ** 2) * (
            obs.signal_value + 1.0
        )  # Shift to ensure positive

        # Bayesian update
        unnormalized = peak_param.posterior * (likelihoods + 1e-10)
        total = unnormalized.sum()
        if total > 1e-10:
            peak_param.posterior = unnormalized / total
            peak_param.value = peak_param.mean()

        # Store observation
        self.belief.last_obs = obs
