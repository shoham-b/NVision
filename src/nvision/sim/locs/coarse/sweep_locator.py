"""Simple sweep locator using core architecture.

This demonstrates how to implement a locator using the new core architecture
with BeliefSignal and incremental Bayesian updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nvision.models.locator import Locator
from nvision.signal.abstract_belief import AbstractBeliefDistribution
from nvision.signal.dtypes import FLOAT_DTYPE
from nvision.signal.grid_belief import GridBeliefDistribution, GridParameter
from nvision.signal.signal import ParamSpec, SignalModel
from nvision.sim.locs.coarse.numba_kernels import gaussian_peak_posterior_update


@dataclass(frozen=True)
class _BlackBoxParams:
    peak_x: float


@dataclass(frozen=True)
class _BlackBoxSampleParams:
    peak_x: np.ndarray


@dataclass(frozen=True)
class _BlackBoxUncertaintyParams:
    peak_x: float


class _BlackBoxSpec(ParamSpec[_BlackBoxParams, _BlackBoxSampleParams, _BlackBoxUncertaintyParams]):
    @property
    def names(self) -> tuple[str, ...]:
        return ("peak_x",)

    @property
    def dim(self) -> int:
        return 1

    def unpack_params(self, values) -> _BlackBoxParams:
        (v,) = values
        return _BlackBoxParams(float(v))

    def pack_params(self, params: _BlackBoxParams) -> tuple[float, ...]:
        return (float(params.peak_x),)

    def unpack_uncertainty(self, values) -> _BlackBoxUncertaintyParams:
        (v,) = values
        return _BlackBoxUncertaintyParams(float(v))

    def pack_uncertainty(self, u: _BlackBoxUncertaintyParams) -> tuple[float, ...]:
        return (float(u.peak_x),)

    def unpack_samples(self, arrays_in_order) -> _BlackBoxSampleParams:
        (px,) = arrays_in_order
        return _BlackBoxSampleParams(peak_x=np.asarray(px, dtype=FLOAT_DTYPE))

    def pack_samples(self, samples: _BlackBoxSampleParams) -> tuple[np.ndarray, ...]:
        return (np.asarray(samples.peak_x, dtype=FLOAT_DTYPE),)


class BlackBoxSignalModel(SignalModel[_BlackBoxParams, _BlackBoxSampleParams, _BlackBoxUncertaintyParams]):
    """Signal model that treats parameters as a single peak position.

    Used for simple sweep locators that only estimate peak location.
    """

    _SPEC = _BlackBoxSpec()

    @property
    def spec(self) -> _BlackBoxSpec:
        return self._SPEC

    def compute(self, x: float, params: _BlackBoxParams) -> float:
        return 0.0

    def compute_vectorized_samples(self, x: float, samples: _BlackBoxSampleParams) -> np.ndarray:
        # Not used by observe() path; exists for compatibility.
        return np.zeros_like(samples.peak_x, dtype=FLOAT_DTYPE)


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

        updated_posterior, total = gaussian_peak_posterior_update(
            peak_param.grid,
            peak_param.posterior,
            float(obs.x),
            float(obs.signal_value),
        )
        if total > 1e-10:
            peak_param.posterior = updated_posterior
            peak_param.value = peak_param.mean()

        # Store observation
        self.belief.last_obs = obs
