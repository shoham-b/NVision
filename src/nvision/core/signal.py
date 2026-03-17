"""Signal model and parameter abstractions for Bayesian localization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from nvision.core.observation import Observation


@dataclass
class Parameter:
    """A parameter with a known value (ground truth or point estimate).

    Base class for parameters. Represents a parameter with a single value
    and no uncertainty. Used in TrueSignal for ground truth parameters.

    Attributes
    ----------
    name : str
        Parameter name (e.g., 'frequency', 'linewidth')
    bounds : tuple[float, float]
        (min, max) bounds for this parameter
    value : float
        Parameter value
    """

    name: str
    bounds: tuple[float, float]
    value: float

    def __post_init__(self) -> None:
        """Validate parameter is within bounds."""
        if not (self.bounds[0] <= self.value <= self.bounds[1]):
            raise ValueError(f"Parameter {self.name} value {self.value} outside bounds {self.bounds}")

    def mean(self) -> float:
        """Get parameter value (for consistency with ParameterWithPosterior)."""
        return self.value

    def uncertainty(self) -> float:
        """Get uncertainty (zero for known parameters)."""
        return 0.0

    def entropy(self) -> float:
        """Get entropy (zero for known parameters)."""
        return 0.0

    def converged(self, threshold: float) -> bool:
        """Check convergence (always True for known parameters)."""
        return True


@dataclass
class ParameterWithPosterior(Parameter):
    """A parameter with uncertainty represented as a posterior distribution.

    Extends Parameter to add a probability distribution over possible values.
    Used in BeliefSignal for parameters being estimated.

    Attributes
    ----------
    name : str
        Parameter name (inherited)
    bounds : tuple[float, float]
        (min, max) bounds (inherited)
    value : float
        Current best estimate (computed from posterior mean)
    grid : np.ndarray
        Discretized range over bounds
    posterior : np.ndarray
        Probability distribution over grid, updated incrementally
    """

    grid: np.ndarray = field(repr=False)
    posterior: np.ndarray = field(repr=False)
    value: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate posterior and compute value from mean."""
        if len(self.grid) != len(self.posterior):
            raise ValueError(f"Grid and posterior must have same length: {len(self.grid)} != {len(self.posterior)}")
        if not np.isclose(self.posterior.sum(), 1.0):
            raise ValueError(f"Posterior must sum to 1.0, got {self.posterior.sum()}")
        # Compute value from posterior mean
        self.value = self._compute_mean()

    def _compute_mean(self) -> float:
        """Compute expected value from posterior."""
        return float(np.sum(self.grid * self.posterior))

    def mean(self) -> float:
        """Get expected value of parameter from posterior."""
        return self._compute_mean()

    def uncertainty(self) -> float:
        """Compute standard deviation of posterior."""
        mean_val = self.mean()
        variance = np.sum(self.posterior * (self.grid - mean_val) ** 2)
        return float(np.sqrt(variance))

    def entropy(self) -> float:
        """Compute Shannon entropy of posterior distribution."""
        p_nonzero = self.posterior[self.posterior > 0]
        return float(-np.sum(p_nonzero * np.log(p_nonzero)))

    def converged(self, threshold: float) -> bool:
        """Check if uncertainty is below threshold."""
        return self.uncertainty() < threshold


class SignalModel(ABC):
    """Abstract signal model defining the shape of the signal.

    Stateless. Knows how to compute the signal given any parameter values.
    Shared between TrueSignal and BeliefSignal.
    """

    @abstractmethod
    def compute(self, x: float, params: list[Parameter]) -> float:
        """Compute signal value at position x given parameters.

        Parameters
        ----------
        x : float
            Position to evaluate signal
        params : list[Parameter]
            List of parameters (can be Parameter or ParameterWithPosterior)

        Returns
        -------
        float
            Signal value at x
        """
        pass

    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Return ordered list of parameter names this model expects."""
        pass

    def _params_to_dict(self, params: list[Parameter]) -> dict[str, float]:
        """Convert parameter list to dict for easier access."""
        return {p.name: p.value for p in params}


@dataclass
class TrueSignal:
    """Ground truth signal with fixed parameters.

    Only exists in simulation — in real hardware this is unknown.
    Produces measurements via __call__.

    Attributes
    ----------
    model : SignalModel
        The signal model defining the shape
    parameters : list[Parameter]
        Exact parameter values, no uncertainty
    """

    model: SignalModel
    parameters: list[Parameter]

    def __post_init__(self) -> None:
        """Validate parameters match model."""
        expected = set(self.model.parameter_names())
        actual = {p.name for p in self.parameters}
        if expected != actual:
            raise ValueError(f"Parameters don't match model. Expected {expected}, got {actual}")

    def __call__(self, x: float) -> float:
        """Evaluate true signal at position x.

        Parameters
        ----------
        x : float
            Position to evaluate

        Returns
        -------
        float
            True signal value
        """
        return self.model.compute(x, self.parameters)

    def get_param(self, name: str) -> Parameter:
        """Get parameter by name."""
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"Parameter {name} not found")


@dataclass
class BeliefSignal:
    """Locator's live belief about the signal.

    Same shape as TrueSignal via shared SignalModel.
    Posterior over parameters updated incrementally as observations come in.
    Starts as prior, narrows toward TrueSignal over time.

    Attributes
    ----------
    model : SignalModel
        Same signal model shape as TrueSignal
    parameters : list[ParameterWithPosterior]
        Each has grid + posterior array
    last_obs : Observation | None
        Most recent observation for history tracking
    """

    model: SignalModel
    parameters: list[ParameterWithPosterior]
    last_obs: Observation | None = None

    def __post_init__(self) -> None:
        """Validate belief configuration."""
        expected = set(self.model.parameter_names())
        actual = {p.name for p in self.parameters}
        if expected != actual:
            raise ValueError(f"Parameters don't match model. Expected {expected}, got {actual}")
        # Ensure all parameters are ParameterWithPosterior
        for p in self.parameters:
            if not isinstance(p, ParameterWithPosterior):
                raise TypeError(f"BeliefSignal requires ParameterWithPosterior, got {type(p)}")

    def update(self, obs: Observation) -> None:
        """Incremental Bayesian update from new observation.

        Updates posterior distributions for all parameters based on
        likelihood of observed signal value given parameter values.
        No history replay — updates use only current posterior and new observation.

        Parameters
        ----------
        obs : Observation
            New observation to update belief with
        """
        self.last_obs = obs

        # Build multi-dimensional grid of parameter combinations
        # For computational efficiency, we'll use a factored approximation:
        # update each parameter's marginal posterior independently

        for param_idx, param in enumerate(self.parameters):
            # Compute likelihood for each value in this parameter's grid
            # holding other parameters at their posterior means
            other_params = [p if i != param_idx else None for i, p in enumerate(self.parameters)]

            likelihoods = np.zeros_like(param.grid)
            for i, val in enumerate(param.grid):
                # Create temporary parameter list with this value
                temp_params = [
                    p if p is not None else Parameter(name=param.name, bounds=param.bounds, value=val)
                    for p in other_params
                ]

                predicted = self.model.compute(obs.x, temp_params)

                # Gaussian likelihood with adaptive noise estimate
                # Use current uncertainty as noise proxy
                noise_std = max(0.01, param.uncertainty() * 0.1)
                likelihood = np.exp(-0.5 * ((obs.signal_value - predicted) / noise_std) ** 2)
                likelihoods[i] = likelihood

            # Bayesian update: posterior ∝ prior × likelihood
            unnormalized = param.posterior * likelihoods

            # Normalize
            total = unnormalized.sum()
            if total > 1e-10:
                param.posterior = unnormalized / total
                # Update value to new posterior mean
                param.value = param.mean()
            # else: keep prior if all likelihoods are near zero (shouldn't happen)

    def __call__(self, x: float) -> float:
        """Evaluate belief signal at position x using posterior means.

        Parameters
        ----------
        x : float
            Position to evaluate

        Returns
        -------
        float
            Expected signal value at x
        """
        # parameters already have .value set to posterior mean
        return self.model.compute(x, self.parameters)

    def uncertainty(self) -> dict[str, float]:
        """Get uncertainty (std) for each parameter."""
        return {p.name: p.uncertainty() for p in self.parameters}

    def entropy(self) -> float:
        """Compute total entropy across all parameters."""
        return sum(p.entropy() for p in self.parameters)

    def estimates(self) -> dict[str, float]:
        """Get current parameter estimates (posterior means)."""
        return {p.name: p.mean() for p in self.parameters}

    def converged(self, threshold: float) -> bool:
        """Check if all parameters have converged below threshold."""
        return all(p.converged(threshold) for p in self.parameters)

    def copy(self) -> BeliefSignal:
        """Create deep copy of this belief for snapshotting."""
        return BeliefSignal(
            model=self.model,  # model is stateless, can share
            parameters=[
                ParameterWithPosterior(
                    name=p.name,
                    bounds=p.bounds,
                    grid=p.grid.copy(),
                    posterior=p.posterior.copy(),
                )
                for p in self.parameters
            ],
            last_obs=self.last_obs,
        )

    def get_param(self, name: str) -> ParameterWithPosterior:
        """Get parameter by name."""
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"Parameter {name} not found")
