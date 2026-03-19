"""Abstract belief distribution interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from nvision.models.observation import Observation
from nvision.signal.signal import Parameter, SignalModel


@dataclass
class AbstractBeliefDistribution(ABC):
    """Abstract base class for all belief distributions.

    Represents the locator's live belief about the signal parameters.
    Can be implemented via discrete grids, Monte Carlo particles, or
    analytical approximations.

    Attributes
    ----------
    model : SignalModel
        The stateless signal model defining the shape.
    last_obs : Observation | None
        Most recent observation for history tracking.
    """

    model: SignalModel
    last_obs: Observation | None = None

    @abstractmethod
    def update(self, obs: Observation) -> None:
        """Incremental Bayesian update from a new observation."""

    @abstractmethod
    def estimates(self) -> dict[str, float]:
        """Get current parameter estimates (e.g., posterior means)."""

    def uncertainty(self) -> dict[str, float]:
        """Get uncertainty (std dev) for each parameter.

        If the underlying SignalModel provides analytical gradients, this will
        attempt to use the Fisher Information Matrix (Cramer-Rao bound) to compute
        theoretical variance. Otherwise, it falls back to the empirical variance
        of the particles/grid.
        """
        # Try mathematical variance first if we have a recent observation
        if self.last_obs is not None:
            fim = self.fisher_information(self.last_obs.x)
            if fim is not None:
                try:
                    # Cramer-Rao bound: Covariance >= Inverse Fisher Information
                    # Add small ridge for numerical stability
                    cov = np.linalg.inv(fim + np.eye(len(fim)) * 1e-6)
                    param_names = self.model.parameter_names()
                    return {name: float(np.sqrt(max(0.0, cov[i, i]))) for i, name in enumerate(param_names)}
                except np.linalg.LinAlgError:
                    pass  # Fall back to empirical if singular

        return self._empirical_uncertainty()

    @abstractmethod
    def _empirical_uncertainty(self) -> dict[str, float]:
        """Compute empirical uncertainty from the underlying grid/particles."""

    @abstractmethod
    def entropy(self) -> float:
        """Compute total entropy across all parameters.

        Like uncertainty(), this could be overridden by future subclasses to
        compute analytical entropy (e.g., 0.5 * log(|2*pi*e*Sigma|) using the
        Fisher Information Matrix) instead of empirical entropy.
        """

    @abstractmethod
    def converged(self, threshold: float) -> bool:
        """Check if all parameters have converged below threshold."""

    @abstractmethod
    def copy(self) -> AbstractBeliefDistribution:
        """Create deep copy of this belief for snapshotting."""

    def expected_information_gain(self, x: float) -> float:
        """Compute expected information gain if we measure at position x.

        By default, this is not implemented. Future analytical models (like a
        LaplaceBeliefDistribution) can implement this mathematically using
        the SignalModel gradients, completely bypassing the need for SMC.
        """
        raise NotImplementedError("Analytical EIG not implemented for this belief type.")

    @abstractmethod
    def sample(self, n: int) -> dict[str, np.ndarray]:
        """Draw n joint samples from the posterior distribution.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping parameter names to arrays of length n.
        """

    @abstractmethod
    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        """Evaluate the marginal Probability Density Function.

        Parameters
        ----------
        param_name : str
            Name of the parameter.
        x : np.ndarray
            Points at which to evaluate the PDF.

        Returns
        -------
        np.ndarray
            PDF values corresponding to x.
        """

    def __call__(self, x: float) -> float:
        """Evaluate belief signal at position x using posterior means."""
        params = [self.get_param(p) for p in self.model.parameter_names()]
        return self.model.compute(x, params)

    def fisher_information(self, x: float) -> np.ndarray | None:
        """Compute the Fisher Information Matrix at position x.

        Returns None if the underlying SignalModel does not support analytical gradients.
        """
        params = [self.get_param(p) for p in self.model.parameter_names()]
        grads = self.model.gradient(x, params)
        if grads is None:
            return None

        d_dim = len(params)
        fim = np.zeros((d_dim, d_dim))

        # Assume Gaussian noise model for the likelihood
        # I(theta) = (1 / sigma^2) * (grad_f * grad_f^T)
        # We use a heuristic noise_std similar to the update step
        noise_std = 0.01  # baseline noise

        grad_vec = np.array([grads[p.name] for p in params])
        fim = np.outer(grad_vec, grad_vec) / (noise_std**2)

        return fim

    @abstractmethod
    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        """Evaluate the marginal Cumulative Density Function.

        Parameters
        ----------
        param_name : str
            Name of the parameter.
        x : np.ndarray
            Points at which to evaluate the CDF.

        Returns
        -------
        np.ndarray
            CDF values corresponding to x.
        """

    @abstractmethod
    def get_param(self, name: str) -> Parameter:
        """Get parameter bounds by name."""
        pass
