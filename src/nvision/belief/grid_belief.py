"""Grid-based belief distribution implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import float64
from numba.experimental.jitclass import jitclass

from nvision.belief.abstract_belief import AbstractBeliefDistribution, ParameterValues
from nvision.models.observation import Observation
from nvision.parameter import Parameter
from nvision.signal.likelihood import likelihood_from_observation_model

# --- Closed numeric core: 1D discrete PMF on a fixed grid (Numba jitclass) -----

_marginal_spec = [
    ("grid", float64[:]),
    ("posterior", float64[:]),
]


@jitclass(_marginal_spec)
class _MarginalGrid1D:
    """JIT-compiled 1D marginal: support grid and PMF vector (same length).

    Pure array math; no Python objects inside methods.  Used by
    :class:`GridParameter` for mean / entropy / Bayesian multiply-normalize.
    """

    def __init__(self, grid: np.ndarray, posterior: np.ndarray) -> None:
        self.grid = grid
        self.posterior = posterior

    def mean(self) -> float:
        m = 0.0
        for i in range(self.grid.shape[0]):
            m += self.posterior[i] * self.grid[i]
        return m

    def mean_variance(self) -> tuple[float, float]:
        m = self.mean()
        v = 0.0
        for i in range(self.grid.shape[0]):
            d = self.grid[i] - m
            v += self.posterior[i] * d * d
        return m, v

    def entropy(self) -> float:
        s = 0.0
        for i in range(self.posterior.shape[0]):
            p = self.posterior[i]
            if p > 0.0:
                s -= p * np.log(p)
        return s

    def multiply_by_likelihood_normalize(self, likelihoods: np.ndarray) -> bool:
        """``posterior *= likelihood``, renormalize; return False if mass vanishes."""
        n = self.posterior.shape[0]
        total = 0.0
        for i in range(n):
            total += self.posterior[i] * likelihoods[i]
        if total <= 1e-10:
            return False
        inv = 1.0 / total
        for i in range(n):
            self.posterior[i] = self.posterior[i] * likelihoods[i] * inv
        return True

    def copy(self):
        return _MarginalGrid1D(self.grid.copy(), self.posterior.copy())


class GridParameter(Parameter):
    """A parameter with uncertainty represented as a discrete 1D grid."""

    __slots__ = ("_marginal",)

    def __init__(self, name: str, bounds: tuple[float, float], grid: np.ndarray, posterior: np.ndarray) -> None:
        g = np.ascontiguousarray(np.asarray(grid, dtype=np.float64))
        p = np.ascontiguousarray(np.asarray(posterior, dtype=np.float64))
        if len(g) != len(p):
            raise ValueError(f"Grid and posterior must have same length: {len(g)} != {len(p)}")
        if not np.isclose(float(np.sum(p)), 1.0):
            raise ValueError(f"Posterior must sum to 1.0, got {float(np.sum(p))}")
        self._marginal = _MarginalGrid1D(g, p)
        super().__init__(name=name, bounds=bounds, value=float(self._marginal.mean()))

    @property
    def grid(self) -> np.ndarray:
        return self._marginal.grid

    @property
    def posterior(self) -> np.ndarray:
        return self._marginal.posterior

    @posterior.setter
    def posterior(self, v: np.ndarray) -> None:
        """Replace PMF values in-place (same length as grid); updates ``value``."""
        arr = np.ascontiguousarray(np.asarray(v, dtype=np.float64))
        n = self._marginal.posterior.shape[0]
        if arr.shape[0] != n:
            raise ValueError("posterior length must match grid length")
        self._marginal.posterior[:] = arr
        object.__setattr__(self, "value", float(self._marginal.mean()))

    def mean(self) -> float:
        return float(self._marginal.mean())

    def uncertainty(self) -> float:
        _, var = self._marginal.mean_variance()
        return float(np.sqrt(max(0.0, var)))

    def entropy(self) -> float:
        return float(self._marginal.entropy())

    def converged(self, threshold: float) -> bool:
        return self.uncertainty() < threshold

    def apply_likelihood(self, likelihoods: np.ndarray) -> bool:
        """Bayesian update: multiply PMF by ``likelihoods`` and renormalize. Updates ``value`` if successful."""
        lik = np.ascontiguousarray(np.asarray(likelihoods, dtype=np.float64))
        ok = self._marginal.multiply_by_likelihood_normalize(lik)
        if ok:
            object.__setattr__(self, "value", float(self._marginal.mean()))
        return ok


@dataclass
class GridBeliefDistribution(AbstractBeliefDistribution):
    """Belief distribution using independent 1D discrete grids."""

    parameters: list[GridParameter] = field(default_factory=list)

    def __post_init__(self) -> None:
        expected = set(self.model.parameter_names())
        actual = {p.name for p in self.parameters}
        if expected != actual:
            raise ValueError(f"Parameters don't match model. Expected {expected}, got {actual}")

    def update(self, obs: Observation) -> None:
        self.last_obs = obs

        param_names = self.model.parameter_names()
        param_by_name = {p.name: p for p in self.parameters}

        for _param_idx, param in enumerate(self.parameters):
            grid = np.asarray(param.grid, dtype=np.float64)
            arrays_in_order: list[np.ndarray] = []
            for name in param_names:
                if name == param.name:
                    arrays_in_order.append(grid)
                else:
                    other = param_by_name[name]
                    arrays_in_order.append(np.full(grid.shape, float(other.value), dtype=np.float64))

            predicted = self.model.compute_vectorized(obs.x, *arrays_in_order)
            noise_std = obs.noise_std
            likelihoods = likelihood_from_observation_model(
                obs_y=obs.signal_value,
                predicted=predicted,
                noise_std=noise_std,
                frequency_noise_model=obs.frequency_noise_model,
            )

            param.apply_likelihood(likelihoods)

    def estimates(self) -> dict[str, float]:
        return {p.name: p.mean() for p in self.parameters}

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        param_names = self.model.parameter_names()
        data = {p.name: p.uncertainty() for p in self.parameters}
        return ParameterValues.from_mapping(param_names, data)

    def entropy(self) -> float:
        return sum(p.entropy() for p in self.parameters)

    def converged(self, threshold: float) -> bool:
        return all(p.converged(threshold) for p in self.parameters)

    def copy(self) -> GridBeliefDistribution:
        return GridBeliefDistribution(
            model=self.model,
            parameters=[
                GridParameter(
                    name=p.name,
                    bounds=p.bounds,
                    grid=p.grid.copy(),
                    posterior=p.posterior.copy(),
                )
                for p in self.parameters
            ],
            last_obs=self.last_obs,
        )

    def get_param(self, name: str) -> GridParameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"Parameter {name} not found")

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        """Sample independently from each marginal grid."""
        param_names = self.model.parameter_names()
        data = {p.name: p.grid[np.random.choice(len(p.grid), size=n, p=p.posterior)] for p in self.parameters}
        return ParameterValues.from_mapping(param_names, data)

    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        """Interpolate the discrete posterior grid."""
        p = self.get_param(param_name)
        # Normalize by grid spacing to make it a true density
        spacing = p.grid[1] - p.grid[0] if len(p.grid) > 1 else 1.0
        density = p.posterior / spacing
        return np.interp(x, p.grid, density, left=0.0, right=0.0)

    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        """Interpolate the discrete CDF grid."""
        p = self.get_param(param_name)
        cdf = np.cumsum(p.posterior)
        return np.interp(x, p.grid, cdf, left=0.0, right=1.0)
