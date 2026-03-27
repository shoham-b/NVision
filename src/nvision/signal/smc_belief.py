"""Sequential Monte Carlo (Particle Filter) belief distribution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from nvision.models.observation import Observation
from nvision.signal.abstract_belief import AbstractBeliefDistribution, ParameterValues
from nvision.signal.likelihood import likelihood_from_observation_model
from nvision.signal.signal import Parameter

# --- Numba helpers (particle weights / resampling) ----------------------------


@njit(cache=True)
def _weighted_mean_variance_1d(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted mean and variance of ``x`` with weights ``w`` (not assumed normalized)."""
    n = x.shape[0]
    s = 0.0
    for i in range(n):
        s += w[i]
    if s <= 0.0:
        return 0.0, 0.0
    mean = 0.0
    for i in range(n):
        mean += (w[i] / s) * x[i]
    var = 0.0
    for i in range(n):
        d = x[i] - mean
        var += (w[i] / s) * d * d
    return mean, var


@njit(cache=True)
def _weighted_mean_axis0(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Column-wise weighted means for ``particles`` shaped ``(n, d)``."""
    n, d = particles.shape
    means = np.empty(d, dtype=np.float64)
    sw = 0.0
    for i in range(n):
        sw += weights[i]
    if sw <= 0.0:
        for j in range(d):
            means[j] = 0.0
        return means
    for j in range(d):
        m = 0.0
        for i in range(n):
            m += weights[i] * particles[i, j]
        means[j] = m / sw
    return means


@njit(cache=True)
def _systematic_resample_indices(cumulative_sum: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Map systematic ``positions`` to indices along non-decreasing ``cumulative_sum``."""
    n = positions.shape[0]
    indices = np.empty(n, dtype=np.int64)
    i = 0
    j = 0
    m = cumulative_sum.shape[0]
    while i < n:
        if j >= m:
            j = m - 1
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


@njit(cache=True)
def _inverse_sum_squares(weights: np.ndarray) -> float:
    """Return ``1 / sum(w**2)`` (ESS denominator for normalized weights)."""
    s = 0.0
    for i in range(weights.shape[0]):
        w = weights[i]
        s += w * w
    if s <= 0.0:
        return 0.0
    return 1.0 / s


@dataclass
class SMCBeliefDistribution(AbstractBeliefDistribution):
    """Belief distribution using Sequential Monte Carlo (Particle Filter).

    Maintains a joint posterior over parameters using a set of weighted particles.
    """

    parameter_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    num_particles: int = 1000
    jitter_scale: float = 0.05
    ess_threshold: float = 0.5

    _particles: np.ndarray = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)
    _param_names: list[str] = field(init=False, repr=False)
    _param_scratch: list[Parameter] = field(init=False, repr=False)

    def _init_param_scratch(self) -> None:
        """One Parameter per dimension, reused across particles (same model, update .value only)."""
        self._param_scratch = [
            Parameter(
                name=name,
                bounds=self.parameter_bounds[name],
                value=0.5 * (self.parameter_bounds[name][0] + self.parameter_bounds[name][1]),
            )
            for name in self._param_names
        ]

    def __post_init__(self) -> None:
        self._param_names = self.model.parameter_names()

        # Initialize particles uniformly within bounds
        d_dim = len(self._param_names)
        self._particles = np.zeros((self.num_particles, d_dim))

        for i, name in enumerate(self._param_names):
            if name not in self.parameter_bounds:
                raise ValueError(f"Missing bounds for parameter: {name}")
            lo, hi = self.parameter_bounds[name]
            self._particles[:, i] = np.random.uniform(lo, hi, self.num_particles)

        self._weights = np.ones(self.num_particles) / self.num_particles
        self._init_param_scratch()

    def update(self, obs: Observation) -> None:
        self.last_obs = obs

        # 1. Compute likelihood for all particles (vectorized model evaluation)
        arrays_in_order = [self._particles[:, j] for j in range(len(self._param_names))]
        predicted = self.model.compute_vectorized(obs.x, *arrays_in_order)

        # Adaptive noise based on current uncertainty (similar to grid)
        # In a rigorous SMC, this should ideally be a fixed measurement noise
        noise_std = obs.noise_std
        likelihoods = likelihood_from_observation_model(
            obs_y=obs.signal_value,
            predicted=predicted,
            noise_std=noise_std,
            frequency_noise_model=obs.frequency_noise_model,
        )

        # 2. Update weights
        self._weights *= likelihoods

        # 3. Normalize weights
        weight_sum = np.sum(self._weights)
        if weight_sum > 1e-10:
            self._weights /= weight_sum
        else:
            # If all particles have 0 likelihood, reset to uniform (should rarely happen)
            self._weights = np.ones(self.num_particles) / self.num_particles

        # 4. Resample if Effective Sample Size (ESS) is too low
        ess = _inverse_sum_squares(self._weights)
        if ess < self.ess_threshold * self.num_particles:
            self._resample()

    def _resample(self) -> None:
        """Systematic resampling with jitter to prevent particle collapse."""
        # Systematic resampling
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        cumulative_sum = np.cumsum(self._weights)
        indices = _systematic_resample_indices(cumulative_sum, positions)

        self._particles = self._particles[indices]
        self._weights = np.ones(self.num_particles) / self.num_particles

        # Add jitter (kernel density perturbation)
        for j, name in enumerate(self._param_names):
            lo, hi = self.parameter_bounds[name]
            std = np.std(self._particles[:, j])
            jitter = np.random.normal(0, std * self.jitter_scale, self.num_particles)
            self._particles[:, j] = np.clip(self._particles[:, j] + jitter, lo, hi)

    def _marginal_std(self, dim_idx: int) -> float:
        _, var = _weighted_mean_variance_1d(self._particles[:, dim_idx], self._weights)
        return float(np.sqrt(max(0.0, var)))

    def estimates(self) -> dict[str, float]:
        means = _weighted_mean_axis0(self._particles, self._weights)
        return {name: float(means[i]) for i, name in enumerate(self._param_names)}

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        uncertainties: dict[str, float] = {}
        for i, name in enumerate(self._param_names):
            uncertainties[name] = self._marginal_std(i)
        return ParameterValues.from_mapping(self._param_names, uncertainties)

    def entropy(self) -> float:
        # Simple Kozachenko-Leonenko nearest-neighbor entropy estimator could go here.
        # For now, approximate via a Gaussian assumption on the particles.
        cov = np.cov(self._particles, rowvar=False, aweights=self._weights)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return 0.0
        d_dim = len(self._param_names)
        return float(0.5 * logdet + 0.5 * d_dim * (1 + np.log(2 * np.pi)))

    def converged(self, threshold: float) -> bool:
        return all(u < threshold for u in self.uncertainty().values())

    def copy(self) -> SMCBeliefDistribution:
        dist = SMCBeliefDistribution(
            model=self.model,
            parameter_bounds=self.parameter_bounds.copy(),
            num_particles=self.num_particles,
            jitter_scale=self.jitter_scale,
            ess_threshold=self.ess_threshold,
            last_obs=self.last_obs,
        )
        dist._param_names = self._param_names.copy()
        dist._particles = self._particles.copy()
        dist._weights = self._weights.copy()
        dist._init_param_scratch()
        return dist

    def get_param(self, name: str) -> Parameter:
        if name not in self.parameter_bounds:
            raise KeyError(f"Parameter {name} not found")

        idx = self._param_names.index(name)
        mean_val, _ = _weighted_mean_variance_1d(self._particles[:, idx], self._weights)
        return Parameter(name=name, bounds=self.parameter_bounds[name], value=float(mean_val))

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        indices = np.random.choice(self.num_particles, size=n, p=self._weights)
        samples = self._particles[indices]
        data = {name: samples[:, i] for i, name in enumerate(self._param_names)}
        return ParameterValues.from_mapping(self._param_names, data)

    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        from scipy.stats import gaussian_kde

        idx = self._param_names.index(param_name)
        samples = self._particles[:, idx]

        # Fit KDE using weights
        kde = gaussian_kde(samples, weights=self._weights)
        return kde.evaluate(x)

    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        # For 1D CDF, we can just sort the particles and compute the empirical CDF
        idx = self._param_names.index(param_name)
        samples = self._particles[:, idx]

        sort_idx = np.argsort(samples)
        sorted_samples = samples[sort_idx]
        sorted_weights = self._weights[sort_idx]

        cdf = np.cumsum(sorted_weights)
        return np.interp(x, sorted_samples, cdf, left=0.0, right=1.0)
