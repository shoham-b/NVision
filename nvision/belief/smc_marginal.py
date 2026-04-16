"""Sequential Monte Carlo (Particle Filter) belief distribution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from nvision.belief.abstract_marginal import AbstractMarginalDistribution, ParameterValues
from nvision.models.observation import Observation
from nvision.spectra.likelihood import likelihood_from_observation_model

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
        if j >= m - 1:
            indices[i] = m - 1
            i += 1
        elif positions[i] < cumulative_sum[j]:
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
class SMCMarginalDistribution(AbstractMarginalDistribution):
    """Belief distribution using Sequential Monte Carlo (Particle Filter).

    Maintains a joint posterior over parameters using a set of weighted particles.

    Parameters
    ----------
    use_full_covariance : bool
        If True, use NIST optbayesexpt-style resampling with full covariance
        multivariate Gaussian nudging and contraction toward mean (a_param).
        If False (default), use independent per-dimension jitter.
    a_param : float
        Contraction parameter for full-covariance resampling. Particles are
        contracted (1 - a_param) of the distance toward the mean after nudging.
        Default 0.98 (as in NIST optbayesexpt).
    scale : bool
        If True and use_full_covariance=True, apply contraction toward mean.
        Default True.
    """

    parameter_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    num_particles: int = 1000
    jitter_scale: float = 0.05
    ess_threshold: float = 0.5
    use_full_covariance: bool = False
    a_param: float = 0.98
    scale: bool = True

    _particles: np.ndarray = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)
    _param_names: list[str] = field(init=False, repr=False)

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

        if self.use_full_covariance:
            self._resample_nist_style()
        else:
            # Add jitter (kernel density perturbation) - independent per-dimension
            for j, name in enumerate(self._param_names):
                lo, hi = self.parameter_bounds[name]
                std = np.std(self._particles[:, j])
                jitter = np.random.normal(0, std * self.jitter_scale, self.num_particles)
                self._particles[:, j] = np.clip(self._particles[:, j] + jitter, lo, hi)

    def _resample_nist_style(self) -> None:
        """NIST optbayesexpt-style resampling with full covariance nudging and contraction.

        This implements the algorithm from Dushenko et al. (2020) as described in
        the supplemental material Section S.3:
        1. Particles are resampled with replacement (already done in _resample)
        2. Each particle gets a random displacement (nudge) from a multivariate
           Gaussian with covariance (1 - a_param^2) * particle_covariance
        3. All particles are contracted toward the mean by factor a_param to
           compensate for the diffusion from nudging
        4. Weights are reset to uniform
        """
        d_dim = len(self._param_names)

        # Compute mean and full covariance of current particles
        mean = _weighted_mean_axis0(self._particles, self._weights)  # shape (d,)
        cov = np.cov(self._particles, rowvar=False, aweights=self._weights)  # shape (d, d)

        if cov.ndim == 0:
            cov = np.array([[cov]])

        # Nudge covariance: (1 - a_param^2) * cov
        # This ensures the nudge scale is small compared to distribution spread
        nudge_cov = (1 - self.a_param ** 2) * cov

        # Ensure nudge_cov is positive semi-definite for numerical stability
        try:
            nudges = np.random.multivariate_normal(
                np.zeros(d_dim),
                nudge_cov,
                self.num_particles
            )  # shape (n_particles, d)
        except np.linalg.LinAlgError:
            # Fall back to diagonal covariance if full covariance is singular
            diag_cov = np.diag(np.diag(nudge_cov))
            nudges = np.random.multivariate_normal(
                np.zeros(d_dim),
                diag_cov,
                self.num_particles
            )

        # Apply nudges
        self._particles = self._particles + nudges

        # Shrinkage/contraction toward mean if enabled
        if self.scale:
            old_center = mean.reshape(1, -1)  # shape (1, d)
            self._particles = (
                self._particles * self.a_param +
                old_center * (1 - self.a_param)
            )

        # Clip to bounds
        for j, name in enumerate(self._param_names):
            lo, hi = self.parameter_bounds[name]
            self._particles[:, j] = np.clip(self._particles[:, j], lo, hi)

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

    def covariance_matrix(self) -> np.ndarray:
        """Return full covariance matrix of particle distribution.

        Returns a (d, d) array where d is the number of parameters.
        """
        cov = np.cov(self._particles, rowvar=False, aweights=self._weights)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        return cov

    def correlation_matrix(self) -> np.ndarray:
        """Return correlation matrix (normalized covariance).

        Returns a (d, d) array with values in [-1, 1].
        Diagonal entries are always 1.0.
        """
        cov = self.covariance_matrix()
        stds = np.sqrt(np.diag(cov))
        # Avoid division by zero
        if np.any(stds < 1e-15):
            stds = np.maximum(stds, 1e-15)
        corr = cov / np.outer(stds, stds)
        # Clip to handle numerical errors
        return np.clip(corr, -1.0, 1.0)

    def generalized_variance(self) -> float:
        """Return determinant of covariance matrix (generalized variance).

        This is a scalar measure of total uncertainty volume.
        Smaller values indicate tighter posterior concentration.
        """
        cov = self.covariance_matrix()
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return 0.0
        return float(np.exp(logdet))

    def converged(self, threshold: float) -> bool:
        return all(u < threshold for u in self.uncertainty().values())

    def copy(self) -> SMCMarginalDistribution:
        dist = SMCMarginalDistribution(
            model=self.model,
            parameter_bounds=self.parameter_bounds.copy(),
            num_particles=self.num_particles,
            jitter_scale=self.jitter_scale,
            ess_threshold=self.ess_threshold,
            use_full_covariance=self.use_full_covariance,
            a_param=self.a_param,
            scale=self.scale,
            last_obs=self.last_obs,
        )
        dist._param_names = self._param_names.copy()
        dist._particles = self._particles.copy()
        dist._weights = self._weights.copy()
        return dist

    def _weighted_mean(self, name: str) -> float:
        if name not in self.parameter_bounds:
            raise KeyError(f"Parameter {name} not found")
        idx = self._param_names.index(name)
        mean_val, _ = _weighted_mean_variance_1d(self._particles[:, idx], self._weights)
        return float(mean_val)

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        indices = np.random.choice(self.num_particles, size=n, p=self._weights)
        samples = self._particles[indices]
        data = {name: samples[:, i] for i, name in enumerate(self._param_names)}
        return ParameterValues.from_mapping(self._param_names, data)

    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        from scipy.stats import gaussian_kde, norm

        idx = self._param_names.index(param_name)
        samples = self._particles[:, idx]

        # Fall back to a narrow Gaussian when particles have collapsed (zero variance)
        if np.std(samples) < 1e-10:
            mean_val, _ = _weighted_mean_variance_1d(samples, self._weights)
            lo, hi = self.parameter_bounds[param_name]
            bw = (hi - lo) * 1e-3
            return norm.pdf(x, loc=mean_val, scale=max(bw, 1e-10))

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
