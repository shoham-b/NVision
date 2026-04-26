"""Sequential Monte Carlo (Particle Filter) belief distribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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

    # Annealed jitter: continuous particle movement every update
    annealed_jitter: bool = False
    annealed_jitter_initial: float = 0.02
    annealed_jitter_min: float = 0.001
    annealed_jitter_decay: float = 0.995

    # Elitist resampling: survival of the fittest
    elitism_ratio: float = 0.2  # Top 20% particles survive intact

    _particles: np.ndarray = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)
    _param_names: list[str] = field(init=False, repr=False)
    _current_annealed_jitter_scale: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        self._param_names = self.model.parameter_names()
        self._current_annealed_jitter_scale = self.annealed_jitter_initial

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

        # 2. Compute information-based weights from Fisher Information
        # Particles suggesting measurements at informative regions (high gradient,
        # strong cross-parameter sensitivity) get higher weight
        info_weights = self._compute_information_weights(obs.x, predicted, noise_std, obs.frequency_noise_model)

        # 3. Update weights combining likelihood AND information content
        # This ensures particles that both fit the data AND carry parameter
        # information are preserved, accounting for cross-parameter correlations
        self._weights *= likelihoods * info_weights

        # 4. Normalize weights
        weight_sum = np.sum(self._weights)
        if weight_sum > 1e-10:
            self._weights /= weight_sum
        else:
            # If all particles have 0 likelihood, reset to uniform (should rarely happen)
            self._weights = np.ones(self.num_particles) / self.num_particles

        # 5. Apply annealed jitter (continuous particle movement every update)
        if self.annealed_jitter:
            self._apply_annealed_jitter()

        # 6. Resample if Effective Sample Size (ESS) is too low
        ess = _inverse_sum_squares(self._weights)
        if ess < self.ess_threshold * self.num_particles:
            self._resample()

    def batch_update(self, observations: list[Observation]) -> None:
        """Update belief from multiple observations in a single batch.

        This is more efficient than calling update() repeatedly for each observation,
        especially during the initial sweep phase. All observations are processed
        together with a single resampling step at the end.

        Parameters
        ----------
        observations : list[Observation]
            List of observations to incorporate into the belief.
        """
        if not observations:
            return

        self.last_obs = observations[-1]

        # Pre-extract particles array for efficiency
        n_particles = self._particles.shape[0]
        n_params = len(self._param_names)
        arrays_in_order = [self._particles[:, j] for j in range(n_params)]

        # Accumulate weight updates across all observations
        for obs in observations:
            # 1. Compute likelihood for all particles (already vectorized)
            predicted = self.model.compute_vectorized(obs.x, *arrays_in_order)

            noise_std = obs.noise_std
            likelihoods = likelihood_from_observation_model(
                obs_y=obs.signal_value,
                predicted=predicted,
                noise_std=noise_std,
                frequency_noise_model=obs.frequency_noise_model,
            )

            # 2. Compute information-based weights (batched across particles)
            info_weights = self._compute_information_weights_batch(obs.x, n_particles, noise_std)

            # 3. Update weights (no normalization between observations)
            self._weights *= likelihoods * info_weights

        # 4. Final normalization after all observations
        weight_sum = np.sum(self._weights)
        if weight_sum > 1e-10:
            self._weights /= weight_sum
        else:
            self._weights = np.ones(self.num_particles) / self.num_particles

        # 5. Single resampling step after all updates
        ess = _inverse_sum_squares(self._weights)
        if ess < self.ess_threshold * self.num_particles:
            self._resample()

    def _compute_information_weights(
        self,
        x: float,
        predicted: np.ndarray,
        noise_std: float,
        frequency_noise_model: tuple[Any, ...] | None,
    ) -> np.ndarray:
        """Compute per-particle weights based on Fisher information content.

        Returns weights proportional to the generalized variance (determinant of FIM)
        which captures both individual parameter sensitivity AND cross-parameter
        correlations. Higher = more informative about the joint parameter distribution.
        """
        n_particles = self._particles.shape[0]
        info_scores = np.zeros(n_particles)

        for i in range(n_particles):
            # Extract particle parameters
            particle_params = {name: float(self._particles[i, j]) for j, name in enumerate(self._param_names)}
            typed_params = self.model.spec.unpack_params([particle_params[n] for n in self._param_names])

            # Compute gradients at this particle's parameter set
            grads = self.model.gradient(x, typed_params)
            if grads is None:
                info_scores[i] = 1.0  # Neutral if no gradients available
                continue

            grad_vec = np.array([grads[name] for name in self._param_names], dtype=np.float64)

            # Build Fisher Information Matrix: FIM = (1/sigma^2) * g * g^T for Gaussian
            # This captures cross-parameter correlations via outer product structure
            sigma = max(float(noise_std), 1e-9)
            fim = np.outer(grad_vec, grad_vec) / (sigma * sigma)

            # Use log-determinant as information score (generalized variance)
            # Higher determinant = more total information about parameters jointly
            sign, logdet = np.linalg.slogdet(fim + np.eye(len(self._param_names)) * 1e-6)
            if sign > 0 and np.isfinite(logdet):
                # Use softplus-like transformation: log(1 + exp(logdet/2)) to avoid overflow
                # This maps [-inf, +inf] -> [0, +inf] smoothly with saturation
                half_logdet = logdet * 0.5
                if half_logdet > 10:  # exp(10) ≈ 22026, start softening here
                    info_scores[i] = half_logdet  # Linear growth for large values
                else:
                    info_scores[i] = np.log1p(np.exp(half_logdet))
            else:
                # Fall back to trace (sum of eigenvalues) if determinant numerically unstable
                info_scores[i] = max(1e-6, np.trace(fim))

        # Clip to prevent overwhelming the likelihoods (0.1 = 10x less weight, 10 = 10x more weight)
        # This keeps information weights as a gentle nudge rather than a hard filter
        clipped_scores = np.clip(info_scores, 0.1, 10.0)

        # Normalize to mean 1.0 so likelihood remains primary driver
        mean_score = np.mean(clipped_scores)
        if mean_score > 0:
            return clipped_scores / mean_score
        return np.ones(n_particles)

    def _compute_information_weights_batch(
        self,
        x: float,
        n_particles: int,
        noise_std: float,
    ) -> np.ndarray:
        """Compute per-particle information weights for batch updates (optimized version).

        Uses model.gradient_vectorized() for fully vectorized gradient computation
        across all particles, eliminating the Python loop entirely for models that
        implement vectorized gradients.
        """
        # Try vectorized gradient computation first
        arrays_in_order = [self._particles[:, j] for j in range(len(self._param_names))]
        grad_dict = self.model.gradient_vectorized(x, *arrays_in_order)

        if grad_dict is None:
            # Model doesn't support gradients - return neutral weights
            return np.ones(n_particles)

        n_params = len(self._param_names)
        sigma = max(float(noise_std), 1e-9)
        sigma_sq = sigma * sigma

        # Extract gradient arrays into matrix (n_particles, n_params)
        grad_matrix = np.empty((n_particles, n_params), dtype=np.float64)
        for j, name in enumerate(self._param_names):
            grad_matrix[:, j] = grad_dict[name]

        # Compute information scores for all particles
        info_scores = np.zeros(n_particles)
        for i in range(n_particles):
            grad_vec = grad_matrix[i, :]
            # FIM = (1/sigma^2) * g * g^T
            fim = np.outer(grad_vec, grad_vec) / sigma_sq
            fim_reg = fim + np.eye(n_params) * 1e-6

            try:
                sign, logdet = np.linalg.slogdet(fim_reg)
                if sign > 0 and np.isfinite(logdet):
                    half_logdet = logdet * 0.5
                    if half_logdet > 10:
                        info_scores[i] = half_logdet
                    else:
                        info_scores[i] = np.log1p(np.exp(half_logdet))
                else:
                    info_scores[i] = max(1e-6, np.trace(fim))
            except Exception:
                info_scores[i] = max(1e-6, np.trace(fim))

        # Clip and normalize
        clipped_scores = np.clip(info_scores, 0.1, 10.0)
        mean_score = np.mean(clipped_scores)
        if mean_score > 0:
            return clipped_scores / mean_score
        return np.ones(n_particles)

    def _resample(self) -> None:
        """Elitist systematic resampling - survival of the fittest.

        Top elitism_ratio particles survive intact. Only eliminated particles
        are redistributed through resampling from the surviving elite.
        """
        n_elite = int(self.num_particles * self.elitism_ratio)
        n_elite = max(1, min(n_elite, self.num_particles - 1))

        # Find elite particles (highest weights) - these survive intact
        elite_indices = np.argsort(self._weights)[-n_elite:]

        # Remaining slots to fill via resampling
        n_resample = self.num_particles - n_elite

        # Systematic resampling from elite particles only
        elite_weights = self._weights[elite_indices]
        elite_weights_sum = np.sum(elite_weights)
        elite_weights = elite_weights / elite_weights_sum if elite_weights_sum > 1e-12 else np.ones(n_elite) / n_elite

        cumulative_sum = np.cumsum(elite_weights)
        positions = (np.arange(n_resample) + np.random.random()) / n_resample
        resample_indices = _systematic_resample_indices(cumulative_sum, positions)
        new_indices = elite_indices[resample_indices]

        # Construct new particle set: elite intact + resampled from elite
        all_indices = np.concatenate([elite_indices, new_indices])
        self._particles = self._particles[all_indices]
        self._weights = np.ones(self.num_particles) / self.num_particles

        # Apply jitter only to resampled particles (not elite)
        if self.use_full_covariance:
            self._resample_nist_style_elitist(n_elite)
        else:
            # Add jitter only to non-elite (resampled) particles
            for j, name in enumerate(self._param_names):
                lo, hi = self.parameter_bounds[name]
                std = np.std(self._particles[:, j])
                # Jitter for resampled particles only
                jitter = np.random.normal(0, std * self.jitter_scale, n_resample)
                self._particles[n_elite:, j] = np.clip(self._particles[n_elite:, j] + jitter, lo, hi)

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
        nudge_cov = (1 - self.a_param**2) * cov

        # Ensure nudge_cov is positive semi-definite for numerical stability
        try:
            nudges = np.random.multivariate_normal(
                np.zeros(d_dim), nudge_cov, self.num_particles
            )  # shape (n_particles, d)
        except np.linalg.LinAlgError:
            # Fall back to diagonal covariance if full covariance is singular
            diag_cov = np.diag(np.diag(nudge_cov))
            nudges = np.random.multivariate_normal(np.zeros(d_dim), diag_cov, self.num_particles)

        # Apply nudges
        self._particles = self._particles + nudges

        # Shrinkage/contraction toward mean if enabled
        if self.scale:
            old_center = mean.reshape(1, -1)  # shape (1, d)
            self._particles = self._particles * self.a_param + old_center * (1 - self.a_param)

        # Clip to bounds
        for j, name in enumerate(self._param_names):
            lo, hi = self.parameter_bounds[name]
            self._particles[:, j] = np.clip(self._particles[:, j], lo, hi)

    def _resample_nist_style_elitist(self, n_elite: int) -> None:
        """NIST-style resampling that only nudges non-elite particles.

        Elite particles (first n_elite indices) remain intact.
        Only resampled particles get the full covariance nudge + contraction.
        """
        d_dim = len(self._param_names)
        n_resample = self.num_particles - n_elite

        # Compute mean and covariance from entire particle set (for proper nudging)
        mean = _weighted_mean_axis0(self._particles, self._weights)
        cov = np.cov(self._particles, rowvar=False, aweights=self._weights)
        if cov.ndim == 0:
            cov = np.array([[cov]])

        # Nudge covariance
        nudge_cov = (1 - self.a_param**2) * cov

        # Generate nudges only for resampled particles
        try:
            nudges = np.random.multivariate_normal(np.zeros(d_dim), nudge_cov, n_resample)
        except np.linalg.LinAlgError:
            diag_cov = np.diag(np.diag(nudge_cov))
            nudges = np.random.multivariate_normal(np.zeros(d_dim), diag_cov, n_resample)

        # Apply nudges only to non-elite particles
        self._particles[n_elite:] = self._particles[n_elite:] + nudges

        # Shrinkage toward mean only for non-elite particles
        if self.scale:
            old_center = mean.reshape(1, -1)
            self._particles[n_elite:] = self._particles[n_elite:] * self.a_param + old_center * (1 - self.a_param)

        # Clip all particles to bounds
        for j, name in enumerate(self._param_names):
            lo, hi = self.parameter_bounds[name]
            self._particles[:, j] = np.clip(self._particles[:, j], lo, hi)

    def _apply_annealed_jitter(self) -> None:
        """Apply continuous particle movement with decaying jitter magnitude.

        Jitter scale starts at annealed_jitter_initial and decays by
        annealed_jitter_decay each update, down to annealed_jitter_min.
        This prevents particle collapse early while allowing convergence later.
        """
        # Ensure we don't go below minimum jitter
        effective_scale = max(self._current_annealed_jitter_scale, self.annealed_jitter_min)

        # Apply independent per-dimension jitter (simpler than full covariance)
        for j, name in enumerate(self._param_names):
            lo, hi = self.parameter_bounds[name]
            std = np.std(self._particles[:, j])
            # Scale jitter by current scale and local std
            jitter = np.random.normal(0, std * effective_scale, self.num_particles)
            self._particles[:, j] = np.clip(self._particles[:, j] + jitter, lo, hi)

        # Decay the jitter scale for next update
        self._current_annealed_jitter_scale = max(
            self._current_annealed_jitter_scale * self.annealed_jitter_decay, self.annealed_jitter_min
        )

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
            annealed_jitter=self.annealed_jitter,
            annealed_jitter_initial=self.annealed_jitter_initial,
            annealed_jitter_min=self.annealed_jitter_min,
            annealed_jitter_decay=self.annealed_jitter_decay,
            elitism_ratio=self.elitism_ratio,
        )
        dist._param_names = self._param_names.copy()
        dist._particles = self._particles.copy()
        dist._weights = self._weights.copy()
        dist._current_annealed_jitter_scale = self._current_annealed_jitter_scale
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

    def select_maximum_likelihood(self, n: int) -> ParameterValues[np.ndarray]:
        """Select top-n particles by posterior weight for MaximumLikelihoodLocator.

        Returns the n particles with highest weights (representing maximum likelihood
        regions of the posterior), without random resampling.
        """
        n = min(n, self.num_particles)
        # Get indices of top-n weights (descending order)
        top_indices = np.argsort(self._weights)[-n:][::-1]
        samples = self._particles[top_indices]
        data = {name: samples[:, i] for i, name in enumerate(self._param_names)}
        return ParameterValues.from_mapping(self._param_names, data)

    def select_max_information_gain(self, candidates: np.ndarray, n: int) -> ParameterValues[np.ndarray]:
        """Select particles maximizing information gain at candidate locations for SBED.

        Scores particles by their prediction variance across candidates - particles
        that predict very differently at different locations contribute most to
        the entropy reduction computation in sequential Bayesian experiment design.

        Parameters
        ----------
        candidates : np.ndarray
            Array of candidate measurement locations.
        n : int
            Number of particles to select.

        Returns
        -------
        ParameterValues[np.ndarray]
            Selected particles that maximize expected information gain.
        """
        n = min(n, self.num_particles)

        # Compute predictions for all particles at all candidates
        arrays_in_order = [self._particles[:, j] for j in range(len(self._param_names))]
        # Shape: (n_candidates, n_particles)
        predictions = self.model.compute_vectorized_many(candidates, arrays_in_order)

        # Score each particle by weighted combination of:
        # 1. Posterior weight (particles must represent posterior mass)
        # 2. Prediction diversity (variance of predictions across candidates)
        pred_variance = np.var(predictions, axis=0)  # variance across candidates

        # Normalize variance to [0, 1] for combining with weights
        var_max = np.max(pred_variance)
        pred_variance_norm = pred_variance / var_max if var_max > 1e-12 else np.zeros_like(pred_variance)

        # Combined score: weight * (1 + prediction_diversity)
        # This ensures high-weight particles are preferred, but prediction
        # diversity can boost lower-weight particles if they are informative
        combined_scores = self._weights * (1.0 + pred_variance_norm)

        # Select top-n by combined score
        top_indices = np.argsort(combined_scores)[-n:][::-1]
        samples = self._particles[top_indices]
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
