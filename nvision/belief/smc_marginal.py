"""Sequential Monte Carlo (Particle Filter) belief distribution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numba
import numpy as np
from dotenv import load_dotenv
from numba import njit

from nvision.belief.abstract_marginal import AbstractMarginalDistribution, ParameterValues
from nvision.models.observation import Observation
from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.likelihood import likelihood_from_observation_model
from nvision.spectra.noise_model import NoiseSignalModel

# --- Environment-driven defaults ---------------------------------------------

load_dotenv()

NVISION_SMC_NUM_PARTICLES: int = int(os.getenv("NVISION_SMC_NUM_PARTICLES", "1000"))
NVISION_SMC_ESS_THRESHOLD: float = float(os.getenv("NVISION_SMC_ESS_THRESHOLD", "0.5"))
NVISION_SMC_A_PARAM: float = float(os.getenv("NVISION_SMC_A_PARAM", "0.98"))
NVISION_SMC_POINTS_PER_MIN_FEATURE: int = int(os.getenv("NVISION_SMC_POINTS_PER_MIN_FEATURE", "5"))
NVISION_SMC_MIN_EXPLORATION_FRAC: float = float(os.getenv("NVISION_SMC_MIN_EXPLORATION_FRAC", "0.01"))
NVISION_SMC_TEMPERING_FACTOR: float = float(os.getenv("NVISION_SMC_TEMPERING_FACTOR", "1.0"))


_EIG_CHUNK_SIZE: int = 64

# --- Numba helpers (particle weights / resampling) ----------------------------


@njit(cache=True)
def _weighted_mean_variance_1d(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted mean and variance of ``x`` with weights ``w`` (Welford's 1-pass algorithm)."""
    n = x.shape[0]
    if n == 0:
        return 0.0, 0.0

    mean = 0.0
    S = 0.0  # noqa: N806
    sum_weight = 0.0
    for i in range(n):
        wi = w[i]
        xi = x[i]
        sum_weight += wi
        if sum_weight > 0.0:
            delta = xi - mean
            mean += (wi / sum_weight) * delta
            S += wi * delta * (xi - mean)  # noqa: N806

    if sum_weight <= 0.0:
        return 0.0, 0.0

    return float(mean), float(S / sum_weight)


@njit(cache=True)
def _weighted_mean_axis0(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Column-wise weighted means for ``particles`` shaped ``(n, d)``."""
    # Ensure weights are the same float type as particles for np.dot
    w_typed = weights.astype(particles.dtype)
    sw = np.sum(w_typed)
    if sw <= 0.0:
        return np.zeros(particles.shape[1], dtype=particles.dtype)
    return np.dot(w_typed, particles) / sw


@njit(cache=True)
def _systematic_resample_indices(cumulative_sum: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Map systematic ``positions`` to indices along non-decreasing ``cumulative_sum``."""
    n = positions.shape[0]
    m = cumulative_sum.shape[0]
    indices = np.empty(n, dtype=np.int64)
    j = 0
    for i in range(n):
        pos = positions[i]
        while j < m and pos >= cumulative_sum[j]:
            j += 1
        # positions[i] is in [0, 1), cumulative_sum[-1] is 1.0, so j < m usually holds.
        indices[i] = j if j < m else m - 1
    return indices


@njit(cache=True, fastmath=True)
def _inverse_sum_squares(weights: np.ndarray) -> float:
    """Return ``1 / sum(w**2)`` (ESS denominator for normalized weights)."""
    # np.dot is significantly faster than a manual loop for this
    s = np.dot(weights, weights)
    if s <= 0.0:
        return 0.0
    return 1.0 / s


@njit(parallel=True, cache=True, fastmath=True)
def _chunk_argmax(eig_scores: np.ndarray, chunk_size: int) -> np.ndarray:
    """Find the argmax index within each chunk in parallel.

    Args:
        eig_scores: 1D array of EIG scores for all candidates.
        chunk_size: Number of candidates per chunk.

    Returns:
        1D int64 array of length ``ceil(len(eig_scores) / chunk_size)`` where
        each element is the *global* index of the argmax within that chunk.
    """
    n = eig_scores.shape[0]
    n_chunks = (n + chunk_size - 1) // chunk_size
    winners = np.empty(n_chunks, dtype=np.int64)
    for c in numba.prange(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, n)
        best_idx = start
        best_val = eig_scores[start]
        for k in range(start + 1, end):
            if eig_scores[k] > best_val:
                best_val = eig_scores[k]
                best_idx = k
        winners[c] = best_idx
    return winners


@njit(cache=True)
def _weighted_cdf(samples: np.ndarray, weights: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """Compute empirical weighted CDF using Numba for speed.

    Args:
        samples: 1D array of particle values.
        weights: 1D array of normalized particle weights.
        x_query: 1D array of values to evaluate the CDF at.

    Returns:
        1D array of CDF values in [0, 1].
    """
    sort_idx = np.argsort(samples)
    sorted_samples = samples[sort_idx]
    sorted_weights = weights[sort_idx]
    cdf_vals = np.cumsum(sorted_weights)

    # Use Numba-compatible interpolation
    return np.interp(x_query, sorted_samples, cdf_vals)


@dataclass
class SMCMarginalDistribution(AbstractMarginalDistribution):
    """Belief distribution using Sequential Monte Carlo (Particle Filter).

    Maintains a joint posterior over parameters using a set of weighted particles.
    Resampling uses systematic resampling (low variance) followed by nudging with a
    multivariate Gaussian kernel and Liu-West shrinkage (contraction) to maintain
    diversity and preserve distribution moments.

    Parameters
    ----------
    a_param : float
        Contraction parameter. After nudging, particles are contracted
        (1 - a_param) of the distance toward the mean. Default 0.98.
    auto_resample : bool
        If True, resample automatically when ESS drops below threshold.
        Set False to let the locator trigger resampling manually.
    """

    parameter_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    num_particles: int = NVISION_SMC_NUM_PARTICLES
    ess_threshold: float = NVISION_SMC_ESS_THRESHOLD
    a_param: float = NVISION_SMC_A_PARAM
    noise_model: NoiseSignalModel | None = None
    auto_resample: bool = True
    resample_delay: int = 0
    priors: dict[str, tuple[float, float]] | None = None
    min_exploration_frac: float = NVISION_SMC_MIN_EXPLORATION_FRAC
    tempering_factor: float = NVISION_SMC_TEMPERING_FACTOR

    _cached_cov: np.ndarray | None = field(init=False, default=None, repr=False)
    _cov_step: int = field(init=False, default=-1, repr=False)

    _particles: np.ndarray = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)
    _step_count: int = field(init=False, repr=False, default=0)
    _param_names: list[str] = field(init=False, repr=False)
    _noise_param_slice: slice | None = field(init=False, repr=False, default=None)
    _current_candidates: np.ndarray = field(init=False, repr=False)
    _global_grid: np.ndarray = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _d_signal: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._param_names = list(self.model.parameter_names())
        if self.noise_model is not None:
            # Append noise parameters to the state space
            for name in self.noise_model.spec.names:
                if name not in self._param_names:
                    self._param_names.append(name)

        # Initialize particles uniformly within bounds
        d_dim = len(self._param_names)
        self._particles = np.zeros((self.num_particles, d_dim), dtype=FLOAT_DTYPE)

        for i, name in enumerate(self._param_names):
            if name not in self.parameter_bounds:
                raise ValueError(f"Missing bounds for parameter: {name}")
            lo, hi = self.parameter_bounds[name]

            if self.priors and name in self.priors:
                mean, std = self.priors[name]
                self._particles[:, i] = np.random.normal(mean, std, self.num_particles)
                self._particles[:, i] = np.clip(self._particles[:, i], lo, hi)
            else:
                self._particles[:, i] = np.random.uniform(lo, hi, self.num_particles)

        self._weights = (np.ones(self.num_particles, dtype=FLOAT_DTYPE) / self.num_particles).astype(FLOAT_DTYPE)
        self._step_count = 0
        self._rng = np.random.default_rng()

        # Detect noise param dimensions
        if self.noise_model is not None:
            noise_names = set(self.noise_model.spec.names)
            indices = [i for i, name in enumerate(self._param_names) if name in noise_names]
            if indices:
                self._noise_param_slice = slice(min(indices), max(indices) + 1)

        # Cache the number of signal dimensions (everything before noise params)
        self._d_signal = (
            self._noise_param_slice.start if self._noise_param_slice is not None else len(self._param_names)
        )

        # Initialize the first epoch-based candidate grid.
        # Spacing must resolve the narrowest possible signal feature so that EIG
        # has a meaningful score at the true frequency even when the local slope
        # grids are focused on the wrong location.
        #
        # signal_min_span returns the minimum signal feature width (e.g. 4 × linewidth_min).
        # We require POINTS_PER_MIN_FEATURE grid points within that span.
        # Formula: n = ceil(domain_width / min_span * POINTS_PER_MIN_FEATURE)
        POINTS_PER_MIN_FEATURE: int = NVISION_SMC_POINTS_PER_MIN_FEATURE  # noqa: N806
        f_lo, f_hi = self.parameter_bounds["frequency"]
        domain_width = float(f_hi - f_lo)
        min_span = self.model.signal_min_span(domain_width)
        if min_span is None or min_span <= 0:
            raise ValueError(
                f"{type(self.model).__name__}.signal_min_span({domain_width}) returned {min_span!r}. "
                "Implement signal_min_span() to return the minimum signal feature width in Hz."
            )
        n_global = int(np.ceil(domain_width / min_span * POINTS_PER_MIN_FEATURE))

        self._global_grid = np.linspace(f_lo, f_hi, n_global).astype(np.float32)
        self._generate_epoch_candidates()

    def update(self, obs: Observation) -> None:
        self.last_obs = obs
        self.resampled = False

        # 1. Compute likelihood for all particles (vectorized model evaluation)
        arrays_in_order = [self._particles[:, j] for j in range(self._d_signal)]
        predicted = self.model.compute_vectorized(obs.x, *arrays_in_order)

        if self.noise_model is not None and self._noise_param_slice is not None:
            # Epistemic spread is passed to the noise model for its own tempering logic
            sigma_epistemic = float(np.std(predicted))
            noise_arrays = [
                self._particles[:, j] for j in range(self._noise_param_slice.start, self._noise_param_slice.stop)
            ]
            residuals = obs.signal_value - predicted
            log_liks = self.noise_model.composite_log_likelihood(predicted, residuals, noise_arrays, sigma_epistemic)

            # Numerically stable exponentiation
            log_liks -= np.max(log_liks)
            likelihoods = np.exp(log_liks).astype(FLOAT_DTYPE, copy=False)
        else:
            # Use true aleatoric noise; stability handled by ESS-triggered resampling
            likelihoods = likelihood_from_observation_model(
                obs_y=obs.signal_value,
                predicted=predicted,
                noise_std=obs.noise_std,
                frequency_noise_model=obs.frequency_noise_model,
                tempering_factor=self.tempering_factor,
            ).astype(FLOAT_DTYPE, copy=False)

        # 2. Update weights — paper-compliant likelihood only (Eq. S3) [cite: 117]
        self._weights *= likelihoods
        self._step_count += 1
        self._cov_step = -1

        # 3. Normalize weights safely without redundant array casting allocations
        weight_sum = np.sum(self._weights)
        if weight_sum > 1e-100:
            self._weights = (self._weights / weight_sum).astype(FLOAT_DTYPE, copy=False)
        else:
            # Total filter collapse: likelihood was zero everywhere
            self._weights = np.ones(self.num_particles, dtype=FLOAT_DTYPE) / self.num_particles

        # 4. Resample if Effective Sample Size (ESS) is too low [cite: 198, 199]
        ess = _inverse_sum_squares(self._weights)
        if (
            self.auto_resample
            and ess < self.ess_threshold * self.num_particles
            and self._step_count >= self.resample_delay
        ):
            self._resample()

    def batch_update(self, observations: list[Observation]) -> None:
        if not observations:
            return

        self.last_obs = observations[-1]
        self.resampled = False

        arrays_in_order = [self._particles[:, j] for j in range(self._d_signal)]
        log_weights = np.zeros(self.num_particles, dtype=FLOAT_DTYPE)

        if self.noise_model is not None and self._noise_param_slice is not None:
            # Path A: Custom composite noise model evaluating epistemic spread
            noise_arrays = [
                self._particles[:, j] for j in range(self._noise_param_slice.start, self._noise_param_slice.stop)
            ]
            for obs in observations:
                predicted = self.model.compute_vectorized(obs.x, *arrays_in_order)
                sigma_epistemic = float(np.std(predicted))
                residuals = obs.signal_value - predicted

                # Accumulated directly; inner max-subtraction removed for speed
                log_weights += self.noise_model.composite_log_likelihood(
                    predicted, residuals, noise_arrays, sigma_epistemic
                )
        else:
            # Path B: Standard/Frequency noise model
            if any(obs.frequency_noise_model is not None for obs in observations):
                # Fallback to match custom frequency noise transformations sequentially
                for obs in observations:
                    predicted = self.model.compute_vectorized(obs.x, *arrays_in_order)
                    lik = likelihood_from_observation_model(
                        obs_y=obs.signal_value,
                        predicted=predicted,
                        noise_std=obs.noise_std,
                        frequency_noise_model=obs.frequency_noise_model,
                        tempering_factor=self.tempering_factor,
                    )
                    log_weights += np.log(np.maximum(lik, 1e-100))
            else:
                # Optimized pure Gaussian path via vectorized matrix operations
                all_xs = np.array([obs.x for obs in observations], dtype=FLOAT_DTYPE)
                predictions = self.model.compute_vectorized_many(all_xs, arrays_in_order)

                ys = np.array([obs.signal_value for obs in observations], dtype=FLOAT_DTYPE)
                sigmas = np.array([max(float(obs.noise_std), 1e-9) for obs in observations], dtype=FLOAT_DTYPE)
                if self.tempering_factor != 1.0:
                    sigmas *= np.sqrt(self.tempering_factor)

                residuals = ys[:, None] - predictions
                log_liks_matrix = -0.5 * (residuals / sigmas[:, None]) ** 2
                log_weights += log_liks_matrix.sum(axis=0)

        self._step_count += len(observations)

        # Convert log-weights back to normalized standard weights safely
        log_weights -= np.max(log_weights)
        raw_weights = np.exp(log_weights) * self._weights.astype(FLOAT_DTYPE)
        weight_sum = np.sum(raw_weights)

        # Threshold aligned to 1e-100 to match standard update behavior
        if weight_sum > 1e-100:
            self._weights = (raw_weights / weight_sum).astype(FLOAT_DTYPE)
        else:
            self._weights = (np.ones(self.num_particles, dtype=FLOAT_DTYPE) / self.num_particles).astype(FLOAT_DTYPE)

        # Evaluate Effective Sample Size (ESS) for resampling
        ess = _inverse_sum_squares(self._weights)
        if (
            self.auto_resample
            and ess < self.ess_threshold * self.num_particles
            and self._step_count >= self.resample_delay
        ):
            self._resample()

    def get_candidates(self) -> np.ndarray:
        """Return the current epoch's slope-targeted candidate grid."""
        return self._current_candidates

    def _generate_epoch_candidates(self) -> None:
        """Generate a dense slope-targeted grid and cache it for the current epoch.

        Grid targets the steepest slopes (center ± linewidth) of the 3 hyperfine
        dips. Resolution and search windows scale with posterior uncertainty.
        """
        # Use unit-space estimates and uncertainties to avoid physical-unit mismatch in subclasses
        estimates = self._estimates_unit()
        uncertainties = self._uncertainty_unit()

        f_b = estimates["frequency"]
        df_hf = estimates["split"]

        # 1. Determine linewidth Omega (HWHM)
        if "linewidth" in estimates:
            omega = estimates["linewidth"]
        elif "fwhm_total" in estimates:
            omega = estimates["fwhm_total"] / 2.0
        else:
            raise KeyError("Linewidth parameter ('linewidth' or 'fwhm_total') not found in estimates")

        # 2. Extract uncertainties
        sigma_f = uncertainties["frequency"]
        if "linewidth" in uncertainties:
            sigma_omega = uncertainties["linewidth"]
        elif "fwhm_total" in uncertainties:
            sigma_omega = uncertainties["fwhm_total"] / 2.0
        else:
            raise KeyError("Linewidth uncertainty ('linewidth' or 'fwhm_total') not found in uncertainties")

        # 3. Compute effective uncertainty and resolution
        sigma_eff = np.sqrt(sigma_f**2 + sigma_omega**2)
        # 0.01 MHz = 10 kHz minimum step
        delta_d = max(sigma_eff / 30.0, 10000.0)
        half_width = 3 * sigma_eff

        # 4. Slope Targeting: 6 locations (2 per dip)
        centers = [f_b - df_hf, f_b, f_b + df_hf]
        slopes = []
        for c in centers:
            slopes.append(c - omega)
            slopes.append(c + omega)

        local_grids = []
        f_lo, f_hi = self.parameter_bounds["frequency"]
        for s in slopes:
            start = max(s - half_width, f_lo)
            stop = min(s + half_width, f_hi)
            if start >= stop:
                continue
            local_grids.append(np.arange(start, stop + delta_d, delta_d))

        # 5. Merge with pre-computed global grid; clip everything to [f_lo, f_hi]
        merged = np.concatenate([*local_grids, self._global_grid]) if local_grids else self._global_grid
        self._current_candidates = np.unique(np.clip(merged, f_lo, f_hi)).astype(np.float32)

    def _resample(self) -> None:
        """Systematic resampling with Gaussian nudging and Liu-West shrinkage.

        All particles are resampled using systematic sampling (low variance),
        then nudged with a multivariate Gaussian kernel based on the current
        particle covariance. Shrinkage toward the mean is optionally applied
        to preserve the distribution variance.
        """
        self.resampled = True
        d_dim = len(self._param_names)

        # 1. Systematic Resampling
        # Map systematic positions to indices along non-decreasing cumulative_sum
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        new_indices = _systematic_resample_indices(np.cumsum(self._weights), positions)

        # 2. Compute covariance from the pre-resample distribution for nudging.
        # Use cached covariance for the current step to avoid duplicate computation.
        mean = _weighted_mean_axis0(self._particles, self._weights)
        cov = self._cached_covariance()

        # 3. Update particles and reset weights
        self._particles = self._particles[new_indices]
        self._weights = (np.ones(self.num_particles, dtype=FLOAT_DTYPE) / self.num_particles).astype(FLOAT_DTYPE)

        # 4. Compute nudge covariance: (1 - a^2) * Sigma
        nudge_cov = (1 - self.a_param**2) * cov

        # 5. Enforce minimum exploration variance based on parameter ranges.
        # This prevents the filter from getting stuck in a tiny volume.
        for j, name in enumerate(self._param_names):
            lo, hi = self.parameter_bounds[name]
            min_var = ((hi - lo) * self.min_exploration_frac) ** 2
            nudge_cov[j, j] = max(float(nudge_cov[j, j]), float(min_var))

        # Make nudge_cov perfectly symmetric and strictly positive definite.
        # Adding a tiny absolute and relative diagonal jitter ensures
        # we don't hit LinAlgError due to float32 numerical precision.
        nudge_cov = 0.5 * (nudge_cov + nudge_cov.T)

        # Enforce relative and absolute threshold for eigenvalues to restrict
        # the condition number to 1e6 (ratio of max to min eigenvalue).
        # Since FLOAT_DTYPE is np.float32, the machine epsilon is ~1.19e-7,
        # so any eigenvalue smaller than 1e-6 * max_eig will be lost in float32 rounding.
        eigvals, eigvecs = np.linalg.eigh(nudge_cov)
        max_eig = np.max(eigvals)
        min_eigval = max(1e-11, 1e-6 * max_eig) if max_eig > 0 else 1e-11
        eigvals = np.maximum(eigvals, min_eigval)
        nudge_cov = (eigvecs * eigvals) @ eigvecs.T

        # 6. Shrinkage contraction toward mean (Liu-West)
        # MUST happen before nudging so we don't shrink the added noise
        old_center = mean.reshape(1, -1)
        self._particles = (self._particles * self.a_param + old_center * (1 - self.a_param)).astype(
            FLOAT_DTYPE, copy=False
        )

        # 7. Apply Nudge (Multivariate Gaussian) — reuse the cached RNG
        try:
            nudges = self._rng.multivariate_normal(
                np.zeros(d_dim, dtype=FLOAT_DTYPE), nudge_cov, self.num_particles, method="cholesky"
            ).astype(FLOAT_DTYPE, copy=False)
        except np.linalg.LinAlgError:
            # Fall back to the highly robust SVD method if Cholesky still fails
            nudges = self._rng.multivariate_normal(
                np.zeros(d_dim, dtype=FLOAT_DTYPE), nudge_cov, self.num_particles, method="svd"
            ).astype(FLOAT_DTYPE, copy=False)
        self._particles += nudges

        # 8. Clip all particles to bounds
        for j, name in enumerate(self._param_names):
            lo, hi = self.parameter_bounds[name]
            self._particles[:, j] = np.clip(self._particles[:, j], lo, hi)

        # 9. Update cached candidate grid for the next epoch
        self._generate_epoch_candidates()

    def _marginal_std(self, dim_idx: int) -> float:
        _, var = _weighted_mean_variance_1d(self._particles[:, dim_idx], self._weights)
        return float(np.sqrt(max(0.0, var)))

    def _estimates_unit(self) -> dict[str, float]:
        """Return parameter estimates in internal unit/belief space."""
        means = _weighted_mean_axis0(self._particles, self._weights)
        return {name: float(means[i]) for i, name in enumerate(self._param_names)}

    def estimates(self) -> dict[str, float]:
        """Return parameter estimates (weighted mean)."""
        return self._estimates_unit()

    def _uncertainty_unit(self) -> ParameterValues[float]:
        """Return parameter uncertainties (std dev) in internal unit/belief space.

        Computes all marginal variances in a single vectorized pass instead of
        calling the 1-D Numba kernel once per dimension.
        """
        w = self._weights
        sw = w.sum()
        if sw <= 0.0:
            stds = {name: 0.0 for name in self._param_names}
            return ParameterValues.from_mapping(self._param_names, stds)
        p = self._particles  # (N, d)
        mean = (w @ p) / sw  # (d,)
        diff = p - mean  # (N, d)
        var = (w @ (diff**2)) / sw  # (d,)  weighted variance
        stds = {name: float(np.sqrt(max(0.0, var[i]))) for i, name in enumerate(self._param_names)}
        return ParameterValues.from_mapping(self._param_names, stds)

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        return self._uncertainty_unit()

    def entropy(self) -> float:
        # Simple Kozachenko-Leonenko nearest-neighbor entropy estimator could go here.
        # For now, approximate via a Gaussian assumption on the particles.
        cov = self.covariance_matrix()
        d_dim = cov.shape[0]

        # Regularize to ensure valid log-determinant even if singular
        reg = 1e-12 * (np.trace(cov) / d_dim + 1e-15)
        cov.flat[:: d_dim + 1] += reg

        _, logdet = np.linalg.slogdet(cov)
        return float(0.5 * logdet + 0.5 * d_dim * (1 + np.log(2 * np.pi)))

    def _cached_covariance(self) -> np.ndarray:
        """Calculate and cache weighted covariance of particles for the current step."""
        if self._cov_step == self._step_count and self._cached_cov is not None:
            return self._cached_cov

        w = self._weights
        sw = w.sum()
        if sw <= 0.0:
            cov = np.zeros((len(self._param_names), len(self._param_names)), dtype=FLOAT_DTYPE)
        else:
            p = self._particles
            mean = (w @ p) / sw
            diff = p - mean
            cov = (diff.T @ (w[:, None] * diff)) / sw

        self._cached_cov = cov
        self._cov_step = self._step_count
        return cov

    def covariance_matrix(self) -> np.ndarray:
        """Return full covariance matrix of particle distribution.

        Returns a (d, d) array where d is the number of parameters.
        """
        # Return cached covariance if available for the current step.
        return self._cached_covariance()

    def correlation_matrix(self) -> np.ndarray:
        """Return correlation matrix (normalized covariance).

        Returns a (d, d) array with values in [-1, 1].
        Diagonal entries are always 1.0.
        """
        cov = self.covariance_matrix()
        d_dim = cov.shape[0]

        # Regularize diagonal to avoid division by zero if variance is zero
        cov.flat[:: d_dim + 1] += 1e-20

        stds = np.sqrt(np.diag(cov))
        corr = cov / np.outer(stds, stds)
        # Clip to handle numerical errors
        return np.clip(corr, -1.0, 1.0)

    def generalized_variance(self) -> float:
        """Return determinant of covariance matrix (generalized variance).

        This is a scalar measure of total uncertainty volume.
        Smaller values indicate tighter posterior concentration.
        """
        cov = self.covariance_matrix()
        d_dim = cov.shape[0]

        # Regularize to ensure valid log-determinant
        reg = 1e-12 * (np.trace(cov) / d_dim + 1e-15)
        cov.flat[:: d_dim + 1] += reg

        _, logdet = np.linalg.slogdet(cov)
        return float(np.exp(logdet))

    def converged(self, threshold: float) -> bool:
        return all(u < threshold for u in self.uncertainty().values())

    def copy(self) -> SMCMarginalDistribution:
        dist = SMCMarginalDistribution(
            model=self.model,
            parameter_bounds=self.parameter_bounds.copy(),
            num_particles=self.num_particles,
            ess_threshold=self.ess_threshold,
            a_param=self.a_param,
            last_obs=self.last_obs,
            noise_model=self.noise_model,
            auto_resample=self.auto_resample,
            resample_delay=self.resample_delay,
            priors=self.priors,
            min_exploration_frac=self.min_exploration_frac,
            tempering_factor=self.tempering_factor,
        )
        dist._param_names = self._param_names.copy()
        dist._particles = self._particles.copy()
        dist._weights = self._weights.copy()
        dist._step_count = self._step_count
        dist.resampled = self.resampled
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

    def select_max_information_gain(self, candidates: np.ndarray, n: int, noise_std: float = 0.02) -> np.ndarray:
        """Select the top-n candidate locations by expected information gain.

        Evaluates EIG across ``candidates`` in chunks of :data:`_EIG_CHUNK_SIZE`
        using :meth:`expected_information_gain`, then uses a Numba parallel
        ``prange`` helper to find each chunk's best candidate.

        Args:
            candidates: 1D array of candidate measurement locations.
            n: Number of top candidates to return.
            noise_std: Measurement noise floor for EIG calculation.

        Returns:
            1D numpy array of up to *n* candidate locations ranked by EIG
            (highest first).
        """
        if len(candidates) == 0:
            return candidates[:0]

        # Evaluate EIG over all candidates in one vectorized call.
        eig_scores = self.expected_information_gain(candidates, noise_std=noise_std).astype(FLOAT_DTYPE)

        # Boltzmann sampling over chunk winners to avoid getting stuck at a
        # single numerical noise peak (same logic, now over the full grid).
        winner_indices = _chunk_argmax(eig_scores, _EIG_CHUNK_SIZE)
        temp = 0.01
        winner_scores = eig_scores[winner_indices]
        shifted_scores = (winner_scores - np.max(winner_scores)) / temp
        probs = np.exp(shifted_scores)
        probs /= np.sum(probs)

        best_chunk_order = np.random.choice(
            len(winner_indices), size=min(n, len(winner_indices)), replace=False, p=probs
        )
        best_chunk_order = best_chunk_order[np.argsort(winner_scores[best_chunk_order])][::-1]
        best_indices = winner_indices[best_chunk_order]

        return candidates[best_indices]

    def expected_information_gain(self, candidates: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        """Compute the approximate expected information gain for candidate locations.

        Uses the approximation:
        EIG(d) ≈ 1/2 * ln(1 + sigma_theta^2 / sigma_eta^2)
        where sigma_theta^2 is the prediction variance (disagreement for that frequency across particles)
        and sigma_eta^2 is the measurement noise variance.
        """
        arrays_in_order = [self._particles[:, j] for j in range(len(self._param_names))]
        w = self._weights  # already float32, already normalized
        n_candidates = len(candidates)
        var_pred = np.empty(n_candidates, dtype=FLOAT_DTYPE)

        # Optimization: Process EIG scoring in chunks to prevent massive (n_candidates, n_particles)
        # memory allocations, reducing peak memory usage from ~1.1GB to ~2MB for 50k candidates * 1k particles.
        for start in range(0, n_candidates, _EIG_CHUNK_SIZE):
            end = min(start + _EIG_CHUNK_SIZE, n_candidates)
            cand_chunk = candidates[start:end]

            # shape: (chunk_size, n_particles) — uses fastmath kernel (acquisition path only)
            predictions = self.model.compute_vectorized_many_fast(cand_chunk, arrays_in_order)

            mean_pred = predictions @ w  # (chunk_size,)
            diff = predictions - mean_pred[:, None]  # (chunk_size, n_particles)
            var_pred[start:end] = (diff**2) @ w  # (chunk_size,)

        return var_pred

    def expected_information_gain_jax(self, candidates: Any, noise_std: float = 0.02) -> Any:
        """Compute the approximate expected information gain using JAX for auto-differentiation."""

        # Broadcast candidates to shape (n_candidates, 1)
        x_2d = jnp.atleast_1d(candidates)[:, None]

        # Construct params with shape (1, n_particles)
        data = {}
        for i, name in enumerate(self._param_names):
            data[name] = jnp.array(self._particles[:, i])[None, :]

        params = self.model.spec.params_cls(**data)

        # Evaluate model: predictions shape is (n_candidates, n_particles)
        predictions = self.model.compute_jax(x_2d, params)

        # Weighted variance
        weights = jnp.array(self._weights)[None, :]
        mean_pred = jnp.sum(predictions * weights, axis=1, keepdims=True)
        var_pred = jnp.sum(((predictions - mean_pred) ** 2) * weights, axis=1)

        noise_var = noise_std**2
        return 0.5 * jnp.log(1.0 + var_pred / (noise_var + 1e-12))

    def narrow_scan_parameter_physical_bounds(self, param_name: str, new_lo: float, new_hi: float) -> None:
        """Shrink physical bounds and clip particles into the new window."""
        if param_name in self.parameter_bounds:
            old_lo, old_hi = self.parameter_bounds[param_name]
            lo, hi = max(old_lo, new_lo), min(old_hi, new_hi)
            self.parameter_bounds[param_name] = (lo, hi)

            # Immediately snap particles into the new tighter bounds
            if param_name in self._param_names:
                idx = self._param_names.index(param_name)
                self._particles[:, idx] = np.clip(self._particles[:, idx], lo, hi)

            # If the scan parameter was narrowed, update the global grid and candidates
            if param_name == "frequency":
                POINTS_PER_MIN_FEATURE: int = NVISION_SMC_POINTS_PER_MIN_FEATURE  # noqa: N806
                domain_width = float(hi - lo)
                min_span = self.model.signal_min_span(domain_width)
                if min_span is not None and min_span > 0:
                    n_global = int(np.ceil(domain_width / min_span * POINTS_PER_MIN_FEATURE))
                    self._global_grid = np.linspace(lo, hi, n_global).astype(np.float32)
                self._generate_epoch_candidates()

    @property
    def physical_param_bounds(self) -> dict[str, tuple[float, float]]:
        """Physical bounds for each parameter (same as parameter_bounds)."""
        return self.parameter_bounds

    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        from scipy.stats import gaussian_kde, norm

        idx = self._param_names.index(param_name)
        samples = self._particles[:, idx]

        # Use weighted variance (via existing Numba helper) for the collapse check
        _, var = _weighted_mean_variance_1d(samples, self._weights)
        if var < 1e-20:
            mean_val, _ = _weighted_mean_variance_1d(samples, self._weights)
            lo, hi = self.parameter_bounds[param_name]
            bw = (hi - lo) * 1e-3
            return norm.pdf(x, loc=mean_val, scale=max(bw, 1e-10))

        kde = gaussian_kde(samples, weights=self._weights)
        return kde.evaluate(x)

    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        """Evaluate the marginal Cumulative Distribution Function (CDF)."""
        idx = self._param_names.index(param_name)
        samples = self._particles[:, idx]
        return _weighted_cdf(samples, self._weights, x)
