"""Sequential Monte Carlo (Particle Filter) belief distribution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numba
import numpy as np
from dotenv import load_dotenv
from numba import njit, prange

from nvision.belief.abstract_marginal import AbstractMarginalDistribution, ParameterValues
from nvision.belief.coordinate import RescaleMap
from nvision.models.observation import Observation
from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.likelihood import likelihood_from_observation_model
from nvision.spectra.noise_model import NoiseSignalModel
from nvision.spectra.numba_kernels import (
    nv_center_lorentzian_eig_variance,
    nv_center_pseudo_voigt_eig_variance,
)

# --- Environment-driven defaults ---------------------------------------------

load_dotenv()

NVISION_SMC_NUM_PARTICLES: int = int(os.getenv("NVISION_SMC_NUM_PARTICLES", "1000"))
NVISION_SMC_ESS_THRESHOLD: float = float(os.getenv("NVISION_SMC_ESS_THRESHOLD", "0.2"))
NVISION_SMC_A_PARAM: float = float(os.getenv("NVISION_SMC_A_PARAM", "0.98"))
NVISION_SMC_POINTS_PER_MIN_FEATURE: int = int(os.getenv("NVISION_SMC_POINTS_PER_MIN_FEATURE", "5"))
NVISION_SMC_MIN_EXPLORATION_FRAC: float = float(os.getenv("NVISION_SMC_MIN_EXPLORATION_FRAC", "0.01"))
NVISION_SMC_TEMPERING_FACTOR: float = float(os.getenv("NVISION_SMC_TEMPERING_FACTOR", "1.0"))

# Maximum particles used for EIG / acquisition scoring.
# EIG only estimates prediction variance — that converges with ~200–500 particles.
# Using all N particles at N=10k produces an 80 MB matrix per EIG call.
# Subsampling keeps the matrix small (< 4 MB) with negligible quality loss.
NVISION_SMC_EIG_PARTICLES: int = int(os.getenv("NVISION_SMC_EIG_PARTICLES", "500"))


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
    """Column-wise weighted means for ``particles`` shaped ``(n, d)``.

    Explicit per-column loops instead of ``np.dot`` so the kernel accepts any
    memory layout (particles are stored F-order; columns are contiguous).
    """
    n, d = particles.shape
    out = np.zeros(d, dtype=particles.dtype)
    sw = 0.0
    for i in range(n):
        sw += weights[i]
    if sw <= 0.0:
        return out
    for j in range(d):
        acc = 0.0
        for i in range(n):
            acc += weights[i] * particles[i, j]
        out[j] = acc / sw
    return out


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


@njit(parallel=True, cache=True, fastmath=True)
def _weighted_variance_rows(predictions: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute row-wise weighted variance of 2D matrix predictions with weights w."""
    m = predictions.shape[0]
    n = predictions.shape[1]
    out = np.empty(m, dtype=predictions.dtype)
    for i in prange(m):
        sum_p = 0.0
        sum_p2 = 0.0
        for j in range(n):
            val = predictions[i, j]
            wi = w[j]
            sum_p += wi * val
            sum_p2 += wi * val * val
        v = sum_p2 - sum_p * sum_p
        out[i] = v if v > 0.0 else 0.0
    return out


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
    noise_discount_factor: float = 0.99
    noise_prior_strength: float = 10.0

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
    _eig_kernel_type: str = field(init=False, repr=False, default="generic")
    # Observation history as flat buffers (amortized growth). Only (x, y) are
    # ever consumed from history (dip detection), so full Observation objects
    # are not stored — see the _observations compatibility property.
    _obs_x_arr: np.ndarray | None = field(init=False, repr=False, default=None)
    _obs_y_arr: np.ndarray | None = field(init=False, repr=False, default=None)
    _obs_count: int = field(init=False, repr=False, default=0)
    _scratch_logw: np.ndarray | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._use_rao_blackwell_noise = False
        if self.noise_model is not None and "noise_sigma" in self.noise_model.spec.names:
            self._use_rao_blackwell_noise = True

        self._param_names = list(self.model.parameter_names())
        if self.noise_model is not None and not self._use_rao_blackwell_noise:
            # Append noise parameters to the state space
            for name in self.noise_model.spec.names:
                if name not in self._param_names:
                    self._param_names.append(name)

        # Initialize particles uniformly within bounds.
        # Column-major (Fortran) layout: per-parameter columns are the access
        # unit everywhere (model evaluation, EIG, marginals), so F-order makes
        # every ``_particles[:, j]`` a zero-copy contiguous view and makes
        # ``_particles.T`` C-contiguous for the EIG kernels.
        d_dim = len(self._param_names)
        self._particles = np.zeros((self.num_particles, d_dim), dtype=FLOAT_DTYPE, order="F")

        for i, name in enumerate(self._param_names):
            if name not in self.parameter_bounds:
                raise ValueError(f"Missing bounds for parameter: {name}")
            lo, hi = self.parameter_bounds[name]

            if self.priors and name in self.priors:
                prior_val = self.priors[name]
                if isinstance(prior_val, tuple) and len(prior_val) >= 2 and prior_val[0] == "sin^2":
                    k = prior_val[1]
                    phys_bounds = getattr(self, "physical_param_bounds", None)
                    if phys_bounds and name in phys_bounds:
                        f_min, f_max = phys_bounds[name]
                    else:
                        f_min, f_max = lo, hi

                    # Rejection sampling in physical space
                    sampled = []
                    while len(sampled) < self.num_particles:
                        candidates = np.random.uniform(f_min, f_max, self.num_particles)
                        probs = np.sin(k * (candidates - f_min)) ** 2
                        u = np.random.uniform(0.0, 1.0, self.num_particles)
                        accepted = candidates[u < probs]
                        sampled.extend(accepted)
                    sampled = np.array(sampled[: self.num_particles])

                    # Map back to unit space if in UnitCubeSMCMarginalDistribution
                    if phys_bounds and name in phys_bounds:
                        self._particles[:, i] = (sampled - f_min) / (f_max - f_min)
                    else:
                        self._particles[:, i] = sampled
                else:
                    mean, std = prior_val
                    self._particles[:, i] = np.random.normal(mean, std, self.num_particles)
                    self._particles[:, i] = np.clip(self._particles[:, i], lo, hi)
            else:
                self._particles[:, i] = np.random.uniform(lo, hi, self.num_particles)

        self._weights = (np.ones(self.num_particles, dtype=FLOAT_DTYPE) / self.num_particles).astype(FLOAT_DTYPE)
        self._step_count = 0
        self._rng = np.random.default_rng()
        self._obs_x_arr = np.empty(256, dtype=np.float64)
        self._obs_y_arr = np.empty(256, dtype=np.float64)
        self._obs_count = 0
        self._scratch_logw = np.empty(self.num_particles, dtype=FLOAT_DTYPE)
        self._dip_centers = []

        if getattr(self, "_use_rao_blackwell_noise", False):
            prior_bounds = self.noise_model.spec.bounds
            lo, hi = prior_bounds.get("noise_sigma", (0.01, 0.1))
            nominal_sigma = float(np.sqrt(max(lo * hi, 0.0)))
            self._noise_alphas = np.full(self.num_particles, self.noise_prior_strength, dtype=np.float32)
            self._noise_betas = np.full(
                self.num_particles, self.noise_prior_strength * (nominal_sigma**2), dtype=np.float32
            )

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

        # Cache which fused EIG-variance kernel to use — avoids isinstance checks
        # and module imports inside the per-step hot path.
        try:
            from nvision.spectra.nv_center import NVCenterLorentzianModel, NVCenterVoigtModel

            if isinstance(self.model, NVCenterLorentzianModel):
                self._eig_kernel_type = "lorentzian"
            elif isinstance(self.model, NVCenterVoigtModel):
                self._eig_kernel_type = "voigt"
            else:
                self._eig_kernel_type = "generic"
        except ImportError:
            self._eig_kernel_type = "generic"

    def update(self, obs: Observation) -> None:
        self._append_observation(obs.x, obs.signal_value)
        self.last_obs = obs
        self.resampled = False

        # 1. Compute likelihood for all particles (vectorized model evaluation)
        arrays_in_order = [self._particles[:, j] for j in range(self._d_signal)]
        predicted = self.model.compute_vectorized(obs.x, *arrays_in_order)

        if getattr(self, "_use_rao_blackwell_noise", False):
            sigmas = self._noise_betas / self._noise_alphas
            np.sqrt(sigmas, out=sigmas)
            np.maximum(sigmas, 1e-9, out=sigmas)
            residuals = obs.signal_value - predicted
            log_liks = -0.5 * (residuals / sigmas) ** 2 - 0.5 * np.log(sigmas**2)
            if self.tempering_factor != 1.0:
                log_liks *= self.tempering_factor
            # In-place Inverse-Gamma posterior update (residuals are dead after this)
            self._noise_alphas *= self.noise_discount_factor
            self._noise_alphas += 0.5
            np.square(residuals, out=residuals)
            residuals *= 0.5
            self._noise_betas *= self.noise_discount_factor
            self._noise_betas += residuals

        elif self.noise_model is not None and self._noise_param_slice is not None:
            # Epistemic spread is passed to the noise model for its own tempering logic
            sigma_epistemic = float(np.std(predicted))
            noise_arrays = [
                self._particles[:, j] for j in range(self._noise_param_slice.start, self._noise_param_slice.stop)
            ]
            residuals = obs.signal_value - predicted
            log_liks = self.noise_model.composite_log_likelihood(predicted, residuals, noise_arrays, sigma_epistemic)
        else:
            if getattr(obs, "frequency_noise_model", None) is not None:
                # Fallback to match custom frequency noise transformations sequentially
                liks = likelihood_from_observation_model(
                    obs_y=obs.signal_value,
                    predicted=predicted,
                    noise_std=obs.noise_std,
                    frequency_noise_model=obs.frequency_noise_model,
                    tempering_factor=self.tempering_factor,
                )
                log_liks = np.log(np.maximum(liks, 1e-30))
            else:
                # Optimized pure Gaussian path, computed in place on the
                # freshly-allocated predictions array (sign of the residual is
                # irrelevant once squared): log_liks = -0.5 * ((y - pred)/sigma)^2
                sigma = max(float(obs.noise_std), 1e-9)
                if self.tempering_factor != 1.0:
                    sigma *= np.sqrt(self.tempering_factor)
                predicted -= obs.signal_value
                predicted /= sigma
                np.square(predicted, out=predicted)
                predicted *= -0.5
                log_liks = predicted

        # 2. Numerically stable weight update (prevents complete underflow collapse).
        # Runs in the persistent scratch buffer — no per-step allocations here.
        log_weights = self._scratch_logw
        np.maximum(self._weights, 1e-30, out=log_weights)
        np.log(log_weights, out=log_weights)
        log_weights += log_liks
        log_weights -= log_weights.max()

        raw_weights = np.exp(log_weights, out=log_weights)
        self._step_count += 1
        self._cov_step = -1

        # 3. Normalize weights safely
        weight_sum = np.sum(raw_weights)
        if weight_sum > 1e-30:
            self._weights = (raw_weights / weight_sum).astype(FLOAT_DTYPE, copy=False)
        else:
            self._weights = (np.ones(self.num_particles, dtype=FLOAT_DTYPE) / self.num_particles).astype(FLOAT_DTYPE)

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
        for obs in observations:
            self._append_observation(obs.x, obs.signal_value)

        self.last_obs = observations[-1]
        self.resampled = False

        arrays_in_order = [self._particles[:, j] for j in range(self._d_signal)]
        log_weights = np.zeros(self.num_particles, dtype=FLOAT_DTYPE)

        if getattr(self, "_use_rao_blackwell_noise", False):
            # Batch the model evaluation (the expensive part) into one
            # vectorized matrix call; the Inverse-Gamma posterior recursion
            # over alphas/betas is inherently sequential but cheap.
            all_xs = np.array([obs.x for obs in observations], dtype=FLOAT_DTYPE)
            predictions = self.model.compute_vectorized_many(all_xs, arrays_in_order)
            for k, obs in enumerate(observations):
                predicted = predictions[k]
                sigmas = self._noise_betas / self._noise_alphas
                np.sqrt(sigmas, out=sigmas)
                np.maximum(sigmas, 1e-9, out=sigmas)
                residuals = obs.signal_value - predicted
                log_liks = -0.5 * (residuals / sigmas) ** 2 - 0.5 * np.log(sigmas**2)
                if self.tempering_factor != 1.0:
                    log_liks *= self.tempering_factor
                log_weights += log_liks
                # In-place Inverse-Gamma posterior update (residuals are dead after this)
                self._noise_alphas *= self.noise_discount_factor
                self._noise_alphas += 0.5
                np.square(residuals, out=residuals)
                residuals *= 0.5
                self._noise_betas *= self.noise_discount_factor
                self._noise_betas += residuals

        elif self.noise_model is not None and self._noise_param_slice is not None:
            # Path A: Custom composite noise model evaluating epistemic spread.
            # Model evaluation is batched into one vectorized matrix call; only
            # the per-observation composite likelihood remains in the loop.
            noise_arrays = [
                self._particles[:, j] for j in range(self._noise_param_slice.start, self._noise_param_slice.stop)
            ]
            all_xs = np.array([obs.x for obs in observations], dtype=FLOAT_DTYPE)
            predictions = self.model.compute_vectorized_many(all_xs, arrays_in_order)
            for k, obs in enumerate(observations):
                predicted = predictions[k]
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
                    log_weights += np.log(np.maximum(lik, 1e-30))
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

        # Convert log-weights back to normalized standard weights safely.
        # The log-prior term is computed in the persistent scratch buffer.
        log_prior = self._scratch_logw
        np.maximum(self._weights, 1e-30, out=log_prior)
        np.log(log_prior, out=log_prior)
        log_weights += log_prior
        log_weights -= np.max(log_weights)
        raw_weights = np.exp(log_weights, out=log_weights)
        weight_sum = np.sum(raw_weights)

        # Threshold aligned to 1e-30 to match standard update behavior
        if weight_sum > 1e-30:
            self._weights = (raw_weights / weight_sum).astype(FLOAT_DTYPE, copy=False)
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

    def observation_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, signal_value)`` of all observations as flat float arrays.

        These are views into the internal history buffers — callers must not
        mutate them.
        """
        return self._obs_x_arr[: self._obs_count], self._obs_y_arr[: self._obs_count]

    @property
    def num_observations(self) -> int:
        """Number of observations recorded so far."""
        return self._obs_count

    def _append_observation(self, x: float, y: float) -> None:
        """Record one observation into the flat history buffers (amortized growth)."""
        n = self._obs_count
        if n >= self._obs_x_arr.shape[0]:
            cap = 2 * self._obs_x_arr.shape[0]
            self._obs_x_arr = np.resize(self._obs_x_arr, cap)
            self._obs_y_arr = np.resize(self._obs_y_arr, cap)
        self._obs_x_arr[n] = x
        self._obs_y_arr[n] = y
        self._obs_count = n + 1

    @property
    def _observations(self) -> list[Observation]:
        """Observation history reconstructed on demand (compatibility shim).

        History is stored as flat (x, signal_value) buffers; this getter
        materializes Observation objects for legacy consumers and tests.
        Hot paths must use :meth:`observation_arrays` / :attr:`num_observations`.
        """
        return [
            Observation(x=float(self._obs_x_arr[i]), signal_value=float(self._obs_y_arr[i]))
            for i in range(self._obs_count)
        ]

    @_observations.setter
    def _observations(self, observations: list[Observation]) -> None:
        n = len(observations)
        cap = max(256, n)
        self._obs_x_arr = np.empty(cap, dtype=np.float64)
        self._obs_y_arr = np.empty(cap, dtype=np.float64)
        for i, o in enumerate(observations):
            self._obs_x_arr[i] = o.x
            self._obs_y_arr[i] = o.signal_value
        self._obs_count = n

    def estimated_noise_std(self) -> float:
        """Conservative (90th percentile highest) noise σ estimate.

        Returns the 90th percentile of the noise standard deviation posterior
        distribution across the weighted particle population. Under Rao-Blackwell,
        uses the mode of each particle's Inverse-Gamma posterior.
        """
        if getattr(self, "_use_rao_blackwell_noise", False):
            sigmas = np.sqrt(self._noise_betas / (self._noise_alphas + 0.5))
        elif hasattr(self, "_param_names") and "noise_sigma" in self._param_names:
            idx = self._param_names.index("noise_sigma")
            raw_sigmas = self._particles[:, idx]
            if hasattr(self, "physical_param_bounds") and "noise_sigma" in self.physical_param_bounds:
                lo, hi = self.physical_param_bounds["noise_sigma"]
                sigmas = lo + raw_sigmas * (hi - lo)
            else:
                sigmas = raw_sigmas
        else:
            # Retrieve nominal noise std from the noise model spec itself
            if self.noise_model is not None and hasattr(self.noise_model, "spec") and self.noise_model.spec is not None:
                bounds = getattr(self.noise_model.spec, "bounds", {})
                if "noise_sigma" in bounds:
                    lo, hi = bounds["noise_sigma"]
                    return float(np.sqrt(max(lo * hi, 0.0)))
            raise ValueError(
                "estimated_noise_std() called but no active noise model or noise parameters are configured in the belief."
            )

        # Compute the 90th percentile of the weighted sigmas distribution
        weights = self._weights
        sum_w = np.sum(weights)
        if sum_w > 1e-30:
            norm_weights = weights / sum_w
        else:
            norm_weights = np.ones_like(weights) / len(weights)

        sort_idx = np.argsort(sigmas)
        sorted_sigmas = sigmas[sort_idx]
        sorted_weights = norm_weights[sort_idx]

        cdf = np.cumsum(sorted_weights)
        cdf_vals = np.concatenate((np.array([0.0], dtype=np.float32), cdf))
        sorted_samples = np.concatenate((np.array([sorted_sigmas[0]], dtype=np.float32), sorted_sigmas))

        return float(np.interp(0.90, cdf_vals, sorted_samples))

    def noise_std_uncertainty(self, est_std: float | None = None) -> float:
        """Return the posterior uncertainty (standard deviation) of the noise parameter.

        If Rao-Blackwell is active, calculates the uncertainty of the standard
        deviation sigma using the Delta method:
        Uncertainty(sigma) = Uncertainty(sigma^2) / (2 * estimated_noise_std).
        Otherwise, returns the empirical uncertainty from the particle population
        if noise is a regular particle parameter.
        """
        if getattr(self, "_use_rao_blackwell_noise", False):
            # overall_var_sigma_sq is the variance of variance (sigma^2).
            # From _uncertainty_unit():
            expected_vars = self._noise_betas / np.maximum(self._noise_alphas - 1.0, 1e-9)
            mean_var = np.sum(self._weights * expected_vars)

            denom = (self._noise_alphas - 1.0) ** 2 * np.maximum(self._noise_alphas - 2.0, 1e-9)
            within_var = self._noise_betas**2 / np.maximum(denom, 1e-15)

            overall_var_sigma_sq = np.sum(self._weights * within_var) + np.sum(
                self._weights * (expected_vars - mean_var) ** 2
            )
            var_sigma_sq_std = float(np.sqrt(max(0.0, overall_var_sigma_sq)))

            if est_std is None:
                est_std = self.estimated_noise_std()
            if est_std > 1e-9:
                return var_sigma_sq_std / (2.0 * est_std)
            return var_sigma_sq_std

        if hasattr(self, "_param_names") and "noise_sigma" in self._param_names:
            uncs = self.uncertainty()
            return float(uncs["noise_sigma"])

        raise ValueError(
            "noise_std_uncertainty() called but no active noise model or noise parameters are configured in the belief."
        )

    def _generate_epoch_candidates(self) -> None:
        """Generate a dense slope-targeted grid and cache it for the current epoch.

        Grid targets the steepest slopes (center ± linewidth) of the 3 hyperfine
        dips. Resolution and search windows scale with posterior uncertainty.
        """
        self._dip_centers = []
        # Use unit-space estimates and uncertainties to avoid physical-unit mismatch in subclasses
        estimates = self._estimates_unit()
        uncertainties = self._uncertainty_unit()

        # All targeting arithmetic (f_b - df_hf) must be done in physical space,
        # because the parameters have different bounds/spans and cannot be
        # algebraically combined in unit space.
        phys_bounds = self.physical_param_bounds

        def _to_phys(name: str, unit_val: float) -> float:
            if name not in phys_bounds:
                raise RuntimeError(
                    f"_generate_epoch_candidates: parameter '{name}' is missing from "
                    f"physical_param_bounds — all parameters must be declared at construction."
                )
            lo, hi = phys_bounds[name]
            return lo + unit_val * (hi - lo)

        def _to_phys_delta(name: str, unit_delta: float) -> float:
            if name not in phys_bounds:
                raise RuntimeError(
                    f"_generate_epoch_candidates: parameter '{name}' is missing from "
                    f"physical_param_bounds — all parameters must be declared at construction."
                )
            lo, hi = phys_bounds[name]
            return unit_delta * (hi - lo)

        def _to_unit_freq(phys_freq: np.ndarray) -> np.ndarray:
            if "frequency" not in phys_bounds:
                raise RuntimeError(
                    "_generate_epoch_candidates: 'frequency' is missing from "
                    "physical_param_bounds — all beliefs must declare frequency bounds at construction."
                )
            lo, hi = phys_bounds["frequency"]
            if hi == lo:
                return np.full_like(phys_freq, 0.5)
            return (phys_freq - lo) / (hi - lo)

        f_b_phys = _to_phys("frequency", estimates["frequency"])
        df_hf_phys = _to_phys_delta("split", estimates["split"])

        # 1. Determine linewidth Omega (HWHM) in physical space
        if "linewidth" in estimates:
            omega_phys = _to_phys_delta("linewidth", estimates["linewidth"])
        elif "fwhm_total" in estimates:
            omega_phys = _to_phys_delta("fwhm_total", estimates["fwhm_total"] / 2.0)
        else:
            raise KeyError("Linewidth parameter ('linewidth' or 'fwhm_total') not found in estimates")

        # 2. Extract uncertainties in physical space
        sigma_f_phys = _to_phys_delta("frequency", uncertainties["frequency"])
        if "linewidth" in uncertainties:
            sigma_omega_phys = _to_phys_delta("linewidth", uncertainties["linewidth"])
        elif "fwhm_total" in uncertainties:
            sigma_omega_phys = _to_phys_delta("fwhm_total", uncertainties["fwhm_total"] / 2.0)
        else:
            raise KeyError("Linewidth uncertainty ('linewidth' or 'fwhm_total') not found in uncertainties")

        # 3. Compute effective uncertainty and resolution in physical space
        sigma_eff_phys = np.sqrt(sigma_f_phys**2 + sigma_omega_phys**2)

        # 10 kHz minimum step in physical space
        min_step_physical = 10000.0
        delta_d_phys = max(sigma_eff_phys / 30.0, min_step_physical)
        half_width_phys = 3 * sigma_eff_phys

        # 4. Slope Targeting: 6 locations (2 per dip) in physical space
        centers_phys = [f_b_phys - df_hf_phys, f_b_phys, f_b_phys + df_hf_phys]
        slopes_phys = []
        for c in centers_phys:
            slopes_phys.append(c - omega_phys)
            slopes_phys.append(c + omega_phys)

        # 5. Generate local grids in physical space, then map to unit space
        local_grids = []
        if "frequency" not in phys_bounds:
            raise RuntimeError(
                "_generate_epoch_candidates: 'frequency' is missing from "
                "physical_param_bounds — all beliefs must declare frequency bounds at construction."
            )
        phys_f_lo, phys_f_hi = phys_bounds["frequency"]
        f_lo_unit, f_hi_unit = self.parameter_bounds["frequency"]

        for s_phys in slopes_phys:
            start_phys = max(s_phys - half_width_phys, phys_f_lo)
            stop_phys = min(s_phys + half_width_phys, phys_f_hi)
            if start_phys >= stop_phys:
                continue
            grid_phys = np.arange(start_phys, stop_phys + delta_d_phys, delta_d_phys)
            grid_unit = _to_unit_freq(grid_phys)
            local_grids.append(grid_unit)

        # 5b. Observation-driven dip focusing.
        # Use empirically measured low-signal values to add dense candidates directly
        # at the true dip locations, correcting for posterior bias when belief is wrong.
        if self._obs_count >= 5 and self.noise_model is not None:
            from nvision.sim.locs.bayesian.dip_detection import identify_dip_candidates

            rescale_maps = self._rescale_maps
            if "frequency" not in rescale_maps:
                raise RuntimeError(
                    f"{type(self).__name__} is missing _rescale_maps['frequency']. "
                    "Ensure physical_param_bounds includes 'frequency' at construction."
                )
            freq_rescale = rescale_maps["frequency"]
            obs_xs, obs_ys = self.observation_arrays()
            obs_xs_phys = freq_rescale.to_phys(obs_xs)
            noise_std = self.estimated_noise_std()
            noise_std_unc = self.noise_std_uncertainty(noise_std)
            # Derive cluster radius from the linewidth prior upper bound
            lw_key = "linewidth" if "linewidth" in phys_bounds else "fwhm_total"
            max_linewidth_hz = phys_bounds[lw_key][1]
            max_split_hz = phys_bounds["split"][1] if "split" in phys_bounds else None

            # Extract per-particle noise sigmas
            if getattr(self, "_use_rao_blackwell_noise", False):
                per_particle_sigmas = np.sqrt(self._noise_betas / (self._noise_alphas + 0.5))
            elif "noise_sigma" in self._param_names:
                idx = self._param_names.index("noise_sigma")
                per_particle_sigmas = self._particles[:, idx]
            else:
                per_particle_sigmas = None

            dip_candidates = identify_dip_candidates(
                obs_xs_phys,
                obs_ys,
                noise_std,
                max_linewidth_hz,
                noise_std_unc=noise_std_unc,
                per_particle_sigmas=per_particle_sigmas,
                particle_weights=self._weights,
                max_split_hz=max_split_hz,
            )

            self._dip_centers = [c.centroid_hz for c in dip_candidates]

            # Proportional candidate density
            if dip_candidates:
                total_sig = sum(c.significance for c in dip_candidates)
                total_budget = 100  # total candidate points across all dip grids
                for candidate in dip_candidates:
                    frac = candidate.significance / total_sig if total_sig > 0 else 1.0 / len(dip_candidates)
                    n_pts = max(10, int(frac * total_budget))
                    window_phys = max(3.0 * omega_phys, sigma_eff_phys, 5e6)
                    s = max(candidate.centroid_hz - window_phys, phys_f_lo)
                    e = min(candidate.centroid_hz + window_phys, phys_f_hi)
                    if s >= e:
                        continue
                    dip_grid_phys = np.linspace(s, e, n_pts)
                    local_grids.append(_to_unit_freq(dip_grid_phys))

        # 6. Merge with pre-computed global grid; clip everything to [f_lo, f_hi]
        merged = np.concatenate([*local_grids, self._global_grid]) if local_grids else self._global_grid
        merged = np.clip(merged, f_lo_unit, f_hi_unit).astype(np.float32, copy=False)
        # Each constituent grid is already ascending, so the concatenation is a
        # handful of sorted runs — a stable (timsort) sort is near-linear here,
        # unlike np.unique's full introsort. Dedup with a neighbor mask.
        merged.sort(kind="stable")
        if merged.shape[0] > 1:
            keep = np.empty(merged.shape[0], dtype=bool)
            keep[0] = True
            np.not_equal(merged[1:], merged[:-1], out=keep[1:])
            merged = merged[keep]
        self._current_candidates = merged

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

        # 3. Update particles and reset weights.
        # Gather on the transposed view then transpose back: one copy that
        # preserves the column-major (F-order) particle layout.
        self._particles = self._particles.T[:, new_indices].T
        self._weights = (np.ones(self.num_particles, dtype=FLOAT_DTYPE) / self.num_particles).astype(FLOAT_DTYPE)

        if getattr(self, "_use_rao_blackwell_noise", False):
            self._noise_alphas = self._noise_alphas[new_indices]
            self._noise_betas = self._noise_betas[new_indices]

        # 4. Enforce minimum exploration variance based on parameter ranges.
        # We apply this to the total covariance before nudging, so the steady
        # state variance can properly shrink down to this minimum without exploding.
        # Decay the exploration floor over time so it can converge.
        decay_factor = np.exp(-self._step_count / 25.0)
        curr_min_exploration_frac = self.min_exploration_frac * decay_factor
        for j, name in enumerate(self._param_names):
            lo, hi = self.parameter_bounds[name]
            min_var = ((hi - lo) * curr_min_exploration_frac) ** 2
            cov[j, j] = max(float(cov[j, j]), float(min_var))

        # 5. Compute nudge covariance: (1 - a^2) * Sigma
        nudge_cov = (1 - self.a_param**2) * cov

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
        # MUST happen before nudging so we don't shrink the added noise.
        # In-place so the F-order layout (and dtype) of _particles is preserved.
        self._particles *= self.a_param
        self._particles += (mean * (1 - self.a_param)).astype(FLOAT_DTYPE, copy=False)

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

        # 8.5 Particle Rejuvenation (Rouging): replace some particles with uniform
        # samples from the prior to prevent filter collapse to incorrect local modes.
        # We decay the rejuvenation fraction exponentially to allow high-precision convergence.
        frac_rejuv = 0.05 * decay_factor
        num_random = int(self.num_particles * frac_rejuv)
        if num_random > 0:
            for j, name in enumerate(self._param_names):
                lo, hi = self.parameter_bounds[name]
                if self.priors and name in self.priors:
                    prior_val = self.priors[name]
                    if isinstance(prior_val, tuple) and len(prior_val) >= 2 and prior_val[0] == "sin^2":
                        k = prior_val[1]
                        phys_bounds = getattr(self, "physical_param_bounds", None)
                        if phys_bounds and name in phys_bounds:
                            f_min, f_max = phys_bounds[name]
                        else:
                            f_min, f_max = lo, hi

                        sampled = []
                        while len(sampled) < num_random:
                            candidates = np.random.uniform(f_min, f_max, num_random)
                            probs = np.sin(k * (candidates - f_min)) ** 2
                            u = np.random.uniform(0.0, 1.0, num_random)
                            accepted = candidates[u < probs]
                            sampled.extend(accepted)
                        sampled = np.array(sampled[:num_random])

                        if phys_bounds and name in phys_bounds:
                            self._particles[-num_random:, j] = (sampled - f_min) / (f_max - f_min)
                        else:
                            self._particles[-num_random:, j] = sampled
                    else:
                        mean_prior, std_prior = prior_val
                        random_vals = self._rng.normal(mean_prior, std_prior, num_random)
                        self._particles[-num_random:, j] = np.clip(random_vals, lo, hi)
                else:
                    self._particles[-num_random:, j] = self._rng.uniform(lo, hi, num_random)

        # 9. Update cached candidate grid for the next epoch
        self._generate_epoch_candidates()

    def _marginal_std(self, dim_idx: int) -> float:
        _, var = _weighted_mean_variance_1d(self._particles[:, dim_idx], self._weights)
        return float(np.sqrt(max(0.0, var)))

    def _estimates_unit(self) -> dict[str, float]:
        """Return parameter estimates in internal unit/belief space."""
        means = _weighted_mean_axis0(self._particles, self._weights)
        res = {name: float(means[i]) for i, name in enumerate(self._param_names)}
        if getattr(self, "_use_rao_blackwell_noise", False):
            est_sigmas = np.sqrt(self._noise_betas / self._noise_alphas)
            res["noise_sigma"] = float(np.sum(self._weights * est_sigmas))
        return res

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
            if getattr(self, "_use_rao_blackwell_noise", False):
                stds["noise_sigma"] = 0.0
            return ParameterValues.from_mapping(list(stds.keys()), stds)
        p = self._particles  # (N, d)
        mean = (w @ p) / sw  # (d,)
        diff = p - mean  # (N, d)
        var = (w @ (diff**2)) / sw  # (d,)  weighted variance
        stds = {name: float(np.sqrt(max(0.0, var[i]))) for i, name in enumerate(self._param_names)}

        if getattr(self, "_use_rao_blackwell_noise", False):
            expected_vars = self._noise_betas / np.maximum(self._noise_alphas - 1.0, 1e-9)
            mean_var = np.sum(self._weights * expected_vars)

            denom = (self._noise_alphas - 1.0) ** 2 * np.maximum(self._noise_alphas - 2.0, 1e-9)
            within_var = self._noise_betas**2 / np.maximum(denom, 1e-15)

            overall_var_sigma_sq = np.sum(self._weights * within_var) + np.sum(
                self._weights * (expected_vars - mean_var) ** 2
            )
            stds["noise_sigma"] = float(np.sqrt(max(0.0, overall_var_sigma_sq)))

        return ParameterValues.from_mapping(list(stds.keys()), stds)

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
            noise_discount_factor=self.noise_discount_factor,
            noise_prior_strength=self.noise_prior_strength,
        )
        dist._param_names = self._param_names.copy()
        dist._particles = self._particles.copy(order="K")  # preserve F-order layout
        dist._weights = self._weights.copy()
        dist._step_count = self._step_count
        dist.resampled = self.resampled
        dist._obs_x_arr = self._obs_x_arr.copy()
        dist._obs_y_arr = self._obs_y_arr.copy()
        dist._obs_count = self._obs_count
        dist._use_rao_blackwell_noise = getattr(self, "_use_rao_blackwell_noise", False)
        if getattr(self, "_use_rao_blackwell_noise", False):
            dist._noise_alphas = self._noise_alphas.copy()
            dist._noise_betas = self._noise_betas.copy()
        if hasattr(self, "_dip_centers"):
            dist._dip_centers = list(self._dip_centers)
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

        When the particle count exceeds ``NVISION_SMC_EIG_PARTICLES``, a weighted
        subsample is used for the prediction-variance estimate.  EIG only needs
        to rank candidates — variance estimation converges with ~200–500 particles
        regardless of the total filter size — so subsampling cuts the matrix from
        O(n_candidates × n_particles) to O(n_candidates × n_eig) with negligible
        quality loss.  This is critical at large particle counts (e.g. N=10 000
        produces an 80 MB matrix at 2000 candidates; subsampling to 500 gives 4 MB).
        """
        n_total = self._particles.shape[0]
        n_eig = NVISION_SMC_EIG_PARTICLES

        if n_total > n_eig:
            # Stratified resampling — O(n_eig) vs O(n_total) for np.random.choice.
            # Divides [0,1] into n_eig equal strata; each stratum gets its own
            # independent U(0, 1/n_eig) draw, so there is no fixed grid pattern
            # across steps while coverage of the weight distribution is still
            # guaranteed (one sample per stratum).
            w_norm = self._weights / (self._weights.sum() + 1e-30)
            cdf = np.cumsum(w_norm).astype(np.float32)
            u = np.random.uniform(0.0, 1.0 / n_eig, size=n_eig).astype(np.float32)
            positions = u + np.arange(n_eig, dtype=np.float32) / n_eig
            idx = _systematic_resample_indices(cdf, positions)
            # Gather straight into (d, n_eig) layout: the fancy index on the
            # transposed view produces C-contiguous rows, so the fused kernel
            # gets contiguous per-parameter arrays without a second copy.
            part_t = self._particles.T[:, idx]  # (d, n_eig) — one copy
            w_sub = w_norm[idx].astype(np.float32)
            w_sub /= w_sub.sum()
        else:
            # _particles is F-order, so .T is already C-contiguous: zero-copy.
            part_t = np.ascontiguousarray(self._particles.T)  # (d, n_total)
            w_sub = self._weights  # already float32, normalized

        # Fused kernel: compute weighted prediction variance per candidate in one
        # pass, without materialising the (n_candidates x n_eig) predictions matrix.
        var_pred = self._eig_variance_fused(candidates, part_t, w_sub)

        if getattr(self, "_use_rao_blackwell_noise", False):
            est_variances = self._noise_betas / np.maximum(self._noise_alphas, 1e-9)
            noise_var = float(np.sum(self._weights * est_variances))
            noise_var = max(noise_var, 1e-12)
        else:
            noise_var = max(noise_std**2, 1e-12)

        return 0.5 * np.log1p(var_pred / noise_var)

    def _eig_variance_fused(self, candidates: np.ndarray, part_t: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Dispatch to the model-specific fused EIG-variance kernel.

        ``part_t`` is the particle subset in transposed ``(d, n)`` layout with
        C-contiguous rows, so each per-parameter slice below is a zero-copy
        contiguous view.

        The kernel type is resolved once at construction and cached in
        ``_eig_kernel_type``, so this method has no isinstance checks or
        module imports in the hot path.

        Falls back to the two-step (matrix + variance) path for models that
        do not have a fused kernel.
        """
        out = np.empty(len(candidates), dtype=np.float32)
        xs = np.asarray(candidates, dtype=np.float32)
        w = np.asarray(weights, dtype=np.float32)

        kernel = self._eig_kernel_type

        if kernel == "lorentzian":
            nv_center_lorentzian_eig_variance(
                xs,
                np.ascontiguousarray(part_t[0], dtype=np.float32),  # freq
                np.ascontiguousarray(part_t[1], dtype=np.float32),  # linewidth
                np.ascontiguousarray(part_t[2], dtype=np.float32),  # split
                np.ascontiguousarray(part_t[3], dtype=np.float32),  # k_np
                np.ascontiguousarray(part_t[4], dtype=np.float32),  # c_total
                w,
                out,
            )
            return out

        if kernel == "voigt":
            nv_center_pseudo_voigt_eig_variance(
                xs,
                np.ascontiguousarray(part_t[0], dtype=np.float32),  # freq
                np.ascontiguousarray(part_t[1], dtype=np.float32),  # fwhm_total
                np.ascontiguousarray(part_t[2], dtype=np.float32),  # lorentz_frac
                np.ascontiguousarray(part_t[3], dtype=np.float32),  # split
                np.ascontiguousarray(part_t[4], dtype=np.float32),  # k_np
                np.ascontiguousarray(part_t[5], dtype=np.float32),  # dip_depth
                w,
                out,
            )
            return out

        # Generic fallback: two-step path for any other model type.
        arrays_in_order = [part_t[j] for j in range(self._d_signal)]
        predictions = self.model.compute_vectorized_many_fast(candidates, arrays_in_order)
        return _weighted_variance_rows(predictions, w)

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
    def _rescale_maps(self) -> dict[str, RescaleMap]:
        """Return a ``RescaleMap`` per parameter derived from ``parameter_bounds``.

        For the base ``SMCMarginalDistribution`` particles live in physical
        space, so ``parameter_bounds`` *is* the physical range and each map is
        the identity rescaling for that parameter.

        ``UnitCubeSMCMarginalDistribution`` overrides this to build maps from
        ``physical_param_bounds`` (and ``_original_physical_x_bounds`` for
        frequency) so that the unit-cube↔physical conversion is always correct
        even after the focus window has narrowed.
        """
        if "frequency" not in self.parameter_bounds:
            raise RuntimeError(
                f"{type(self).__name__} is missing 'frequency' in parameter_bounds. "
                "All beliefs must declare physical frequency bounds at construction."
            )
        return {name: RescaleMap(lo=float(lo), hi=float(hi)) for name, (lo, hi) in self.parameter_bounds.items()}

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
