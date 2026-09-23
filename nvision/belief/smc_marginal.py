"""Sequential Monte Carlo (Particle Filter) belief distribution."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv
from numba import njit, prange
from scipy.special import gammaln

from nvision.belief.abstract_marginal import AbstractMarginalDistribution, ParameterValues
from nvision.belief.coordinate import RescaleMap
from nvision.models.observation import Observation
from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.likelihood import likelihood_from_observation_model
from nvision.spectra.noise_model import NoiseSignalModel
from nvision.spectra.numba_kernels import (
    nv_center_lorentzian_eig_variance,
    nv_center_pseudo_voigt_eig_variance,
    nv_center_zeeman_pseudo_voigt_eig_variance,
)

# --- Environment-driven defaults ---------------------------------------------

load_dotenv()

NVISION_SMC_NUM_PARTICLES: int = int(os.getenv("NVISION_SMC_NUM_PARTICLES", "1000"))
NVISION_SMC_ESS_THRESHOLD: float = float(os.getenv("NVISION_SMC_ESS_THRESHOLD", "0.2"))
NVISION_SMC_A_PARAM: float = float(os.getenv("NVISION_SMC_A_PARAM", "0.98"))
NVISION_SMC_MIN_EXPLORATION_FRAC: float = float(os.getenv("NVISION_SMC_MIN_EXPLORATION_FRAC", "0.01"))
NVISION_SMC_TEMPERING_FACTOR: float = float(os.getenv("NVISION_SMC_TEMPERING_FACTOR", "1.0"))

# Maximum particles used for EIG / acquisition scoring.
# EIG only estimates prediction variance — that converges with ~200–500 particles.
# Using all N particles at N=10k produces an 80 MB matrix per EIG call.
# Subsampling keeps the matrix small (< 4 MB) with negligible quality loss.
NVISION_SMC_EIG_PARTICLES: int = int(os.getenv("NVISION_SMC_EIG_PARTICLES", "500"))

# EIG prediction-matrix cache: between resamples the particles and candidate
# grid are frozen, so the prediction surface M[candidate, particle] is invariant
# and only the weights change. When enabled, M (and M^2) are built once per
# epoch and each subsequent step computes EIG as two matrix-vector products
# against the current weights — no model re-evaluation per step.
NVISION_SMC_EIG_CACHE: bool = os.getenv("NVISION_SMC_EIG_CACHE", "1") not in ("0", "false", "False")

# Minimum physical spacing (Hz) for the epoch candidate grid. Controls the
# finest resolution the slope-targeting grid can achieve regardless of sigma.
# Set to 0 (or a very small value) via env to let the grid refine with sigma.
NVISION_SMC_EPOCH_GRID_MIN_STEP_HZ: float = float(os.getenv("NVISION_SMC_EPOCH_GRID_MIN_STEP_HZ", "10000.0"))

# Minimum half-width (Hz) for a dip-focus kernel's window in the epoch candidate
# density mixture (see _generate_epoch_candidates), mirroring the role the old
# per-slope grid half-width played: a detected dip narrower than this still gets
# a window wide enough to be found even if its empirical centroid is slightly off.
NVISION_SMC_DIP_WINDOW_MIN_HZ: float = float(os.getenv("NVISION_SMC_DIP_WINDOW_MIN_HZ", "5000000.0"))

# Single coherent candidate budget for the epoch candidate density mixture,
# replacing the old three independently-tuned counts (a ~12,500-point global
# grid for Lorentzian models, an implicit ~180 points/slope from the sigma/30
# spacing floor, and a 100-point dip-grid budget). Fixed rather than
# env-configurable: unlike the physical constants around it, this is a pure
# computational-budget/quality tradeoff knob, not a physical or algorithmic
# parameter someone would need to retune per experiment.
NVISION_SMC_EPOCH_CANDIDATE_BUDGET: int = 800

_EIG_CHUNK_SIZE: int = 64

# Beyond this many standard deviations, scipy's truncnorm loses precision (both
# CDF endpoints round to the same float), so the far-tail branch of
# _sample_truncated_normal takes over.
_TRUNC_NORM_TAIL_SIGMAS: float = 30.0


def _sample_truncated_normal(
    mean: float,
    std: float,
    lo: float,
    hi: float,
    size: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw ``size`` samples from ``N(mean, std)`` **truncated** to ``[lo, hi]``.

    Prior means are drawn per-repeat and routinely land outside a parameter's
    physical range (e.g. ``zeeman_split``'s prior mean is
    ``gauss(true_split, 6 MHz)`` while the parameter itself is bounded at 0, so a
    true split well under 6 MHz makes a negative prior mean the common case).
    The obvious ``clip(normal(...))`` implementation turns *all* of that
    out-of-range mass into a Dirac atom exactly on the boundary: with a prior
    mean 1.6 sigma below the bound, ~95% of particles start at literally the same
    value. That is particle degeneracy at initialization -- the filter has almost
    no diversity left to reweight, so the posterior stays pinned to the boundary
    with an artificially tight spread and reports a confidently wrong estimate.
    Truncation is what "a Gaussian prior on a bounded parameter" means; clipping
    is a different (and degenerate) distribution.

    ``rng`` defaults to the legacy global ``np.random`` state so seeding via
    ``np.random.seed`` keeps working for callers that rely on it.
    """
    if size <= 0:
        return np.empty(0, dtype=float)
    if not (hi > lo):
        return np.full(size, lo, dtype=float)
    if not (std > 0) or not math.isfinite(std):
        return np.full(size, min(max(mean, lo), hi), dtype=float)

    a = (lo - mean) / std
    b = (hi - mean) / std

    # Far-tail branches: the truncation window sits so deep in one tail that the
    # normal is indistinguishable from an exponential with rate |a|/std (resp.
    # |b|/std) anchored at the near bound. Sampling that directly keeps the
    # spread finite instead of collapsing back onto the boundary.
    if a >= _TRUNC_NORM_TAIL_SIGMAS or b <= -_TRUNC_NORM_TAIL_SIGMAS:
        near, far, rate = (lo, hi, a) if a >= _TRUNC_NORM_TAIL_SIGMAS else (hi, lo, -b)
        scale = std / rate
        draws = (rng or np.random).exponential(scale, size)
        return np.clip(near + math.copysign(1.0, far - near) * draws, min(lo, hi), max(lo, hi))

    from scipy.stats import truncnorm

    # random_state=None makes scipy use numpy's global state, which is what the
    # previous np.random.normal call here used.
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=rng)


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


@njit(cache=True, fastmath=True)
def _chunk_argmax(eig_scores: np.ndarray, chunk_size: int) -> np.ndarray:
    """Find the argmax index within each chunk.

    Serial, not ``parallel=True``: called once per SBED step on realistic
    candidate-grid sizes (hundreds to tens of thousands), where the per-chunk
    work (a ~64-element linear scan) is microseconds — dwarfed by numba's
    thread-pool coordination overhead (measured ~6ms, roughly constant
    regardless of size). Serial is 200-1500x faster across that whole range;
    see the ``_vectorized_many_fast_serial`` variants elsewhere in this module
    for the same size-dependent parallel/serial tradeoff, documented at the
    ~500k-element crossover -- this function never gets remotely close to it.

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
    for c in range(n_chunks):
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


# --- Epoch candidate density mixture (see _generate_epoch_candidates) --------

# Baseline (flat) term's mass share of the local (slope + dip) kernel mass when
# use_global_grid is enabled. A judgment call, not derived: big enough that the
# domain-wide backstop coverage the old global grid provided isn't lost, small
# enough that it never dominates the budget away from the slope/dip kernels
# that carry the actual acquired information.
_EPOCH_BASELINE_MASS_FRACTION: float = 0.2

# Resolution of the scaffold grid used to numerically build the mixture's CDF
# before quantile inversion (see _quantile_place_candidates). This is a pure
# numerical-accuracy knob for that intermediate scaffold -- unrelated to the
# final candidate budget (NVISION_SMC_EPOCH_CANDIDATE_BUDGET) or to the
# resolution floor (NVISION_SMC_EPOCH_GRID_MIN_STEP_HZ) the final candidates
# are snapped to.
_EPOCH_DENSITY_BASELINE_POINTS: int = 2000
_EPOCH_DENSITY_KERNEL_POINTS: int = 400
_EPOCH_DENSITY_KERNEL_SPAN_SIGMAS: float = 6.0


def _quantile_place_candidates(
    kernels: list[tuple[float, float, float]],
    baseline_weight: float,
    lo: float,
    hi: float,
    n_candidates: int,
) -> np.ndarray:
    """Deterministically place ``n_candidates`` points at evenly-spaced quantiles
    of a density mixture over ``[lo, hi]``.

    The mixture is a weighted sum of Gaussian kernels, each given as
    ``(center, bandwidth, weight)``, plus an optional flat baseline term of total
    mass ``baseline_weight``. Placement is exact inverse-CDF quantile sampling
    (deterministic -- evenly-spaced quantile levels, not a random draw), so it
    does not consume or depend on any RNG state and leaves reproducibility under
    a fixed ``NVISION_RNG_SEED`` untouched.

    Implementation note: there is no closed form for a truncated Gaussian-mixture
    CDF inverse, so the density is instead evaluated on an adaptive scaffold grid
    (locally dense around each kernel, coarse elsewhere) and its CDF built by
    trapezoidal integration; quantiles are then read off by linear interpolation
    against that empirical CDF. The scaffold itself is discarded -- it is not the
    returned candidate set.
    """
    if not (hi > lo):
        raise ValueError(f"_quantile_place_candidates: degenerate domain [{lo}, {hi}].")
    if n_candidates <= 0:
        raise ValueError(f"_quantile_place_candidates: n_candidates must be positive, got {n_candidates}.")

    pieces = [np.linspace(lo, hi, _EPOCH_DENSITY_BASELINE_POINTS)]
    for center, bandwidth, weight in kernels:
        if weight <= 0:
            continue
        bw = max(bandwidth, 1e-9)
        span = _EPOCH_DENSITY_KERNEL_SPAN_SIGMAS * bw
        lo_k = max(center - span, lo)
        hi_k = min(center + span, hi)
        if hi_k <= lo_k:
            continue
        pieces.append(np.linspace(lo_k, hi_k, _EPOCH_DENSITY_KERNEL_POINTS))
    xs = np.unique(np.concatenate(pieces))

    density = np.zeros_like(xs)
    if baseline_weight > 0:
        density += baseline_weight / (hi - lo)
    for center, bandwidth, weight in kernels:
        if weight <= 0:
            continue
        bw = max(bandwidth, 1e-9)
        density += weight * np.exp(-0.5 * ((xs - center) / bw) ** 2) / (bw * math.sqrt(2.0 * math.pi))

    seg_area = 0.5 * (density[1:] + density[:-1]) * np.diff(xs)
    cdf = np.concatenate([[0.0], np.cumsum(seg_area)])
    total_mass = cdf[-1]
    if not (total_mass > 0) or not math.isfinite(total_mass):
        raise ValueError(f"_quantile_place_candidates: degenerate density (total mass={total_mass!r}).")
    cdf /= total_mass

    # Midpoint quantile levels avoid q=0/q=1, which would map to exactly lo/hi
    # regardless of where the mixture's actual mass sits.
    q = (np.arange(n_candidates, dtype=np.float64) + 0.5) / n_candidates
    return np.interp(q, cdf, xs)


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
    # Fast path for copy(): skip prior particle sampling and candidate-grid
    # construction in __post_init__ — the caller assigns real state right after.
    # Without this, every snapshot copy pays 10k x d random draws plus a full
    # epoch-grid (and dip-detection) rebuild that is immediately overwritten.
    skip_state_init: bool = field(default=False, repr=False)

    _cached_cov: np.ndarray | None = field(init=False, default=None, repr=False)
    _cov_step: int = field(init=False, default=-1, repr=False)

    # ESS as measured *before* any resample at the current step -- i.e. the value
    # the resample decision was actually made on. This cannot be recovered later
    # from the snapshot weights: _resample() resets them to uniform, so a step
    # that resampled always reads back ESS == num_particles. Diagnostics that
    # want to see the threshold crossing must read this.
    last_ess: float = field(init=False, default=float("nan"), repr=False)

    # Belief-state version: bumped on every mutation of _particles/_weights
    # (update, batch_update, _resample, narrow_scan_parameter_physical_bounds).
    # estimates()/uncertainty() are pure functions of that state and get called
    # many times per step (convergence gates, noise estimation, dip detection,
    # milestones) — memoizing on this counter turns N recomputes/step into 1.
    _belief_version: int = field(init=False, default=0, repr=False)
    _estimates_cache_version: int = field(init=False, default=-1, repr=False)
    _estimates_cache: dict[str, float] | None = field(init=False, default=None, repr=False)
    _uncertainty_cache_version: int = field(init=False, default=-1, repr=False)
    _uncertainty_cache: ParameterValues[float] | None = field(init=False, default=None, repr=False)

    _particles: np.ndarray = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)
    _step_count: int = field(init=False, repr=False, default=0)
    _param_names: list[str] = field(init=False, repr=False)
    _noise_param_slice: slice | None = field(init=False, repr=False, default=None)
    _current_candidates: np.ndarray = field(init=False, repr=False)
    # Whether the epoch candidate density mixture includes the flat domain-wide
    # baseline term (see _generate_epoch_candidates / use_global_grid property).
    _use_global_grid: bool = field(init=False, default=True, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _d_signal: int = field(init=False, repr=False, default=0)
    _eig_kernel_type: str = field(init=False, repr=False, default="generic")
    # Observation history as flat buffers (amortized growth). Only (x, y) are
    # ever consumed from history (dip detection), so full Observation objects
    # are not stored — see the _observations compatibility property.
    _obs_x_arr: np.ndarray | None = field(init=False, repr=False, default=None)
    _obs_y_arr: np.ndarray | None = field(init=False, repr=False, default=None)
    _obs_count: int = field(init=False, repr=False, default=0)
    # Permutation of [0, _obs_count) such that _obs_x_arr[_obs_sort_order[:_obs_count]]
    # is ascending, for indices < _obs_sort_valid_count. Maintained lazily by
    # sorted_observation_arrays() (one searchsorted + in-place shift per pending
    # point) so dip detection can consume an already-sorted observation view
    # instead of re-sorting from scratch each call.
    _obs_sort_order: np.ndarray | None = field(init=False, repr=False, default=None)
    _obs_sort_valid_count: int = field(init=False, repr=False, default=0)
    _scratch_logw: np.ndarray | None = field(init=False, repr=False, default=None)
    # EIG prediction-matrix cache (see NVISION_SMC_EIG_CACHE). _eig_epoch is
    # bumped whenever the candidate grid / particles change so the cache is
    # rebuilt; _eig_cache holds (key, M, M2, sub_idx) for the current epoch.
    _eig_epoch: int = field(init=False, repr=False, default=0)
    _eig_cache: tuple | None = field(init=False, repr=False, default=None)

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
        self._particles = np.zeros(
            (0 if self.skip_state_init else self.num_particles, d_dim), dtype=FLOAT_DTYPE, order="F"
        )

        for i, name in enumerate(self._param_names if not self.skip_state_init else []):
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
                    self._particles[:, i] = _sample_truncated_normal(mean, std, lo, hi, self.num_particles)
            else:
                self._particles[:, i] = np.random.uniform(lo, hi, self.num_particles)

        self._weights = (np.ones(self.num_particles, dtype=FLOAT_DTYPE) / self.num_particles).astype(FLOAT_DTYPE)
        self._step_count = 0
        self._rng = np.random.default_rng()
        self._obs_x_arr = np.empty(256, dtype=np.float64)
        self._obs_y_arr = np.empty(256, dtype=np.float64)
        self._obs_sort_order = np.empty(256, dtype=np.int64)
        self._obs_sort_valid_count = 0
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

        if self.skip_state_init:
            # copy() assigns the real candidates right after construction.
            self._current_candidates = np.array([], dtype=np.float32)
        else:
            # Initialize the first epoch-based candidate grid (see
            # _generate_epoch_candidates for the density-mixture construction).
            self._generate_epoch_candidates()

        # Cache which fused EIG-variance kernel to use — avoids isinstance checks
        # and module imports inside the per-step hot path.
        # Unit-cube beliefs wrap the physical model, so unwrap before dispatching;
        # _eig_variance_fused then converts the (small) particle subset and
        # candidates from unit to physical space instead of falling back to the
        # generic path that materializes the (n_candidates x n_eig) matrix.
        self._eig_needs_phys = False
        self._eig_hf_offset = 0.0
        self._eig_w_center = 1.0
        try:
            from nvision.spectra.nv_center import (
                NVCenterLorentzianModel,
                NVCenterSaturationVoigtModel,
                NVCenterVoigtModel,
            )
            from nvision.spectra.unit_cube import UnitCubeSignalModel

            kernel_model = self.model
            is_unit_cube = isinstance(kernel_model, UnitCubeSignalModel)
            if is_unit_cube:
                kernel_model = kernel_model.inner

            # Hyperfine geometry the fused kernels need: the fixed line offset to
            # substitute when `split` isn't a particle column, and the center-line
            # weight that selects triplet vs. doublet vs. merged-single.
            self._eig_hf_offset = float(getattr(kernel_model, "_hf_offset", 0.0))
            self._eig_w_center = float(getattr(kernel_model, "_w_center", 1.0))

            if isinstance(kernel_model, NVCenterSaturationVoigtModel):
                self._eig_kernel_type = "saturation_voigt"
                self._eig_needs_phys = is_unit_cube
            elif isinstance(kernel_model, NVCenterLorentzianModel):
                # Same issue as the NVCenterVoigtModel branch below:
                # nv_center_lorentzian_eig_variance is a single-dip, 5-column
                # kernel [freq, linewidth, split, k_np, c_total]. A Zeeman-split
                # model has a 6th column (zeeman_split, inserted before split), so
                # every column from "split" onward would be read one slot off and
                # c_total wouldn't be read at all. No fused Zeeman-Lorentzian EIG
                # kernel exists (nv_center_zeeman_lorentzian_eig_variance is
                # referenced only in a docstring elsewhere, never implemented) --
                # route to the generic path, which is correct by construction.
                # It also hardcodes `split`/`k_np` as columns 2/3, which only exist
                # when infer_hyperfine=True -- with the structure unresolved (the
                # default) the model has no such columns and c_total would be read
                # as `split`. Require the exact layout, else use the generic path.
                names = kernel_model.parameter_names()
                has_zeeman = "zeeman_split" in names
                layout_ok = not has_zeeman and "split" in names and "k_np" in names
                self._eig_kernel_type = "lorentzian" if layout_ok else "generic"
                self._eig_needs_phys = is_unit_cube and layout_ok
            elif isinstance(kernel_model, NVCenterVoigtModel):
                # nv_center_pseudo_voigt_eig_variance (the fused kernel below) is a
                # single-dip kernel with a hardcoded 6-column layout [freq,
                # fwhm_total, lorentz_frac, split, k_np, dip_depth]. A Zeeman-split
                # NVCenterVoigtModel has a 7th column (zeeman_split, inserted before
                # split), so every column from "split" onward would be read one slot
                # off and dip_depth wouldn't be read at all -- scoring every
                # acquisition candidate against a wrong-parameter single-dip
                # prediction. There's also no fused Zeeman kernel for this model's
                # dip_depth (physical-depth) amplitude scheme, unlike
                # saturation_voigt's population-normalized c_total (which
                # nv_center_zeeman_pseudo_voigt_eig_variance below actually
                # implements) -- so route to the generic path instead of guessing.
                # That path is correct-by-construction: it calls
                # model.compute_vectorized_many_fast(), which already dispatches on
                # with_zeeman_splitting.
                # Same `split`/`k_np` column requirement as the Lorentzian branch
                # above: those columns only exist when infer_hyperfine=True.
                # nv_center_pseudo_voigt_eig_variance also predates the isotope
                # split and always evaluates a centre line, so it cannot stand in
                # for a ¹⁵N doublet (w_center = 0).
                names = kernel_model.parameter_names()
                has_zeeman = "zeeman_split" in names
                layout_ok = not has_zeeman and "split" in names and "k_np" in names and self._eig_w_center == 1.0
                self._eig_kernel_type = "voigt" if layout_ok else "generic"
                self._eig_needs_phys = is_unit_cube and layout_ok
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
            # Shot-batch sufficient statistics: n_shots (k) and within-batch
            # variance s^2. k == 1 reproduces the single-shot update exactly.
            k = int(getattr(obs, "n_shots", 1) or 1)
            sample_var = getattr(obs, "sample_var", None)

            residuals = obs.signal_value - predicted
            # Rao-Blackwellized marginal likelihood: sigma^2 is analytically
            # *integrated out* of its current per-particle Inverse-Gamma(alpha,
            # beta) posterior rather than plugged in as a point estimate
            # sqrt(beta/alpha). Plugging in the point estimate is a Gaussian
            # likelihood that, for a single residual, is maximized exactly at
            # sigma == |residual| -- so resampling (which selects on this
            # likelihood) systematically favors whichever particle's *current*
            # sigma estimate happens to match that step's single noise draw.
            # Since median(|N(0, sigma)|) ~= 0.6745*sigma, an under-estimating
            # particle wins more often than not, and nothing thereafter
            # perturbs the noise state back apart (_resample() reindexes
            # _noise_alphas/_noise_betas but never nudges them) -- the
            # population's sigma estimate drifts to the prior's lower bound
            # over successive resamples. Integrating over the Inverse-Gamma
            # instead gives the exact Normal-InverseGamma predictive, a
            # (shifted, scaled) Student-t with nu = 2*alpha d.o.f. and
            # scale^2 = beta / (k*alpha) for the k-shot batch mean; its
            # fatter tails don't reward an under-confident sigma for
            # coincidentally matching one residual.
            alpha = self._noise_alphas
            beta = self._noise_betas
            z_sq = k * residuals**2 / beta
            log_liks = (
                gammaln(alpha + 0.5)
                - gammaln(alpha)
                - 0.5 * np.log(2.0 * np.pi * beta / (k * alpha))
                - (alpha + 0.5) * np.log1p(z_sq / (2.0 * alpha))
            )
            if self.tempering_factor != 1.0:
                log_liks *= self.tempering_factor
            # In-place Inverse-Gamma posterior update (residuals are dead after this).
            # Two orthogonal pieces of evidence about sigma:
            #   between-batch: the fit residual, rescaled by k since the mean's
            #                  variance is sigma^2/k, so k*res^2 estimates sigma^2;
            #   within-batch:  the empirical sample variance s^2 with (k-1) dof.
            self._noise_alphas *= self.noise_discount_factor
            self._noise_betas *= self.noise_discount_factor
            np.square(residuals, out=residuals)
            residuals *= 0.5 * k
            self._noise_alphas += 0.5
            self._noise_betas += residuals
            if k >= 2 and sample_var is not None:
                self._noise_alphas += 0.5 * (k - 1)
                self._noise_betas += 0.5 * (k - 1) * float(sample_var)

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
        self._belief_version += 1

        # 4. Resample if Effective Sample Size (ESS) is too low [cite: 198, 199]
        ess = _inverse_sum_squares(self._weights)
        self.last_ess = float(ess)
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
            for obs_idx, obs in enumerate(observations):
                predicted = predictions[obs_idx]
                residuals = obs.signal_value - predicted
                # Rao-Blackwellized marginal likelihood (see the matching branch
                # in update() for why the point-estimate plug-in sigma biases
                # the posterior toward zero): integrate sigma^2 out of its
                # current Inverse-Gamma(alpha, beta) posterior instead of
                # substituting sqrt(beta/alpha), giving the Normal-InverseGamma
                # Student-t predictive with nu = 2*alpha and scale^2 = beta/alpha.
                alpha = self._noise_alphas
                beta = self._noise_betas
                z_sq = residuals**2 / beta
                log_liks = (
                    gammaln(alpha + 0.5)
                    - gammaln(alpha)
                    - 0.5 * np.log(2.0 * np.pi * beta / alpha)
                    - (alpha + 0.5) * np.log1p(z_sq / (2.0 * alpha))
                )
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
        self._belief_version += 1

        # Evaluate Effective Sample Size (ESS) for resampling
        ess = _inverse_sum_squares(self._weights)
        self.last_ess = float(ess)
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

    def sorted_observation_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, signal_value)`` of all observations sorted ascending by x.

        Maintains ``_obs_sort_order`` lazily: indices in
        ``[_obs_sort_valid_count, _obs_count)`` haven't been incorporated into
        the sorted permutation yet, so each is inserted in turn via a single
        ``searchsorted`` (O(log n)) plus an in-place slice shift (O(n),
        memmove-safe under overlapping numpy basic-index assignment). A call
        with no new observations since the last one is O(1). This avoids
        paying O(n log n) to re-sort the full history from scratch on every
        call (e.g. once per SMC resample epoch, for dip detection).

        Resolving lazily (reading each index's *current* ``_obs_x_arr`` value
        at insertion time, not at append time) is required for correctness:
        ``UnitCubeSMCMarginalDistribution`` writes a provisional narrowed-frame
        x into ``_obs_x_arr`` via ``_append_observation`` and then overwrites it
        with the original-frame value afterwards (see its ``update``/
        ``batch_update``) — see ``_resync_sort_position`` for the case where
        that overwrite happens *after* this method already consumed the index.

        Returns freshly-gathered arrays (not views), safe for callers to hold
        onto even as more observations are appended.
        """
        n = self._obs_count
        order = self._obs_sort_order
        for i in range(self._obs_sort_valid_count, n):
            sorted_xs_so_far = self._obs_x_arr[order[:i]]
            pos = int(np.searchsorted(sorted_xs_so_far, self._obs_x_arr[i], side="right"))
            order[pos + 1 : i + 1] = order[pos:i]
            order[pos] = i
        self._obs_sort_valid_count = n
        order = order[:n]
        return self._obs_x_arr[order], self._obs_y_arr[order]

    def _resync_sort_position(self, idx: int) -> None:
        """Fix up ``_obs_sort_order`` after ``_obs_x_arr[idx]`` was mutated directly.

        Only ``UnitCubeSMCMarginalDistribution`` needs this: it overwrites the
        narrowed-frame x that ``_append_observation`` stored for the newest
        observation(s) with the original-frame value, after ``super().update()``/
        ``super().batch_update()`` returns. If ``sorted_observation_arrays()``
        never ran in between, ``idx`` hasn't been incorporated into the sort
        order yet and the next call will naturally pick up the corrected value
        (no-op here). If it *did* run — dip detection can trigger mid-``update()``
        via resampling, before this correction — ``idx`` is already placed in
        ``_obs_sort_order`` at a position based on the stale value; this removes
        it and re-inserts it using the corrected one, preserving sortedness for
        ``[0, _obs_sort_valid_count)`` without needing a full re-sort.
        """
        valid = self._obs_sort_valid_count
        if idx >= valid:
            return
        order = self._obs_sort_order
        cur_pos = int(np.where(order[:valid] == idx)[0][0])
        order[cur_pos : valid - 1] = order[cur_pos + 1 : valid]
        sorted_xs = self._obs_x_arr[order[: valid - 1]]
        new_pos = int(np.searchsorted(sorted_xs, self._obs_x_arr[idx], side="right"))
        order[new_pos + 1 : valid] = order[new_pos : valid - 1]
        order[new_pos] = idx

    @property
    def num_observations(self) -> int:
        """Number of observations recorded so far."""
        return self._obs_count

    def _append_observation(self, x: float, y: float) -> None:
        """Record one observation into the flat history buffers (amortized growth).

        Does not touch ``_obs_sort_order`` — that's maintained lazily by
        ``sorted_observation_arrays()`` (see its docstring for why: subclasses
        may still mutate ``_obs_x_arr[n]`` after this call returns).
        """
        n = self._obs_count
        if n >= self._obs_x_arr.shape[0]:
            cap = 2 * self._obs_x_arr.shape[0]
            self._obs_x_arr = np.resize(self._obs_x_arr, cap)
            self._obs_y_arr = np.resize(self._obs_y_arr, cap)
            self._obs_sort_order = np.resize(self._obs_sort_order, cap)
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
        self._obs_sort_order = np.empty(cap, dtype=np.int64)
        for i, o in enumerate(observations):
            self._obs_x_arr[i] = o.x
            self._obs_y_arr[i] = o.signal_value
        self._obs_count = n
        # Bulk-set path (not the hot per-observation append) — a single sort is fine.
        self._obs_sort_order[:n] = np.argsort(self._obs_x_arr[:n])
        self._obs_sort_valid_count = n

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
                "estimated_noise_std() called but no active noise model or noise "
                "parameters are configured in the belief."
            )

        # Compute the 90th percentile of the weighted sigmas distribution
        weights = self._weights
        sum_w = np.sum(weights)
        norm_weights = weights / sum_w if sum_w > 1e-30 else np.ones_like(weights) / len(weights)

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

    @property
    def use_global_grid(self) -> bool:
        """Whether the epoch candidate density mixture includes the flat,
        domain-wide baseline term (see :meth:`_generate_epoch_candidates`).

        Defaults to ``True``. The SBED locator sets this ``False`` once its
        focus-window confidence signal (``compute_focus_window_confidence(...)
        .is_stable``) fires, on the reasoning that once the true dip location is
        confidently found, spending candidate budget on domain-wide backstop
        coverage is no longer worth it.
        """
        return self._use_global_grid

    @use_global_grid.setter
    def use_global_grid(self, value: bool) -> None:
        value = bool(value)
        if value == self._use_global_grid:
            return
        self._use_global_grid = value
        self._generate_epoch_candidates()

    def _generate_epoch_candidates(self) -> None:
        """Generate the epoch candidate density mixture and cache its quantile
        placement for the current epoch.

        Targets the steepest slopes (center ± linewidth) of the hyperfine dips
        and (once enough observations exist) empirically-detected dip centroids.
        Kernel bandwidths and window sizes scale with posterior uncertainty. See
        the "5. Build the unified candidate-density mixture" comment below for
        the full construction.
        """
        # New epoch: candidates and particles have changed, so the EIG
        # prediction-matrix cache (built from both) must be rebuilt.
        self._eig_epoch += 1
        self._eig_cache = None
        self._dip_centers = []
        # Use unit-space estimates and uncertainties to avoid physical-unit mismatch in subclasses
        estimates = self._estimates_unit()
        uncertainties = self._uncertainty_unit()

        # All targeting arithmetic (f_b - df_hf) must be done in physical space,
        # because the parameters have different bounds/spans and cannot be
        # algebraically combined in unit space.
        phys_bounds = self.physical_param_bounds

        # "frequency" may be fixed (not a particle dimension, e.g.
        # NVCenterVoigtModel(with_fixed_frequency=True)) and therefore absent from
        # estimates/uncertainties above. Slope-targeting still needs a center and
        # a (zero) uncertainty for it, so pull the fixed value from the model spec
        # and synthesize both entries -- exact center, no spread.
        if "frequency" not in estimates and "frequency" in phys_bounds:
            inner = getattr(self.model, "inner", None)
            fixed_vals = getattr(getattr(inner, "spec", None), "fixed_values", None) or {}
            if "frequency" not in fixed_vals:
                raise RuntimeError(
                    "_generate_epoch_candidates: 'frequency' is not a particle dimension and "
                    "not in the model spec's fixed_values -- cannot determine its value."
                )
            lo, hi = phys_bounds["frequency"]
            estimates = {**estimates, "frequency": (fixed_vals["frequency"] - lo) / (hi - lo) if hi != lo else 0.5}
            uncertainties = {**uncertainties, "frequency": 0.0}

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
        df_hf_phys = _to_phys_delta("split", estimates["split"]) if "split" in estimates else 0.0
        df_zeeman_phys = (
            _to_phys_delta("zeeman_split", estimates["zeeman_split"]) if "zeeman_split" in estimates else 0.0
        )

        # 1. Determine linewidth Omega (HWHM) in physical space
        is_saturation_voigt = "saturation" in estimates and "sigma_inhom" in estimates
        if is_saturation_voigt:
            from nvision.spectra.nv_center import saturation_voigt_effective_hwhm_and_unc

            saturation_phys = _to_phys("saturation", estimates["saturation"])
            sigma_inhom_phys = _to_phys("sigma_inhom", estimates["sigma_inhom"])
            omega_phys, _ = saturation_voigt_effective_hwhm_and_unc(saturation_phys, sigma_inhom_phys)
        elif "linewidth" in estimates:
            omega_phys = _to_phys_delta("linewidth", estimates["linewidth"])
        elif "homogeneous_linewidth" in estimates:
            omega_phys = _to_phys_delta("homogeneous_linewidth", estimates["homogeneous_linewidth"])
        elif "fwhm_total" in estimates:
            omega_phys = _to_phys_delta("fwhm_total", estimates["fwhm_total"] / 2.0)
        else:
            raise KeyError(
                "Linewidth parameter ('linewidth', 'homogeneous_linewidth', or 'fwhm_total') not found in estimates"
            )

        # 2. Extract uncertainties in physical space
        sigma_f_phys = _to_phys_delta("frequency", uncertainties["frequency"])
        if is_saturation_voigt:
            sigma_s_phys = _to_phys_delta("saturation", uncertainties["saturation"])
            sigma_sigma_inhom_phys = _to_phys_delta("sigma_inhom", uncertainties["sigma_inhom"])
            _, sigma_omega_phys = saturation_voigt_effective_hwhm_and_unc(
                saturation_phys, sigma_inhom_phys, sigma_s_phys, sigma_sigma_inhom_phys
            )
        elif "linewidth" in uncertainties:
            sigma_omega_phys = _to_phys_delta("linewidth", uncertainties["linewidth"])
        elif "homogeneous_linewidth" in uncertainties:
            sigma_omega_phys = _to_phys_delta("homogeneous_linewidth", uncertainties["homogeneous_linewidth"])
        elif "fwhm_total" in uncertainties:
            sigma_omega_phys = _to_phys_delta("fwhm_total", uncertainties["fwhm_total"] / 2.0)
        else:
            raise KeyError(
                "Linewidth uncertainty ('linewidth', 'homogeneous_linewidth', or "
                "'fwhm_total') not found in uncertainties"
            )

        # 3. Compute effective uncertainty in physical space -- the bandwidth of
        # each slope-targeting Gaussian kernel below.
        sigma_eff_phys = np.sqrt(sigma_f_phys**2 + sigma_omega_phys**2)
        min_step_physical = NVISION_SMC_EPOCH_GRID_MIN_STEP_HZ

        # 4. Slope Targeting: centers for all Zeeman groups × HF sub-peaks (deduplicated)
        zeeman_offsets = [-df_zeeman_phys, df_zeeman_phys] if df_zeeman_phys > 0 else [0.0]
        hf_offsets = [-df_hf_phys, 0.0, df_hf_phys] if df_hf_phys > 0 else [0.0]
        centers_seen: set[int] = set()
        centers_phys = []
        for zo in zeeman_offsets:
            for hfo in hf_offsets:
                c = f_b_phys + zo + hfo
                c_key = round(c)
                if c_key not in centers_seen:
                    centers_seen.add(c_key)
                    centers_phys.append(c)
        slopes_phys = []
        for c in centers_phys:
            slopes_phys.append(c - omega_phys)
            slopes_phys.append(c + omega_phys)

        # 5. Build the unified candidate-density mixture.
        #
        # Replaces the old three independently-sized grids (global coarse grid +
        # per-slope uniform windows + a separately-budgeted dip grid) with one
        # continuous density over [f_lo, f_hi]: a Gaussian kernel per slope point
        # (bandwidth = sigma_eff_phys, matching the old uniform slope windows'
        # 3-sigma half-width), a Gaussian kernel per empirically-detected dip
        # centroid (bandwidth = its detection window / 3, same span-matching
        # logic), and an optional flat baseline term for domain-wide backstop
        # coverage. NVISION_SMC_EPOCH_CANDIDATE_BUDGET candidates are then placed
        # at that density's evenly-spaced CDF quantiles -- deterministic (no RNG
        # draw, so NVISION_RNG_SEED reproducibility is untouched), and
        # concentrated where the mixture says information actually is, rather
        # than uniform-within-window-then-a-density-cliff at the window edge.
        if "frequency" not in phys_bounds:
            raise RuntimeError(
                "_generate_epoch_candidates: 'frequency' is missing from "
                "physical_param_bounds — all beliefs must declare frequency bounds at construction."
            )
        phys_f_lo, phys_f_hi = phys_bounds["frequency"]
        f_lo_unit, f_hi_unit = self.parameter_bounds["frequency"]
        if not (phys_f_hi > phys_f_lo):
            raise ValueError(f"_generate_epoch_candidates: degenerate frequency domain [{phys_f_lo}, {phys_f_hi}].")

        slope_bw_phys = max(sigma_eff_phys, min_step_physical)
        kernels: list[tuple[float, float, float]] = [(s_phys, slope_bw_phys, 1.0) for s_phys in slopes_phys]
        n_slope_kernels = len(kernels)
        if n_slope_kernels == 0:
            raise RuntimeError(
                "_generate_epoch_candidates: no slope-targeting centers were generated -- "
                "slopes_phys must always contain at least one dip's ±omega points."
            )

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
            # Sorted-by-x view (maintained incrementally in _append_observation,
            # not re-sorted here). to_phys() is a strictly increasing affine map
            # (RescaleMap enforces hi > lo), so sortedness is preserved.
            obs_xs, obs_ys = self.sorted_observation_arrays()
            obs_xs_phys = freq_rescale.to_phys(obs_xs)
            noise_std = self.estimated_noise_std()
            noise_std_unc = self.noise_std_uncertainty(noise_std)
            # Derive cluster radius from the linewidth prior upper bound (lineshape-agnostic:
            # mirrors sbed_locator._effective_max_linewidth_hz).
            if "linewidth" in phys_bounds:
                max_linewidth_hz = phys_bounds["linewidth"][1]
            elif "saturation" in phys_bounds and "sigma_inhom" in phys_bounds:
                from nvision.spectra.nv_center import NV_NATURAL_HWHM_HZ

                s_hi = phys_bounds["saturation"][1]
                sigma_inhom_hi = phys_bounds["sigma_inhom"][1]
                gamma_hom_hi = NV_NATURAL_HWHM_HZ * math.sqrt(1.0 + s_hi)
                max_linewidth_hz = gamma_hom_hi + math.sqrt(2.0 * math.log(2.0)) * sigma_inhom_hi
            elif "homogeneous_linewidth" in phys_bounds:
                hl_hi = phys_bounds["homogeneous_linewidth"][1]
                sigma_inhom_hi = phys_bounds.get("sigma_inhom", (0.0, 0.0))[1]
                max_linewidth_hz = hl_hi + math.sqrt(2.0 * math.log(2.0)) * sigma_inhom_hi
            elif "fwhm_total" in phys_bounds:
                max_linewidth_hz = phys_bounds["fwhm_total"][1] / 2.0
            else:
                max_linewidth_hz = 1e6
            max_zeeman_hz = phys_bounds["zeeman_split"][1] if "zeeman_split" in phys_bounds else 0.0
            max_hf_split_hz = phys_bounds["split"][1] if "split" in phys_bounds else 0.0
            if max_zeeman_hz > 0 or max_hf_split_hz > 0:
                max_split_hz: float | None = 2.0 * max_zeeman_hz + 2.0 * max_hf_split_hz
            else:
                max_split_hz = None

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
                assume_sorted=True,
            )

            self._dip_centers = [c.centroid_hz for c in dip_candidates]

            if dip_candidates:
                total_sig = sum(c.significance for c in dip_candidates)
                # The dip family competes with the slope family on equal aggregate
                # footing (each gets n_slope_kernels total mixture weight); within
                # the dip family, individual dips are still split by significance,
                # mirroring the old total_budget=100 proportional split.
                dip_family_weight = float(n_slope_kernels)
                for candidate in dip_candidates:
                    frac = candidate.significance / total_sig if total_sig > 0 else 1.0 / len(dip_candidates)
                    window_phys = max(3.0 * omega_phys, sigma_eff_phys, NVISION_SMC_DIP_WINDOW_MIN_HZ)
                    dip_bw_phys = max(window_phys / 3.0, min_step_physical)
                    kernels.append((candidate.centroid_hz, dip_bw_phys, frac * dip_family_weight))

        # 6. Optional flat baseline term for domain-wide backstop coverage, gated
        # by use_global_grid. Scaled relative to the local (slope + dip) mass so
        # its share stays roughly constant whether or not dips have been detected.
        local_weight_total = sum(w for _, _, w in kernels)
        baseline_weight = _EPOCH_BASELINE_MASS_FRACTION * local_weight_total if self.use_global_grid else 0.0

        total_weight = local_weight_total + baseline_weight
        if not (total_weight > 0) or not math.isfinite(total_weight):
            raise ValueError(
                "_generate_epoch_candidates: candidate density mixture has zero total weight "
                f"(use_global_grid={self.use_global_grid}, n_kernels={len(kernels)}, "
                f"local_weight_total={local_weight_total!r}). There is nothing to build an "
                "epoch candidate grid from."
            )

        candidates_phys = _quantile_place_candidates(
            kernels=kernels,
            baseline_weight=baseline_weight,
            lo=phys_f_lo,
            hi=phys_f_hi,
            n_candidates=NVISION_SMC_EPOCH_CANDIDATE_BUDGET,
        )

        # Snap to the resolution floor (skipped if the floor is disabled via env,
        # matching NVISION_SMC_EPOCH_GRID_MIN_STEP_HZ's documented "0 to let the
        # grid refine with sigma" escape hatch), clip into bounds, dedup.
        if min_step_physical > 0:
            candidates_phys = np.round(candidates_phys / min_step_physical) * min_step_physical
        candidates_phys = np.clip(candidates_phys, phys_f_lo, phys_f_hi)
        merged = np.clip(_to_unit_freq(candidates_phys), f_lo_unit, f_hi_unit).astype(np.float32, copy=False)
        # Quantile placement returns an ascending sequence, so this is already
        # (near-)sorted -- a stable (timsort) sort is near-linear here, unlike
        # np.unique's full introsort. Dedup with a neighbor mask.
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
        # Capture the triggering ESS before step 3 resets the weights to uniform.
        self.last_ess = float(_inverse_sum_squares(self._weights))
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
                        self._particles[-num_random:, j] = _sample_truncated_normal(
                            mean_prior, std_prior, lo, hi, num_random, rng=self._rng
                        )
                else:
                    self._particles[-num_random:, j] = self._rng.uniform(lo, hi, num_random)

        # 8.6 Joint dip-informed rejuvenation for (frequency, zeeman_split).
        # The per-dimension loop above samples each parameter independently, so
        # the chance a rejuvenated particle lands near the *correct joint*
        # (frequency, zeeman_split) combination is negligible once other
        # dimensions have converged -- this is how the filter mode-locks onto a
        # mirror-symmetric wrong solution (mistaking one real Zeeman dip for the
        # other, frequency off by ~1 zeeman_split) and never recovers: no
        # particle exists near the true mode to be reweighted back up. When
        # both params are free and at least two dip candidates were detected in
        # observed data (self._dip_centers, most-significant first, from the
        # prior epoch's dip detection), overwrite a slice of the rejuvenated
        # particles with the joint (frequency, zeeman_split) implied directly
        # by the two most significant detected dips -- the same "midpoint /
        # half-separation of the two dips" physics used by
        # GenericSweepLocator's fit seeding (see generic_sweep_locator.py) --
        # so the correct mode gets a chance to be rediscovered and reweighted
        # by the next likelihood update. The other dimensions of these
        # particles are set to the pre-resample weighted mean (not left at the
        # independent-uniform draw above) so they represent the current
        # best-fit shape at the alternate frequency/zeeman position, a
        # meaningfully competing hypothesis rather than pure noise.
        if (
            num_random > 0
            and "frequency" in self._param_names
            and "zeeman_split" in self._param_names
            and len(self._dip_centers) >= 2
        ):
            freq_idx = self._param_names.index("frequency")
            zeeman_idx = self._param_names.index("zeeman_split")
            d0, d1 = sorted(self._dip_centers[:2])
            freq_candidate_phys = 0.5 * (d0 + d1)
            zeeman_candidate_phys = 0.5 * (d1 - d0)
            rescale = self._rescale_maps
            freq_candidate = rescale["frequency"].to_unit(freq_candidate_phys)
            zeeman_candidate = rescale["zeeman_split"].to_unit(zeeman_candidate_phys)
            f_lo, f_hi = self.parameter_bounds["frequency"]
            z_lo, z_hi = self.parameter_bounds["zeeman_split"]

            self._particles[-num_random:, :] = mean
            jitter_f = self._rng.normal(0.0, 0.01 * (f_hi - f_lo), num_random)
            jitter_z = self._rng.normal(0.0, 0.01 * (z_hi - z_lo), num_random)
            self._particles[-num_random:, freq_idx] = np.clip(freq_candidate + jitter_f, f_lo, f_hi)
            self._particles[-num_random:, zeeman_idx] = np.clip(zeeman_candidate + jitter_z, z_lo, z_hi)

        # 9. Update cached candidate grid for the next epoch
        self._generate_epoch_candidates()
        self._belief_version += 1

    def _marginal_std(self, dim_idx: int) -> float:
        _, var = _weighted_mean_variance_1d(self._particles[:, dim_idx], self._weights)
        return float(np.sqrt(max(0.0, var)))

    def _estimates_unit(self) -> dict[str, float]:
        """Return parameter estimates in internal unit/belief space.

        Memoized on ``_belief_version`` — pure function of (particles, weights),
        which only change in update/batch_update/_resample/narrow_scan_parameter_
        physical_bounds. Callers get their own fresh dict each time (built from the
        cached values) so mutating the returned dict can never corrupt the cache.
        """
        if self._estimates_cache_version == self._belief_version and self._estimates_cache is not None:
            return dict(self._estimates_cache)

        means = _weighted_mean_axis0(self._particles, self._weights)
        res = {name: float(means[i]) for i, name in enumerate(self._param_names)}
        if getattr(self, "_use_rao_blackwell_noise", False):
            est_sigmas = np.sqrt(self._noise_betas / self._noise_alphas)
            res["noise_sigma"] = float(np.sum(self._weights * est_sigmas))

        self._estimates_cache = res
        self._estimates_cache_version = self._belief_version
        return dict(res)

    def estimates(self) -> dict[str, float]:
        """Return parameter estimates (weighted mean)."""
        return self._estimates_unit()

    def mode_estimates(self) -> dict[str, float]:
        """Return the highest-weight particle's joint state (posterior mode).

        A weighted mean over multimodal or correlated particles is not itself
        a state the posterior supports; the highest-weight particle is one
        that is.
        """
        idx = int(np.argmax(self._weights))
        res = {name: float(self._particles[idx, i]) for i, name in enumerate(self._param_names)}
        if getattr(self, "_use_rao_blackwell_noise", False):
            res["noise_sigma"] = float(np.sqrt(self._noise_betas[idx] / self._noise_alphas[idx]))
        return res

    def _uncertainty_unit(self) -> ParameterValues[float]:
        """Return parameter uncertainties (std dev) in internal unit/belief space.

        Computes all marginal variances in a single vectorized pass instead of
        calling the 1-D Numba kernel once per dimension.

        Memoized on ``_belief_version`` (see ``_estimates_unit``). Safe to return
        the cached instance directly — ``ParameterValues`` is a frozen dataclass.
        """
        if self._uncertainty_cache_version == self._belief_version and self._uncertainty_cache is not None:
            return self._uncertainty_cache

        w = self._weights
        sw = w.sum()
        if sw <= 0.0:
            stds = {name: 0.0 for name in self._param_names}
            if getattr(self, "_use_rao_blackwell_noise", False):
                stds["noise_sigma"] = 0.0
            result = ParameterValues.from_mapping(list(stds.keys()), stds)
            self._uncertainty_cache = result
            self._uncertainty_cache_version = self._belief_version
            return result
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

        result = ParameterValues.from_mapping(list(stds.keys()), stds)
        self._uncertainty_cache = result
        self._uncertainty_cache_version = self._belief_version
        return result

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        return self._uncertainty_unit()

    def _robust_uncertainty_unit(self) -> ParameterValues[float]:
        """Outlier-insensitive marginal spread (weighted IQR / 1.349) in unit space.

        Every resample injects a decaying fraction of particles redrawn from the
        prior (see ``_resample``). They vanish by the next likelihood update, but
        standard deviation is quadratic in distance, so even a couple of them
        sitting far from the bulk of the cloud inflate ``_uncertainty_unit`` several-
        fold for exactly one step -- a rendering/consumer-visible sawtooth that
        reads as the belief repeatedly widening and re-narrowing when nothing of
        the sort happened. The interquartile range only depends on the *bulk* of
        the (weighted) particle mass, so those transient particles don't move it;
        dividing by 1.349 rescales it to a Gaussian-equivalent sigma so it's
        directly comparable to ``_uncertainty_unit``'s output.

        Deliberately NOT used as the default ``uncertainty()`` / ``_empirical_
        uncertainty()``: IQR only reflects the middle 50% of the mass, so a
        genuinely bimodal posterior (e.g. two still-live candidate frequencies)
        can report a falsely tight spread if both modes' particles fall outside
        that middle 50% -- masking real ambiguity the raw std correctly flags.
        This exists for callers that specifically want the resample artifact
        filtered out of a streak/consecutive-checks decision (see
        SequentialBayesianLocator._check_and_resample's convergence-streak use),
        not as a general-purpose replacement for the reported uncertainty.
        """
        w = self._weights
        sw = w.sum()
        n_params = len(self._param_names)
        if sw <= 0.0 or self.num_particles < 4:
            stds = dict.fromkeys(self._param_names, float("nan"))
            if getattr(self, "_use_rao_blackwell_noise", False):
                stds["noise_sigma"] = float("nan")
            return ParameterValues.from_mapping(list(stds.keys()), stds)

        p = self._particles  # (N, d)
        w_norm = w / sw
        stds: dict[str, float] = {}
        for i in range(n_params):
            values = p[:, i]
            order = np.argsort(values)
            v = values[order]
            wo = w_norm[order]
            # Midpoint CDF: the quantile of a particle is the mass strictly below
            # it plus half its own -- keeps the estimate unbiased for small clouds
            # (matches nvision.runner.plots_data._weighted_robust_sigma).
            cw = np.cumsum(wo) - wo * 0.5
            q1, q3 = np.interp([0.25, 0.75], cw, v)
            stds[self._param_names[i]] = float((q3 - q1) / 1.349)

        if getattr(self, "_use_rao_blackwell_noise", False):
            # noise_sigma has no particle dimension of its own (it's a Rao-
            # Blackwellized per-particle Normal-Inverse-Gamma posterior, not a
            # sampled coordinate) -- no IQR is defined for it; fall back to the
            # same value _uncertainty_unit reports.
            stds["noise_sigma"] = float(self._uncertainty_unit().get("noise_sigma", float("nan")))

        return ParameterValues.from_mapping(list(stds.keys()), stds)

    def _empirical_robust_uncertainty(self) -> ParameterValues[float]:
        return self._robust_uncertainty_unit()

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
            skip_state_init=True,
        )
        # Candidates depend only on bounds/particles, which are identical here —
        # share by reference (consumers rebind on narrowing/resample, never
        # mutate in place).
        dist._use_global_grid = self._use_global_grid
        dist._current_candidates = self._current_candidates
        dist._param_names = self._param_names.copy()
        dist._particles = self._particles.copy(order="K")  # preserve F-order layout
        dist._weights = self._weights.copy()
        dist._step_count = self._step_count
        dist.resampled = self.resampled
        dist.last_ess = self.last_ess
        dist._obs_x_arr = self._obs_x_arr.copy()
        dist._obs_y_arr = self._obs_y_arr.copy()
        dist._obs_sort_order = self._obs_sort_order.copy()
        dist._obs_sort_valid_count = self._obs_sort_valid_count
        dist._obs_count = self._obs_count
        dist._use_rao_blackwell_noise = getattr(self, "_use_rao_blackwell_noise", False)
        if getattr(self, "_use_rao_blackwell_noise", False):
            dist._noise_alphas = self._noise_alphas.copy()
            dist._noise_betas = self._noise_betas.copy()
        if hasattr(self, "_dip_centers"):
            dist._dip_centers = list(self._dip_centers)
        dist._use_global_grid = self._use_global_grid
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

        if NVISION_SMC_EIG_CACHE:
            var_pred = self._eig_variance_cached(candidates, n_total, n_eig)
        else:
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

            # Fused kernel: weighted prediction variance per candidate in one pass,
            # without materialising the (n_candidates x n_eig) predictions matrix.
            var_pred = self._eig_variance_fused(candidates, part_t, w_sub)

        if getattr(self, "_use_rao_blackwell_noise", False):
            est_variances = self._noise_betas / np.maximum(self._noise_alphas, 1e-9)
            noise_var = float(np.sum(self._weights * est_variances))
            noise_var = max(noise_var, 1e-12)
        else:
            noise_var = max(noise_std**2, 1e-12)

        return 0.5 * np.log1p(var_pred / noise_var)

    def _eig_variance_cached(self, candidates: np.ndarray, n_total: int, n_eig: int) -> np.ndarray:
        """Weighted prediction variance per candidate via a cached prediction matrix.

        Between resamples the particles and candidate grid are frozen, so the
        prediction matrix ``M[candidate, particle]`` is invariant and only the
        weights change. ``M`` and ``M2 = M * M`` are built once per epoch and the
        per-step variance is two matrix-vector products against the current
        weights:

            Var_w[pred] = (M2 @ w) - (M @ w) ** 2

        The particle subset is fixed for the epoch (drawn when the matrix is
        built), so the estimator uses explicit importance weights instead of the
        per-step weight-stratified resample used by the fused path.
        """
        n_c = candidates.shape[0]
        # Key from the raw array -- no dtype/contiguity conversion needed just to
        # read a length and two endpoint values. The float32-contiguous copy is
        # only actually consumed inside the cache-miss branch below, so it must
        # not be paid on every call regardless of whether the epoch matrix hits.
        key = (
            (self._eig_epoch, n_c, float(candidates[0]), float(candidates[-1]))
            if n_c
            else (self._eig_epoch, 0, 0.0, 0.0)
        )

        cache = self._eig_cache
        if cache is not None and cache[0] == key:
            _, mat, mat2, sub_idx = cache
        else:
            cand = np.ascontiguousarray(candidates, dtype=np.float32)
            if n_total > n_eig:
                # Fix the subset for this epoch. Built right after a resample,
                # so the current weights are ~uniform and a stratified draw
                # gives broad posterior coverage.
                w_norm = self._weights / (self._weights.sum() + 1e-30)
                cdf = np.cumsum(w_norm).astype(np.float32)
                u = np.random.uniform(0.0, 1.0 / n_eig, size=n_eig).astype(np.float32)
                positions = u + np.arange(n_eig, dtype=np.float32) / n_eig
                sub_idx = _systematic_resample_indices(cdf, positions)
                part_cols = [self._particles[sub_idx, j] for j in range(self._d_signal)]
            else:
                sub_idx = None
                part_cols = [self._particles[:, j] for j in range(self._d_signal)]

            # M[candidate, particle]; C-contiguous so each row (candidate) is a
            # contiguous dot against the weight vector.
            mat = np.ascontiguousarray(self.model.compute_vectorized_many_fast(cand, part_cols), dtype=np.float32)
            mat2 = mat * mat
            self._eig_cache = (key, mat, mat2, sub_idx)

        w = self._weights if sub_idx is None else self._weights[sub_idx]
        w = np.asarray(w, dtype=np.float32)
        sw = w.sum()
        if sw > 0.0:
            w = w / sw

        mean = mat @ w
        var_pred = mat2 @ w
        var_pred -= mean * mean
        np.maximum(var_pred, 0.0, out=var_pred)
        return var_pred

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

        if kernel != "generic" and getattr(self, "_eig_needs_phys", False):
            # Unit-cube belief with a physical fused kernel: map candidates and
            # the per-parameter rows to physical space. These are tiny arrays
            # (n_candidates + d_signal x n_eig), so the linear maps cost nothing
            # next to the kernel; semantics match UnitCubeSignalModel exactly.
            from nvision.spectra.unit_cube import _unit_interval_to_physical

            x_lo, x_hi = self.model.x_bounds_phys
            xs = np.float32(x_lo) + xs * (np.float32(x_hi) - np.float32(x_lo))

            if kernel == "lorentzian":
                n_sig = 5
            elif kernel == "saturation_voigt":
                # 3 (single dip) / 4 (Zeeman) / 5 (hyperfine) / 6 (Zeeman + hyperfine).
                n_sig = self._d_signal
            else:
                n_sig = 6
            bounds = self.model.param_bounds_phys
            part_phys = np.empty((n_sig, part_t.shape[1]), dtype=np.float32)
            for j, name in enumerate(self._param_names[:n_sig]):
                lo, hi = bounds[name]
                part_phys[j] = _unit_interval_to_physical(
                    np.asarray(part_t[j], dtype=np.float32), float(lo), float(hi), name
                )
            part_t = part_phys

        if kernel == "lorentzian":
            nv_center_lorentzian_eig_variance(
                xs,
                np.ascontiguousarray(part_t[0], dtype=np.float32),  # freq
                np.ascontiguousarray(part_t[1], dtype=np.float32),  # linewidth
                np.ascontiguousarray(part_t[2], dtype=np.float32),  # split
                np.ascontiguousarray(part_t[3], dtype=np.float32),  # k_np
                self._eig_w_center,
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

        if kernel == "saturation_voigt":
            from nvision.spectra.nv_center import (
                NV_SATURATION_C_MAX,
                _saturation_voigt_reparam,
            )

            n_p = part_t.shape[1]
            # Signal columns follow parameter_names order:
            #   freq, saturation, sigma_inhom, [zeeman_split,] [split, k_np]
            # c_max is a fixed constant (NV_SATURATION_C_MAX), not a particle column.
            has_zeeman = "zeeman_split" in self._param_names
            has_hf = "split" in self._param_names
            saturation = part_t[1]
            sigma_inhom = part_t[2]
            if has_zeeman:
                zeeman_split = np.ascontiguousarray(part_t[3], dtype=np.float32)
                next_idx = 4
            else:
                zeeman_split = np.zeros(n_p, dtype=np.float32)
                next_idx = 3
            if has_hf:
                hf_split = np.ascontiguousarray(part_t[next_idx], dtype=np.float32)
                k_np = np.ascontiguousarray(part_t[next_idx + 1], dtype=np.float32)
            else:
                # Not an inferred column: use the model's own fixed line offset
                # (0 when the structure is unresolved -- one merged dip).
                hf_split = np.full(n_p, np.float32(self._eig_hf_offset), dtype=np.float32)
                k_np = np.ones(n_p, dtype=np.float32)
            c_max = np.full(n_p, np.float32(NV_SATURATION_C_MAX), dtype=np.float32)

            fwhm_total, lorentz_frac, c_total = _saturation_voigt_reparam(saturation, sigma_inhom, c_max)
            nv_center_zeeman_pseudo_voigt_eig_variance(
                xs,
                np.ascontiguousarray(part_t[0], dtype=np.float32),  # freq
                np.ascontiguousarray(fwhm_total, dtype=np.float32),
                np.ascontiguousarray(lorentz_frac, dtype=np.float32),
                zeeman_split,
                hf_split,
                k_np,
                self._eig_w_center,
                np.ascontiguousarray(c_total, dtype=np.float32),
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
                self._belief_version += 1

            # If the scan parameter was narrowed, rebuild the candidate density mixture
            # against the new (tighter) domain bounds.
            if param_name == "frequency":
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
