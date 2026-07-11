"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from numba import njit

from nvision.belief.smc_marginal import _inverse_sum_squares
from nvision.models.observation import Observation
from nvision.sim.defaults import (
    NVISION_CONVERGENCE_THRESHOLD,
    NVISION_DIP_CONFIDENCE,
    NVISION_FREQ_CRLB_SAFETY_FACTOR,
    NVISION_NOISE_BG_SPAN_FACTOR,
    NVISION_NOISE_MIN_BG_POINTS,
    NVISION_SMC_CANDIDATE_STEP_HZ,
)
from nvision.sim.locs.bayesian.dip_detection import identify_dip_candidates
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator

# Minimum number of consecutive converged checks before declaring convergence.
# Prevents false early stops on the first measurement, especially with no noise.
NVISION_CONVERGENCE_PATIENCE: int = int(os.getenv("NVISION_CONVERGENCE_PATIENCE", "8"))


def background_noise_std(
    xs: np.ndarray,
    ys: np.ndarray,
    f_hat: float,
    linewidth_hat: float,
    k: float = NVISION_NOISE_BG_SPAN_FACTOR,
    min_bg_points: int = NVISION_NOISE_MIN_BG_POINTS,
    max_dip_cluster_span_hz: float | None = None,
) -> float | None:
    """Robust noise estimate from out-of-span (background) measurements.

    Points with |x - f_hat| > span are considered background, where span is
    ``k * linewidth_hat`` widened to also clear the outermost dip when
    ``max_dip_cluster_span_hz`` (e.g. from Zeeman + hyperfine splitting) is
    given — otherwise real dips far from ``f_hat`` get misclassified as
    background and bias the MAD estimate upward.
    Returns None when fewer than ``min_bg_points`` background points exist.
    Otherwise returns the MAD-based sigma: 1.4826 * median(|y_bg - median(y_bg)|).
    """
    if len(xs) == 0:
        return None
    span = max(k * abs(linewidth_hat), max_dip_cluster_span_hz or 0.0)
    mask = np.abs(xs - f_hat) > span
    y_bg = ys[mask]
    if len(y_bg) < min_bg_points:
        return None
    med = float(np.median(y_bg))
    mad = float(np.median(np.abs(y_bg - med)))
    return 1.4826 * mad


def _max_dip_cluster_span_hz(phys_bounds: dict, est: dict | None = None) -> float | None:
    """Maximum expected span (Hz) between the outermost dips of one NV signal.

    Mirrors ``SMCMarginalDistribution._generate_epoch_candidates``: the outermost
    dips are the two Zeeman groups (± ``zeeman_split``) each displaced by the
    hyperfine split (± ``split``), so the full extent is
    ``2·zeeman + 2·split``. Returns ``None`` when neither splitting is
    modeled (single-dip signal), matching the pre-Zeeman behavior.

    Uses the belief's *current point estimate* of ``zeeman_split``/``split``
    (from ``est``) when available, rather than the prior's worst-case upper
    bound: the bound stays pinned at its max (e.g. 100 MHz Zeeman -> 200 MHz
    span) for the entire run even after the belief has localized the true
    split to a much narrower value, which starves background-point
    classification and forces repeated background-calibration passes. Falls
    back to the bound before any estimate exists (e.g. before the first
    resample) or for models without these parameters.
    """
    if est is not None and "zeeman_split" in est:
        zeeman_hz = abs(est.get("zeeman_split", 0.0))
        split_hz = abs(est.get("split", 0.0)) if "split" in est else 0.0
        if zeeman_hz > 0 or split_hz > 0:
            return 2.0 * zeeman_hz + 2.0 * split_hz
        return None

    max_zeeman_hz = phys_bounds["zeeman_split"][1] if "zeeman_split" in phys_bounds else 0.0
    max_hf_split_hz = phys_bounds["split"][1] if "split" in phys_bounds else 0.0
    if max_zeeman_hz > 0 or max_hf_split_hz > 0:
        return 2.0 * max_zeeman_hz + 2.0 * max_hf_split_hz
    return None


def _effective_max_linewidth_hz(phys_bounds: dict) -> float:
    """Upper-bound effective HWHM (Hz), lineshape-agnostic.

    Used for background-span classification and dip-clustering thresholds,
    which only need a single width scale regardless of how that width is
    parameterized by the underlying model.
    """
    if "linewidth" in phys_bounds:
        return phys_bounds["linewidth"][1]
    if "saturation" in phys_bounds and "sigma_inhom" in phys_bounds:
        from nvision.spectra.nv_center import NV_NATURAL_HWHM_HZ

        s_hi = phys_bounds["saturation"][1]
        sigma_inhom_hi = phys_bounds["sigma_inhom"][1]
        gamma_hom_hi = NV_NATURAL_HWHM_HZ * math.sqrt(1.0 + s_hi)
        return gamma_hom_hi + math.sqrt(2.0 * math.log(2.0)) * sigma_inhom_hi
    if "fwhm_total" in phys_bounds:
        return phys_bounds["fwhm_total"][1] / 2.0
    return 1e6


def _effective_linewidth_and_contrast_estimate(
    est: dict, phys_bounds: dict
) -> tuple[float, float | None]:
    """Return (effective HWHM Hz estimate, realized contrast estimate), lineshape-agnostic.

    Falls back to the bound's max effective linewidth when the current
    estimate is unavailable, mirroring the pre-existing ``lw_bounds[1]``
    fallback. Contrast is ``None`` when the model has no population-style
    amplitude estimate (e.g. dip_depth-only Voigt without a matching field).
    """
    if "linewidth" in est:
        lw_bounds = phys_bounds.get("linewidth", (0.0, 1e6))
        return est.get("linewidth", lw_bounds[1]), est.get("c_total")
    if "saturation" in est and "sigma_inhom" in est:
        from nvision.spectra.nv_center import NV_SATURATION_C_MAX, _saturation_voigt_reparam_scalar

        saturation = est.get("saturation")
        sigma_inhom = est.get("sigma_inhom")
        if saturation is None or sigma_inhom is None:
            return _effective_max_linewidth_hz(phys_bounds), None
        fwhm_total, _, c_total = _saturation_voigt_reparam_scalar(saturation, sigma_inhom, NV_SATURATION_C_MAX)
        return fwhm_total / 2.0, c_total
    if "fwhm_total" in est:
        fwhm_bounds = phys_bounds.get("fwhm_total", (0.0, 2e6))
        lw = est.get("fwhm_total", fwhm_bounds[1]) / 2.0
        return lw, est.get("dip_depth", est.get("c_total"))
    return _effective_max_linewidth_hz(phys_bounds), None


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted empirical quantile via sorted cumulative weights."""
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    cw /= cw[-1]
    return float(np.interp(q, cw, values[order]))


@dataclass(frozen=True)
class FocusWindowConfidence:
    """Joint confidence estimate for the focus window, merging empirical and posterior signals.

    Empirical fields (from dip detector, model-free):
        left_bound / right_bound — f_min/f_max of the dominant dip cluster.
        left_unc / right_unc — gap to the nearest observation outside each edge (Hz);
            inf when no observations exist on that side.
        detector_confidence — binomial confidence of the dominant cluster.
        background — empirical baseline (70th pct of obs_ys) used by the detector.

    Posterior fields (from SMC particles):
        center — weighted-mean frequency (Hz).
        center_std — weighted std of frequency (Hz).
        center_ci_lo / center_ci_hi — 16th / 84th pct of frequency posterior (Hz).

    Merged:
        methods_agree — True when center ∈ [left_bound, right_bound].
        is_stable — agreement-gated: detector_confidence ≥ floor AND methods_agree
                    AND center_std < stability_ratio × window_width.
    """

    left_bound: float
    right_bound: float
    left_unc: float
    right_unc: float
    detector_confidence: float
    background: float
    center: float
    center_std: float
    center_ci_lo: float
    center_ci_hi: float
    methods_agree: bool
    is_stable: bool


def compute_focus_window_confidence(
    belief,
    noise_std: float,
    stability_ratio: float = 0.5,
    confidence_floor: float = NVISION_DIP_CONFIDENCE,
) -> FocusWindowConfidence | None:
    """Compute focus window confidence by merging the empirical dip detector with the particle posterior.

    Returns None when fewer than 5 observations exist, when the belief lacks particles,
    or when the empirical detector does not qualify any cluster (still in early search).
    """
    if not hasattr(belief, "_particles") or not hasattr(belief, "_weights"):
        return None

    # --- Build observation arrays in physical Hz ---
    rescale_maps = getattr(belief, "_rescale_maps", None)
    if rescale_maps is None or "frequency" not in rescale_maps:
        return None

    if hasattr(belief, "observation_arrays"):
        obs_xs_unit, obs_ys = belief.observation_arrays()
        obs_xs_phys = rescale_maps["frequency"].to_phys(obs_xs_unit)
    else:
        obs_list = getattr(belief, "_observations", [])
        if not obs_list:
            return None
        obs_xs_phys = rescale_maps["frequency"].to_phys(np.array([o.x for o in obs_list]))
        obs_ys = np.array([o.signal_value for o in obs_list])

    if len(obs_xs_phys) < 5:
        return None

    # --- Build identify_dip_candidates kwargs (mirrors _acquire / _check_crlb_early_stop) ---
    phys_bounds = getattr(belief, "physical_param_bounds", getattr(belief, "parameter_bounds", {}))
    max_linewidth_hz = _effective_max_linewidth_hz(phys_bounds)
    max_split_hz = _max_dip_cluster_span_hz(phys_bounds, belief.estimates())

    noise_std_unc = 0.0
    per_particle_sigmas = None
    particle_weights = None  # only set when per_particle_sigmas is also set (arrays must match)
    if getattr(belief, "_use_rao_blackwell_noise", False):
        per_particle_sigmas = np.sqrt(belief._noise_betas / (belief._noise_alphas + 0.5))
        particle_weights = belief._weights
    elif hasattr(belief, "_param_names") and "noise_sigma" in belief._param_names:
        idx = belief._param_names.index("noise_sigma")
        raw_sigmas = belief._particles[:, idx]
        if "noise_sigma" in phys_bounds:
            lo_ns, hi_ns = phys_bounds["noise_sigma"]
            per_particle_sigmas = lo_ns + raw_sigmas * (hi_ns - lo_ns)
        else:
            per_particle_sigmas = raw_sigmas
        particle_weights = belief._weights

    if hasattr(belief, "estimated_noise_std") and getattr(belief, "noise_model", None) is not None:
        noise_std_used = belief.estimated_noise_std()
        noise_std_unc = belief.noise_std_uncertainty(noise_std_used)
    else:
        noise_std_used = noise_std

    candidates = identify_dip_candidates(
        obs_xs_phys,
        obs_ys,
        noise_std_used,
        max_linewidth_hz,
        noise_std_unc=noise_std_unc,
        per_particle_sigmas=per_particle_sigmas,
        particle_weights=particle_weights,
        max_split_hz=max_split_hz,
    )

    if not candidates:
        return None

    c = candidates[0]  # already sorted by significance descending
    left_bound = c.f_min
    right_bound = c.f_max

    # --- Per-side sampling-gap uncertainty ---
    left_outside = obs_xs_phys[obs_xs_phys < left_bound]
    left_unc = float(left_bound - left_outside.max()) if len(left_outside) > 0 else math.inf
    right_outside = obs_xs_phys[obs_xs_phys > right_bound]
    right_unc = float(right_outside.min() - right_bound) if len(right_outside) > 0 else math.inf

    # --- Posterior center from particles ---
    param_names = list(belief._param_names)
    if "frequency" not in param_names:
        return None
    f_idx = param_names.index("frequency")
    f_unit = belief._particles[:, f_idx]
    f_phys = rescale_maps["frequency"].to_phys(f_unit)
    w = belief._weights
    sw = float(w.sum())
    if sw <= 0:
        return None

    center = float((w @ f_phys) / sw)
    center_std = float(np.sqrt(max(0.0, float((w @ ((f_phys - center) ** 2)) / sw))))
    center_ci_lo = _weighted_quantile(f_phys, w, 0.16)
    center_ci_hi = _weighted_quantile(f_phys, w, 0.84)

    # --- Merge ---
    methods_agree = left_bound <= center <= right_bound
    window_width = right_bound - left_bound
    is_stable = (
        c.confidence >= confidence_floor
        and methods_agree
        and window_width > 0
        and center_std < stability_ratio * window_width
    )

    return FocusWindowConfidence(
        left_bound=left_bound,
        right_bound=right_bound,
        left_unc=left_unc,
        right_unc=right_unc,
        detector_confidence=c.confidence,
        background=c.background,
        center=center,
        center_std=center_std,
        center_ci_lo=center_ci_lo,
        center_ci_hi=center_ci_hi,
        methods_agree=methods_agree,
        is_stable=is_stable,
    )


@njit(cache=True)
def _thin_by_step_indices(candidates: np.ndarray, step: float) -> np.ndarray:
    """Indices of a greedy minimum-spacing subset of sorted *candidates*.

    Keeps the first and last candidates so the full range stays represented.
    """
    n = candidates.shape[0]
    kept = np.empty(n, dtype=np.int64)
    kept[0] = 0
    m = 1
    last = candidates[0]
    for i in range(1, n - 1):
        if candidates[i] - last >= step:
            kept[m] = i
            m += 1
            last = candidates[i]
    kept[m] = n - 1
    return kept[: m + 1]


class SequentialBayesianExperimentDesignLocator(SequentialBayesianLocator):
    """Sequential Bayesian Experiment Design acquisition.

    Uses Expected Information Gain (prediction variance disagreement) to select
    the next measurement point from a fine frequency grid. No JAX gradient ascent
    is performed — the chunked EIG search on the belief is sufficient.
    """

    REQUIRES_BELIEF = True
    USES_SWEEP_MAX_STEPS = True

    def __init__(
        self,
        belief,
        max_steps: int = 150,
        convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
        scan_param: str | None = None,
        noise_std: float = 0.02,
        candidate_step_hz: float | None = None,
        convergence_patience_steps: int = NVISION_CONVERGENCE_PATIENCE,
    ) -> None:
        super().__init__(
            belief,
            max_steps,
            convergence_threshold,
            scan_param,
            noise_std=noise_std,
            convergence_patience_steps=convergence_patience_steps,
        )
        self.candidate_step_hz: float = (
            float(candidate_step_hz) if candidate_step_hz is not None else NVISION_SMC_CANDIDATE_STEP_HZ
        )

        # We handle resampling manually to check convergence at the right moment
        if hasattr(self.belief, "auto_resample"):
            self.belief.auto_resample = False
        self._is_converged = False

        # Background noise estimation / forced calibration state
        self._forced_bg_mode: bool = False
        self._forced_bg_measurements: int = 0
        self._bg_noise_std: float | None = None
        self._bg_points_used: int = 0
        self._focus_window_conf: FocusWindowConfidence | None = None
        self.focus_stable_step: int | None = None
        # Empirical batch-mean noise (running mean of received obs.noise_std over
        # multi-shot batches). Reflects the σ/√k precision of the measurement the
        # locator will actually take; used for EIG scoring and as a noise-floor
        # fallback for the CRLB early-stop before background points accumulate.
        self._empirical_batch_noise_std: float | None = None
        self._batch_noise_sum: float = 0.0
        self._batch_noise_count: int = 0
        # Physical frequency of the most recent EIG selection, re-injected into
        # the candidate grid so a second batch there is a legitimate EIG outcome
        # rather than being dropped by minimum-spacing thinning.
        self._last_eig_physical_x: float | None = None
        # Theoretical step budget: K_theory × n_theory, computed from σ̂ + belief estimates.
        # None until the first valid background noise estimate is available.
        self._theory_step_budget: int | None = None

    @classmethod
    def create(
        cls,
        builder=None,
        max_steps: int = 150,
        convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
        scan_param: str | None = None,
        parameter_bounds=None,
        noise_std: float | None = None,
        candidate_step_hz: float | None = None,
        convergence_patience_steps: int = NVISION_CONVERGENCE_PATIENCE,
        **grid_config,
    ):
        if builder is None:
            raise ValueError(f"{cls.__name__} requires a 'builder' callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            noise_std=noise_std,
            candidate_step_hz=candidate_step_hz,
            convergence_patience_steps=convergence_patience_steps,
        )

    def _generate_candidates(self, num_candidates: int | None = None) -> np.ndarray:
        """Generate candidates spanning the whole frequency spectrum.

        When ``num_candidates`` is not provided, compute it dynamically from
        the acquisition bounds so that the step resolution is two decimal
        places finer than the order of magnitude of the range.
        """
        if num_candidates is None:
            lo, hi = self._acquisition_bounds()
            range_val = float(hi - lo)
            if range_val <= 0:
                num_candidates = 1
            else:
                magnitude = math.floor(math.log10(range_val))
                resolution = 10 ** (magnitude - 2)
                num_candidates = max(1, math.ceil(range_val / resolution)) + 1
        return super()._generate_candidates(num_candidates)

    def _thin_candidates_by_step(self, candidates: np.ndarray) -> np.ndarray:
        """Return a subset of *candidates* (physical space) with minimum physical spacing.

        Walks the sorted candidate array once and keeps a candidate only when it
        is at least ``candidate_step_hz`` away from the previously kept one.
        This is O(n) and preserves the first and last candidates so the full
        acquisition range is always represented.
        """
        if len(candidates) <= 1:
            return candidates
        kept = _thin_by_step_indices(np.ascontiguousarray(candidates, dtype=np.float64), self.candidate_step_hz)
        return candidates[kept]

    def _acquire(self) -> float:
        """Select the next measurement point by maximizing EIG over a frequency grid."""
        lo, hi = self._acquisition_bounds()
        if hi <= lo:
            return float(lo)

        # Forced background calibration: sample out-of-span until we have
        # enough background points to estimate noise for the CRLB early-stop.
        if self._forced_bg_mode:
            est = self.belief.estimates()
            f_hat = est.get("frequency", (lo + hi) / 2.0)
            lw_hat, _ = _effective_linewidth_and_contrast_estimate(est, self.belief.physical_param_bounds)
            max_split_hz = _max_dip_cluster_span_hz(self.belief.physical_param_bounds, est)
            # Must match the classification span in background_noise_std() below, or forced
            # sampling lands points that its own classifier then rejects as non-background.
            span = max(NVISION_NOISE_BG_SPAN_FACTOR * abs(lw_hat), max_split_hz or 0.0)
            # Sample from left or right background region at random
            left_lo, left_hi = lo, max(lo, f_hat - span)
            right_lo, right_hi = min(hi, f_hat + span), hi
            left_width = max(0.0, left_hi - left_lo)
            right_width = max(0.0, right_hi - right_lo)
            total_width = left_width + right_width
            if total_width > 0:
                if np.random.rand() < left_width / total_width:
                    return float(np.random.uniform(left_lo, left_hi))
                else:
                    return float(np.random.uniform(right_lo, right_hi))
            # No valid background region — fall through to normal acquisition
            self._forced_bg_mode = False

        # Mix EIG with dip-observation-biased exploration. The exploration/dip
        # branches are drawn first so the (much more expensive) EIG grid search
        # in _eig_acquire() is skipped entirely on steps where it would be
        # discarded anyway.
        # The uniform exploration probability decays exponentially to focus on EIG as the scan progresses.
        decay = np.exp(-self.inference_step_count / 25.0)
        rand_val = np.random.rand()
        if rand_val < 0.1 * decay:
            # Explore globally uniformly to find missing peaks (probability decays over time)
            return float(np.random.uniform(lo, hi))
        elif rand_val < 0.2:
            # Dip-observation biased sampling: find the empirically lowest measured signal values
            # and draw near one of them. This corrects for a biased posterior that has drifted
            # away from the true dip location.
            n_obs = getattr(self.belief, "num_observations", None)
            if n_obs is None:
                n_obs = len(getattr(self.belief, "_observations", []))
            if n_obs >= 5:
                # Use precomputed/cached dip centers from the belief (calculated only during resampling)
                # to satisfy "dip detection only upon resampling" and avoid massive sorting overhead.
                dip_centers = getattr(self.belief, "_dip_centers", None)
                if dip_centers is None:
                    # Fallback (e.g. if using a belief type that does not precompute them)
                    rescale_maps = self.belief._rescale_maps
                    if "frequency" not in rescale_maps:
                        raise RuntimeError(
                            f"{type(self.belief).__name__} is missing _rescale_maps['frequency']. "
                            "Ensure physical_param_bounds includes 'frequency' at construction."
                        )
                    freq_rescale = rescale_maps["frequency"]
                    if hasattr(self.belief, "observation_arrays"):
                        obs_xs_unit, obs_ys = self.belief.observation_arrays()
                        obs_xs_phys = freq_rescale.to_phys(obs_xs_unit)
                    else:
                        obs_list = self.belief._observations
                        obs_xs_phys = freq_rescale.to_phys(np.array([o.x for o in obs_list]))
                        obs_ys = np.array([o.signal_value for o in obs_list])
                    if (
                        hasattr(self.belief, "estimated_noise_std")
                        and getattr(self.belief, "noise_model", None) is not None
                    ):
                        noise_std = self.belief.estimated_noise_std()
                        noise_std_unc = self.belief.noise_std_uncertainty(noise_std)
                    else:
                        noise_std = self._noise_std
                        noise_std_unc = 0.0

                    phys_bounds = getattr(
                        self.belief, "physical_param_bounds", getattr(self.belief, "parameter_bounds", {})
                    )
                    max_linewidth_hz = _effective_max_linewidth_hz(phys_bounds)
                    max_split_hz = _max_dip_cluster_span_hz(phys_bounds, self.belief.estimates())

                    per_particle_sigmas = None
                    particle_weights = None
                    if hasattr(self.belief, "_weights"):
                        particle_weights = self.belief._weights
                        if getattr(self.belief, "_use_rao_blackwell_noise", False):
                            per_particle_sigmas = np.sqrt(self.belief._noise_betas / (self.belief._noise_alphas + 0.5))
                        elif hasattr(self.belief, "_param_names") and "noise_sigma" in self.belief._param_names:
                            idx = self.belief._param_names.index("noise_sigma")
                            raw_sigmas = self.belief._particles[:, idx]
                            if (
                                hasattr(self.belief, "physical_param_bounds")
                                and "noise_sigma" in self.belief.physical_param_bounds
                            ):
                                lo_ns, hi_ns = self.belief.physical_param_bounds["noise_sigma"]
                                per_particle_sigmas = lo_ns + raw_sigmas * (hi_ns - lo_ns)
                            else:
                                per_particle_sigmas = raw_sigmas

                    dip_candidates = identify_dip_candidates(
                        obs_xs_phys,
                        obs_ys,
                        noise_std,
                        max_linewidth_hz,
                        noise_std_unc=noise_std_unc,
                        per_particle_sigmas=per_particle_sigmas,
                        particle_weights=particle_weights,
                        max_split_hz=max_split_hz,
                    )
                    dip_centers = [c.centroid_hz for c in dip_candidates]

                valid_dip_centers = [c for c in dip_centers if lo <= c <= hi]
                if valid_dip_centers:
                    # Pick a random dip centroid and jitter within ±5 MHz around it, keeping it strictly within [lo, hi]
                    center = float(np.random.choice(valid_dip_centers))
                    j_min = max(-5e6, lo - center)
                    j_max = min(5e6, hi - center)
                    if j_max >= j_min:
                        jitter = float(np.random.uniform(j_min, j_max))
                        val = center + jitter
                        if not (lo <= val <= hi):
                            raise ValueError(f"Jittered dip value {val} is outside acquisition bounds {(lo, hi)}")
                        return val
                    else:
                        if not (lo <= center <= hi):
                            raise ValueError(f"Dip center {center} is outside acquisition bounds {(lo, hi)}")
                        return center

            # Fallback: Thompson sampling from posterior particles
            if hasattr(self.belief, "_particles") and hasattr(self.belief, "_weights"):
                weights = self.belief._weights
                if np.sum(weights) > 0:
                    idx = int(np.random.choice(len(weights), p=weights))
                    param_names = getattr(self.belief, "_param_names", [])
                    scan_param = self._scan_param
                    if scan_param in param_names:
                        p_idx = param_names.index(scan_param)
                        val = float(self.belief._particles[idx, p_idx])
                        return self.belief._to_physical(scan_param, val)

        return self._eig_acquire()

    def _eig_acquire(self) -> float:
        """Maximize EIG over the belief's slope-targeted candidate grid."""
        # Retrieve candidates directly from the belief (slope-targeted epoch grid)
        candidates = self.belief.get_candidates()

        # Thin candidates to minimum physical step spacing.
        # The epoch grid window is ±3σ_f, so candidate count ≈ 6σ_f / step_hz:
        # many candidates early (large σ_f), few near convergence (σ_f ≈ step_hz).
        candidates = self._thin_candidates_by_step(candidates)

        # Keep the most recently EIG-selected frequency in the candidate set so a
        # second batch there is a legitimate EIG outcome rather than being dropped
        # by minimum-spacing thinning. EIG's diminishing returns decide when
        # re-batching stops paying off (no explicit repeat counter needed).
        lo, hi = self._acquisition_bounds()
        if (
            self._last_eig_physical_x is not None
            and lo <= self._last_eig_physical_x <= hi
            and not np.any(np.isclose(candidates, self._last_eig_physical_x))
        ):
            candidates = np.append(candidates, self._last_eig_physical_x)

        # Score candidates with the noise of the measurement actually taken: a
        # k-shot batch with precision σ/√k. Falls back to the constructor prior
        # until an empirical batch estimate exists.
        noise_std_for_eig = (
            self._empirical_batch_noise_std if self._empirical_batch_noise_std is not None else self._noise_std
        )
        best = self.belief.select_max_information_gain(candidates, 1, noise_std=noise_std_for_eig)
        result = float(best[0]) if len(best) > 0 else float(candidates[len(candidates) // 2])
        self._last_eig_physical_x = result
        return result

    def _observe_acquisition(self, obs: Observation) -> None:
        """Handle acquisition observations and manually trigger resample checks.

        Updates the belief directly instead of via super(), which would run the
        convergence-milestone check a second time — _check_and_resample already
        performs it with a single shared uncertainty pass.
        """
        if self._forced_bg_mode:
            self._forced_bg_measurements += 1
        # Track the running mean of batch-mean noise (s/√k) from multi-shot
        # batches. This is a direct, model-free estimate of the precision of the
        # measurements the locator receives, available from the first k≥2 batch.
        n_shots = int(getattr(obs, "n_shots", 1) or 1)
        if n_shots >= 2 and obs.noise_std > 0:
            self._batch_noise_count += 1
            self._batch_noise_sum += float(obs.noise_std)
            self._empirical_batch_noise_std = self._batch_noise_sum / self._batch_noise_count
        self.belief.update(obs)
        self.belief.accumulate_fim(obs)
        self._check_and_resample(check_convergence=True)

    def _check_and_resample(self, check_convergence: bool = True) -> None:
        if not hasattr(self.belief, "_weights"):
            return
        ess = _inverse_sum_squares(self.belief._weights)
        ess_threshold = getattr(self.belief, "ess_threshold", 0.0) * getattr(self.belief, "num_particles", 0)
        did_resample = False
        if ess < ess_threshold:
            if hasattr(self.belief, "_resample"):
                self.belief._resample()
                did_resample = True

        if check_convergence:
            # One uncertainty pass shared by the streak and milestone checks
            # (each belief.uncertainty() call is a full O(particles x params) pass).
            physical_uncertainties = self.belief.uncertainty()
            if self._target_params_converged(physical_uncertainties):
                self._convergence_streak += 1
                if self._convergence_streak >= self._convergence_patience_steps:
                    self._is_converged = True
            else:
                self._convergence_streak = 0
            self._check_convergence_milestones(physical_uncertainties)

            # CRLB-based early-stop: evaluated every step for deterministic convergence reporting.
            # Uses background-scatter noise estimate (robust to signal-region bias).
            if not self._is_converged:
                self._check_crlb_early_stop(physical_uncertainties)
                conf = compute_focus_window_confidence(self.belief, noise_std=self._noise_std)
                if conf is not None:
                    self._focus_window_conf = conf
                    if conf.is_stable and self.focus_stable_step is None:
                        self.focus_stable_step = self.step_count

    def _acquisition_done(self) -> bool:
        """Extend base stop logic with a permissive theory-step-budget backstop.

        When enough background points exist to estimate σ̂, computes the theoretical
        minimum step count for uniform sampling to reach the convergence threshold
        and stops if ``inference_step_count`` exceeds ``K_theory × n_theory``.
        This only fires when something has gone wrong — EIG should converge much sooner.
        """
        if super()._acquisition_done():
            return True
        if self._theory_step_budget is not None and self.inference_step_count > self._theory_step_budget:
            return True
        return False

    def _check_crlb_early_stop(self, physical_uncertainties) -> None:
        """Background-noise CRLB early-stop check, run on resample events only.

        Estimates the noise floor from out-of-span (background) measurements via
        MAD.  If enough background points exist and every target parameter's
        uncertainty is within NVISION_FREQ_CRLB_SAFETY_FACTOR × CRLB, marks the
        run as converged.  Otherwise triggers forced background calibration if
        the noise estimate is unavailable.
        """

        # Get observation arrays
        if hasattr(self.belief, "observation_arrays"):
            obs_xs_unit, obs_ys = self.belief.observation_arrays()
            rescale_maps = getattr(self.belief, "_rescale_maps", None)
            if rescale_maps and "frequency" in rescale_maps:
                obs_xs_phys = rescale_maps["frequency"].to_phys(obs_xs_unit)
            else:
                obs_xs_phys = obs_xs_unit
        else:
            obs_list = getattr(self.belief, "_observations", [])
            if not obs_list:
                return
            obs_xs_phys = np.array([o.x for o in obs_list])
            obs_ys = np.array([o.signal_value for o in obs_list])

        # Get current frequency and linewidth estimates for background classification
        est = self.belief.estimates()
        f_hat = est.get("frequency", float(np.mean(obs_xs_phys)))
        lw_hat, c_hat = _effective_linewidth_and_contrast_estimate(est, self.belief.physical_param_bounds)
        max_split_hz = _max_dip_cluster_span_hz(self.belief.physical_param_bounds, est)

        span = max(NVISION_NOISE_BG_SPAN_FACTOR * abs(lw_hat), max_split_hz or 0.0)
        bg_count = int(np.sum(np.abs(obs_xs_phys - f_hat) > span))
        sigma_hat = background_noise_std(
            obs_xs_phys, obs_ys, f_hat, lw_hat, max_dip_cluster_span_hz=max_split_hz
        )

        if (sigma_hat is None or sigma_hat <= 0) and self._empirical_batch_noise_std is not None:
            # No background points yet, but multi-shot batches give a direct,
            # model-free noise estimate (same batch-mean units as the MAD estimate
            # above), so the CRLB early-stop need not wait for calibration.
            sigma_hat = self._empirical_batch_noise_std

        if sigma_hat is None or sigma_hat <= 0:
            # Not enough background points — trigger forced calibration
            self._forced_bg_mode = True
            return

        self._bg_noise_std = sigma_hat
        self._bg_points_used = bg_count
        self._forced_bg_mode = False

        # Compute permissive theoretical step budget from σ̂ + belief estimates.
        # n_theory = 4σ̂²·lw·bandwidth / (π·c²·T²) — steps needed for uniform sampling to reach T.
        # SBED is better than uniform. (4, not 2: matches the verified CRLB constant in
        # UnitCubeSMCMarginalDistribution.crlb_frequency — n_theory is that same CRLB_var
        # solved for n at rho = n/bandwidth, evaluated at the convergence threshold T.)
        if c_hat is not None and c_hat > 0 and lw_hat > 0:
            freq_lo, freq_hi = self.belief.physical_param_bounds.get("frequency", (0.0, 1.0))
            bandwidth = freq_hi - freq_lo
            if bandwidth > 0:
                from nvision.sim.defaults import NVISION_FREQ_CONVERGENCE_THRESHOLD, NVISION_SBED_STEPS_THEORY_FACTOR
                n_theory = (4.0 * sigma_hat**2 * lw_hat * bandwidth) / (
                    math.pi * c_hat**2 * NVISION_FREQ_CONVERGENCE_THRESHOLD**2
                )
                self._theory_step_budget = max(self.max_steps, int(NVISION_SBED_STEPS_THEORY_FACTOR * n_theory) + 1)

        # Build per-param CRLBs scaled to σ̂.
        # Primary: cumulative FIM accumulated during observations (needs model.gradient).
        # Fallback: analytical closed-form crlb_frequency() for frequency only.
        crlbs_stored = self.belief.crlb_per_param()
        scale = sigma_hat / max(self._noise_std, 1e-12)

        if not crlbs_stored:
            # Model has no analytical gradient → use closed-form frequency CRLB only.
            # crlb_frequency() is computed at last_obs.noise_std ≈ self._noise_std.
            # Store the raw value; the `* scale` below converts to σ̂.
            crlb_fn = getattr(self.belief, "crlb_frequency", None)
            if crlb_fn is None:
                return
            crlb_f_raw = crlb_fn()
            if not math.isfinite(crlb_f_raw) or crlb_f_raw <= 0:
                return
            crlbs_stored = {"frequency": crlb_f_raw}

        bounds = self.belief.physical_param_bounds
        target_params = (
            list(self._convergence_params)
            if self._convergence_params
            else list(self.belief.model.parameter_names())
        )

        from nvision.sim.defaults import PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS
        from nvision.sim.locs.bayesian.sequential_bayesian_locator import (
            _SATURATION_VOIGT_RAW_PARAMS,
            saturation_voigt_derived_bounds,
            saturation_voigt_derived_sigmas,
        )

        # Saturation-Voigt: gate via derived effective-HWHM / realized-contrast rather
        # than the raw params (see saturation_voigt_derived_sigmas). Uncertainties and
        # CRLBs propagate through the same local Jacobian, so both are mapped.
        derived_unc: dict[str, float] | None = None
        derived_crlbs: dict[str, float] = {}
        if "saturation" in physical_uncertainties and "sigma_inhom" in physical_uncertainties:
            est = self.belief.estimates()
            derived_unc = saturation_voigt_derived_sigmas(est, physical_uncertainties)
            if derived_unc is not None:
                scaled_crlbs = {p: crlbs_stored.get(p, math.inf) * scale for p in _SATURATION_VOIGT_RAW_PARAMS}
                derived_crlbs = saturation_voigt_derived_sigmas(est, scaled_crlbs) or {}
                bounds = {**bounds, **saturation_voigt_derived_bounds(bounds)}

        # (name, uncertainty, crlb already scaled to sigma-hat) triples to evaluate.
        eval_items: list[tuple[str, float, float]] = [
            (
                name,
                float(physical_uncertainties.get(name, math.inf)),
                crlbs_stored.get(name, math.inf) * scale,
            )
            for name in target_params
            if not (derived_unc is not None and name in _SATURATION_VOIGT_RAW_PARAMS)
        ]
        if derived_unc is not None:
            eval_items.extend(
                (dname, dunc, derived_crlbs.get(dname, math.inf)) for dname, dunc in derived_unc.items()
            )

        # Per-param evaluation. For each param with a valid threshold:
        #   done = unc < convergence threshold
        #
        # Stop conditions:
        #   all_converged_step  : ALL checked params are done (abs or crlb)
        #   freq_converged_step : frequency is done (abs or crlb)
        #   _is_converged       : ALL checked params hit strict CRLB floor
        checked = 0
        all_crlb_done = True
        all_milestone_done = True
        freq_milestone_done = False

        for name, unc, crlb_scaled in eval_items:
            # 1. CRLB check
            crlb_threshold = NVISION_FREQ_CRLB_SAFETY_FACTOR * crlb_scaled
            crlb_done = unc < crlb_threshold if crlb_threshold > 0 and math.isfinite(crlb_threshold) else False

            # 2. Absolute threshold check
            absolute_threshold = PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS.get(name)
            if absolute_threshold is None:
                lo, hi = bounds.get(name, (0.0, 0.0))
                absolute_threshold = (hi - lo) * self.convergence_threshold
            abs_done = unc < absolute_threshold if absolute_threshold > 0 else False

            # We must be able to evaluate at least one threshold to count this param
            if not (crlb_threshold > 0 and math.isfinite(crlb_threshold)) and not (absolute_threshold > 0):
                continue

            checked += 1

            if not crlb_done:
                all_crlb_done = False
                
            if not (crlb_done or abs_done):
                all_milestone_done = False
                
            if name == "frequency" and (crlb_done or abs_done):
                freq_milestone_done = True

        if checked == 0:
            return

        if all_crlb_done:
            self._is_converged = True

        if freq_milestone_done and self.freq_converged_step is None:
            self.freq_converged_step = self.step_count

        if all_milestone_done and self.all_converged_step is None:
            self.all_converged_step = self.step_count
