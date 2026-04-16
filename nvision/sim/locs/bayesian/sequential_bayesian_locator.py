"""Abstract base class for all Bayesian locators."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter
from nvision.belief.unit_cube_grid_marginal import UnitCubeGridMarginalDistribution
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.models.locator import Locator
from nvision.models.measurement_noise import DEFAULT_MEASUREMENT_NOISE_STD
from nvision.models.observation import Observation
from nvision.sim.locs.coarse.sobol_locator import sobol_1d_sequence


_POSTERIOR_NARROWING_INTERVAL: int = 20
_POSTERIOR_CREDIBLE_LEVEL: float = 0.95
_POSTERIOR_MIN_NARROWING_FRACTION: float = 0.05


def _posterior_credible_interval(
    belief: UnitCubeGridMarginalDistribution | UnitCubeSMCMarginalDistribution,
    param_name: str,
    level: float = _POSTERIOR_CREDIBLE_LEVEL,
) -> tuple[float, float] | None:
    """Return a ``level``-credible interval in physical units for ``param_name``.

    For grid beliefs: uses the marginal CDF to find the equal-tailed interval.
    For SMC beliefs: uses weighted particle quantiles.
    Returns ``None`` if the interval cannot be computed or is degenerate.
    """
    tail = (1.0 - level) / 2.0
    lo_phys, hi_phys = belief.physical_param_bounds[param_name]
    if hi_phys <= lo_phys:
        return None

    if isinstance(belief, UnitCubeGridMarginalDistribution):
        p = GridMarginalDistribution.get_grid_param(belief, param_name)
        cdf = np.cumsum(p.posterior)
        if cdf[-1] <= 0:
            return None
        cdf = cdf / cdf[-1]
        u_lo = float(np.interp(tail, cdf, p.grid))
        u_hi = float(np.interp(1.0 - tail, cdf, p.grid))
        return (lo_phys + u_lo * (hi_phys - lo_phys), lo_phys + u_hi * (hi_phys - lo_phys))

    if isinstance(belief, UnitCubeSMCMarginalDistribution):
        j = belief._param_names.index(param_name)
        u_vals = belief._particles[:, j]
        u_lo = float(np.quantile(u_vals, tail))
        u_hi = float(np.quantile(u_vals, 1.0 - tail))
        return (lo_phys + u_lo * (hi_phys - lo_phys), lo_phys + u_hi * (hi_phys - lo_phys))

    return None


def _normalized_acquisition_interval_from_sweep(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float] | None:
    """Return ``(lo, hi)`` in normalized [0, 1] scan coordinates, or ``None`` to use the full domain.

    Detects signal dips by looking for local minima — points significantly below
    their neighbors. This is more robust than deviation-from-median for NV-style
    signals where the feature is a dip from a high baseline.
    """
    if xs.size < 5 or ys.size < 5:
        return None

    padding_fraction = 0.02
    min_width_fraction = 0.15
    segment_peak_ratio = 0.6
    merge_gap_factor = 2.0

    # Sort by x for neighborhood analysis
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]

    # Compute local "dip score": how much below neighbors (left and right)
    # Use 2-point neighborhood to avoid noise sensitivity
    n = len(ys_sorted)
    dip_scores = np.zeros(n)
    for i in range(1, n - 1):
        left_neighbor = ys_sorted[i - 1]
        right_neighbor = ys_sorted[i + 1]
        local_baseline = 0.5 * (left_neighbor + right_neighbor)
        # Positive dip_score means point is below neighbors (a dip)
        dip_scores[i] = max(0.0, local_baseline - ys_sorted[i])

    # Boundary points: compare to single neighbor
    if n > 1:
        dip_scores[0] = max(0.0, ys_sorted[1] - ys_sorted[0])
        dip_scores[-1] = max(0.0, ys_sorted[-2] - ys_sorted[-1])

    # Keep points with meaningful dips (top 60% of dip scores)
    thr = float(np.quantile(dip_scores, 0.4))  # 0.4 quantile = bottom 40%, so top 60% pass
    keep = dip_scores > thr
    if not np.any(keep):
        return None

    x_keep = xs_sorted[keep]
    info_keep = dip_scores[keep]
    if x_keep.size == 0:
        return None

    # Segment by gaps
    diffs_all = np.diff(xs_sorted)
    positive_diffs = diffs_all[diffs_all > 0]
    median_dx = float(np.median(positive_diffs)) if positive_diffs.size else 0.0
    split_gap = max(3.0 * median_dx, 1e-6)
    seg_breaks = np.where(np.diff(x_keep) > split_gap)[0] + 1
    seg_xs = np.split(x_keep, seg_breaks)
    seg_dips = np.split(info_keep, seg_breaks)
    seg_peaks = np.array([float(np.max(s)) for s in seg_dips], dtype=float)
    if seg_peaks.size == 0:
        return None

    # Select segments with strong dips
    peak_thr = segment_peak_ratio * float(np.max(seg_peaks))
    selected = [i for i, peak in enumerate(seg_peaks) if peak >= peak_thr]
    if not selected:
        selected = [int(np.argmax(seg_peaks))]

    selected_intervals = sorted((float(seg_xs[i][0]), float(seg_xs[i][-1])) for i in selected)
    merge_gap = merge_gap_factor * median_dx
    merged: list[list[float]] = []
    for start, end in selected_intervals:
        if not merged or start - merged[-1][1] > merge_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    lo_norm = float(merged[0][0])
    hi_norm = float(merged[-1][1])

    span_norm = hi_norm - lo_norm
    lo_norm -= padding_fraction * max(span_norm, 1e-9)
    hi_norm += padding_fraction * max(span_norm, 1e-9)
    lo_norm = float(np.clip(lo_norm, 0.0, 1.0))
    hi_norm = float(np.clip(hi_norm, 0.0, 1.0))
    if hi_norm - lo_norm < min_width_fraction:
        mid = 0.5 * (lo_norm + hi_norm)
        half = 0.5 * min_width_fraction
        lo_norm = float(np.clip(mid - half, 0.0, 1.0))
        hi_norm = float(np.clip(mid + half, 0.0, 1.0))
        if hi_norm - lo_norm < min_width_fraction:
            lo_norm = max(0.0, hi_norm - min_width_fraction)
            hi_norm = min(1.0, lo_norm + min_width_fraction)

    return (lo_norm, hi_norm)


def _detect_dip_edges(xs_sorted: np.ndarray, ys_sorted: np.ndarray, min_idx: int) -> tuple[int, int]:
    """Find left and right edges of a dip around its minimum point.

    A dip is the pattern: baseline -> down (decreasing y) -> minimum -> up (increasing y) -> baseline.
    Returns (left_edge_idx, right_edge_idx) as indices into the sorted arrays.
    """
    n = len(ys_sorted)
    if n < 3:
        return (0, n - 1)

    # Estimate baseline from upper quartile (assuming most points are baseline)
    baseline = float(np.percentile(ys_sorted, 75))
    min_signal = float(ys_sorted[min_idx])
    dip_depth = baseline - min_signal
    if dip_depth <= 0:
        return (max(0, min_idx - 1), min(n - 1, min_idx + 1))

    # Threshold for being "in the dip" vs at baseline
    # Use 50% of dip depth: if signal is within 50% of baseline, it's still in the dip
    dip_region_threshold = baseline - 0.5 * dip_depth

    # Find left edge: move left from minimum until we leave the dip region
    left_idx = min_idx
    while left_idx > 0:
        prev_y = float(ys_sorted[left_idx - 1])

        # If previous point is at/near baseline (above threshold), we've left the dip
        if prev_y >= dip_region_threshold:
            # Include one more point at/beyond baseline for full coverage
            if left_idx > 1:
                left_idx -= 1  # Step to the baseline point
            else:
                left_idx = 0
            break

        left_idx -= 1

    # Find right edge: move right from minimum until we leave the dip region
    right_idx = min_idx
    while right_idx < n - 1:
        next_y = float(ys_sorted[right_idx + 1])

        # If next point is at/near baseline (above threshold), we've left the dip
        if next_y >= dip_region_threshold:
            # Include one more point at/beyond baseline for full coverage
            if right_idx < n - 2:
                right_idx += 1  # Step to the baseline point
            else:
                right_idx = n - 1
            break

        right_idx += 1

    return (left_idx, right_idx)


def _best_segment_from_sweep(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float] | None:
    """Return a tight window around the single strongest dip in normalized [0, 1] coordinates.

    Detects the full dip pattern (down then back up) and creates a window spanning
    from where the signal starts descending to where it recovers to baseline.
    """
    if xs.size < 5 or ys.size < 5:
        return None

    min_width_fraction = 0.03  # Very tight: ~3% of domain
    max_width_fraction = 0.50  # Allow wide window to capture full dip extent

    # Sort by x for analysis
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]

    # Estimate baseline from upper quartile
    baseline = float(np.percentile(ys_sorted, 75))

    # Find the deepest point
    min_idx = int(np.argmin(ys_sorted))
    min_signal = float(ys_sorted[min_idx])
    dip_depth = baseline - min_signal

    # Require a meaningful dip (at least 5% below baseline)
    if dip_depth < 0.05:
        return None

    # Detect full dip edges: where it goes down and comes back up
    left_idx, right_idx = _detect_dip_edges(xs_sorted, ys_sorted, min_idx)

    lo_norm = float(xs_sorted[left_idx])
    hi_norm = float(xs_sorted[right_idx])

    # Ensure minimum and maximum width
    width = hi_norm - lo_norm
    if width < min_width_fraction:
        mid = 0.5 * (lo_norm + hi_norm)
        lo_norm = float(np.clip(mid - 0.5 * min_width_fraction, 0.0, 1.0))
        hi_norm = float(np.clip(mid + 0.5 * min_width_fraction, 0.0, 1.0))
    elif width > max_width_fraction:
        # Too wide - shrink to max width centered on minimum point
        mid = float(xs_sorted[min_idx])
        lo_norm = float(np.clip(mid - 0.5 * max_width_fraction, 0.0, 1.0))
        hi_norm = float(np.clip(mid + 0.5 * max_width_fraction, 0.0, 1.0))

    return (lo_norm, hi_norm)


class SequentialBayesianLocator(Locator):
    """Shared Bayesian loop infrastructure for all acquisition strategies.

    Handles the mechanics common to every Bayesian locator:
    - incrementing the step counter
    - convergence-based stopping
    - extracting results from the belief posterior

    After an initial Sobol sweep, a **single** physical interval is derived from the
    sweep data; all Bayesian :meth:`_acquire` calls search only inside that interval
    (same interval shown in the scan UI as the focus band). For unit-cube beliefs,
    the scan parameter's physical bounds (and probe-axis mapping when it matches the
    sweep axis) are updated accordingly so normalization and posteriors stay consistent.

    Subclasses must implement:
    - ``create(**config)`` — build the model-specific ``BeliefSignal`` with priors.
    - ``_acquire()``       — select the next measurement position from the current belief.

    The only behavioral difference between Bayesian strategies is *how* the
    next measurement position is chosen.  All other wiring is identical and
    lives here so it never needs to be repeated.

    Parameters
    ----------
    belief : AbstractMarginalDistribution
        Initial belief (usually a flat / uniform prior over all parameters).
    max_steps : int
        Hard upper bound on Bayesian inference steps (excludes initial sweep).
    convergence_threshold : float
        Posterior-uncertainty threshold below which we consider all parameters
        converged and stop early.
    scan_param : str | None
        The parameter we are proposing measurements along. Defaults to the
        first parameter in the belief.
    initial_sweep_steps : int | None
        Number of initial coarse sweep measurements to take before Bayesian
        acquisition starts. If ``None``, a heuristic based on ``max_steps`` is used.
    initial_sweep_builder : Callable[[int], np.ndarray] | None
        Builder for initial sweep points in normalized ``[0, 1]`` coordinates.
        Defaults to a Sobol-like low-discrepancy 1D sequence.
    """

    DEFAULT_INITIAL_SWEEP_STEPS = 24
    # Minimum number of sweep points that should land inside the expected
    # signal footprint so that :func:`_normalized_acquisition_interval_from_sweep`
    # can reliably identify the focus region.
    _MIN_SIGNAL_HITS = 6

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        max_steps: int = 450,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
    ) -> None:
        super().__init__(belief)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.convergence_threshold = convergence_threshold
        # Total measurement count (includes initial sweep + Bayesian steps).
        self.step_count: int = 0
        # Bayesian acquisition count only (excludes initial sweep).
        self.inference_step_count: int = 0
        self._scan_param = scan_param or belief.model.parameter_names()[0]
        # Default convergence target is all model parameters.
        self._convergence_params: tuple[str, ...] = (
            tuple(convergence_params) if convergence_params is not None else tuple(self.belief.model.parameter_names())
        )
        self._convergence_patience_steps = max(1, int(convergence_patience_steps))
        self._convergence_streak = 0
        # Stored for noise-aware dip detection during sweep refocus.
        self._noise_std: float = float(noise_std) if (noise_std is not None and noise_std > 0) else DEFAULT_MEASUREMENT_NOISE_STD
        # Pre-computed maximum expected noise deviation for mid-sweep threshold.
        # When provided by the executor (via CompositeNoise.estimated_max_noise_deviation),
        # this is used directly as the dip threshold, bypassing the IQR fallback.
        self._noise_max_dev: float | None = float(noise_max_dev) if (noise_max_dev is not None and noise_max_dev > 0) else None
        # Physical max signal span (from signal spec's _signal_max_span bound).
        # Drives both sweep density and refocus window width directly.
        self._signal_max_span: float | None = float(signal_max_span) if (signal_max_span is not None and signal_max_span > 0) else None

        # Set domain bounds BEFORE calculating sweep steps (needed for _model_signal_min_span)
        # For UnitCube beliefs, use physical_param_bounds; for Grid beliefs, use parameter_bounds
        if isinstance(self.belief, (UnitCubeGridMarginalDistribution, UnitCubeSMCMarginalDistribution)):
            self._scan_lo, self._scan_hi = self.belief.physical_param_bounds[self._scan_param]
        else:
            self._scan_lo, self._scan_hi = self.belief.parameter_bounds[self._scan_param]
        # Full scan axis for :class:`~nvision.models.experiment.CoreExperiment` (never narrowed).
        # Belief / :meth:`_normalize` may use a tighter domain after the sweep; returned ``x`` must
        # stay normalized to this full range so ``measure()`` probes the intended frequency.
        self._full_domain_lo, self._full_domain_hi = float(self._scan_lo), float(self._scan_hi)

        if initial_sweep_steps is None:
            # Use signal_min_span for sweep step calculation (need enough steps to catch narrowest signal)
            initial_sweep_steps = self._sweep_steps_for_signal_coverage(
                belief, noise_std=noise_std, signal_min_span=self._model_signal_min_span()
            )

        self.initial_sweep_steps = max(0, int(initial_sweep_steps))
        self._initial_sweep_builder = initial_sweep_builder or sobol_1d_sequence
        if self.initial_sweep_steps > 0:
            self._initial_sweep_points = self._initial_sweep_builder(self.initial_sweep_steps)
        else:
            self._initial_sweep_points = np.empty(0, dtype=float)
        # Post-sweep interval in physical units where _acquire may search; starts at full scan.
        self._acquisition_lo, self._acquisition_hi = self._scan_lo, self._scan_hi
        self._sweep_observations: list[Observation] = []
        # Non-scan parameter bounds narrowed after the sweep (empty = not yet set).
        self._narrowed_param_bounds: dict[str, tuple[float, float]] = {}
        # Staged sweep state: have we performed the mid-sweep refocus?
        self._sweep_refocus_done = False
        # Fallback sweep state: have we done a second sweep with offset when no signal found?
        self._sweep_fallback_done = False
        # Secondary refinement sweep state: triggered after initial sweep when signal found
        self._secondary_sweep_active = False
        self._secondary_sweep_steps_done = 0
        self._secondary_sweep_steps_total = 0
        self._secondary_sweep_points: np.ndarray = np.empty(0, dtype=float)
        self._secondary_sweep_observations: list[Observation] = []
        # Track actual step count when initial sweep completed (including any fallback)
        self._initial_sweep_completed_at_step: int = 0
        # Per-dip focus windows: list of (lo, hi) tuples for individual dip targeting
        self._per_dip_windows: list[tuple[float, float]] | None = None
        # Current dip window index for round-robin acquisition across multiple dips
        self._current_dip_window_idx: int = 0

    def _inner_model(self):
        """Return the inner physical model (unwraps UnitCubeSignalModel if needed)."""
        return getattr(self.belief.model, "inner", self.belief.model)

    def _model_signal_min_span(self) -> float | None:
        """Read signal_min_span from the inner model using the current domain width."""
        domain_width = self._full_domain_hi - self._full_domain_lo
        if domain_width <= 0:
            return None
        m = getattr(self.belief.model, "signal_min_span", None)
        if callable(m):
            return m(domain_width)
        return None

    def _model_signal_max_span(self) -> float | None:
        """Read signal_max_span from the inner model using the current domain width."""
        domain_width = self._full_domain_hi - self._full_domain_lo
        if domain_width <= 0:
            return None
        m = getattr(self.belief.model, "signal_max_span", None)
        if callable(m):
            return m(domain_width)
        return None

    @classmethod
    def _sweep_steps_for_signal_coverage(
        cls,
        belief: AbstractMarginalDistribution,
        *,
        noise_std: float | None = None,
        signal_min_span: float | None = None,
    ) -> int:
        """Derive sweep count from the required spacing to resolve the signal.

        The sweep spacing must be small enough that at least
        :data:`_MIN_SIGNAL_HITS` points land inside the *detectable* width of a
        single dip/peak.  A sweep sample is only a useful "hit" when the signal
        deviation at that point exceeds the measurement-noise amplitude.

        For a Lorentzian dip with half-width ``γ`` and peak depth ``A``, the
        signal at distance ``d`` from centre is ``A · γ²/(d² + γ²)``.  This
        exceeds noise level ``σ`` when::

            d < γ · √(A/σ − 1)          (SNR-aware detectable half-width)

        The required spacing is then::

            spacing = 2 · d_detect / _MIN_SIGNAL_HITS
            n_steps = ⌈domain_width / spacing⌉

        When depth/noise information is unavailable or the representative depth
        sits below the noise floor the method falls back to
        :data:`DEFAULT_INITIAL_SWEEP_STEPS`.
        """
        phys: dict[str, tuple[float, float]] | None = getattr(belief, "physical_param_bounds", None)
        if phys is None:
            return cls.DEFAULT_INITIAL_SWEEP_STEPS

        # Scan domain width in physical units.
        param_names = belief.model.parameter_names()
        scan_name: str = param_names[0] if param_names else ""
        if scan_name not in phys:
            return cls.DEFAULT_INITIAL_SWEEP_STEPS
        scan_lo, scan_hi = phys[scan_name]
        domain_width = float(scan_hi - scan_lo)
        if domain_width <= 0:
            return cls.DEFAULT_INITIAL_SWEEP_STEPS

        # Lower-quartile linewidth estimate (lo^0.75 x hi^0.25):
        # more conservative than the geometric mean so the sweep is dense
        # enough to catch features near the narrow end of the prior.
        linewidth_est: float = 0.0
        for key in ("linewidth", "fwhm_total", "fwhm_lorentz", "fwhm_gauss", "sigma"):
            if key in phys:
                lo, hi = float(phys[key][0]), float(phys[key][1])
                if hi <= 0:
                    continue
                safe_lo = max(lo, 1e-12)
                lq = float(np.exp(0.75 * np.log(safe_lo) + 0.25 * np.log(hi)))
                linewidth_est = max(linewidth_est, lq)

        if linewidth_est <= 0:
            return cls.DEFAULT_INITIAL_SWEEP_STEPS

        # --- SNR-aware detectable width ---
        # A sweep point is a "hit" only when the signal exceeds the noise —
        # anything below the noise floor is surely not signal.
        #
        # For a Lorentzian dip: signal(d) = depth × γ²/(d² + γ²).
        # Detectable when signal(d) > noise_std → d < γ √(depth/noise − 1).
        #
        # We use a conservative (lower-quartile) depth estimate: the prior
        # minimum may sit at the noise floor (undetectable), while the maximum
        # is optimistic. Using dlo^0.75 * dhi^0.25 ensures we don't undersample
        # weak signals near the lower bound of the prior.
        # --- Conservative depth estimate ---
        # Use lower-quartile depth (dlo^0.75 * dhi^0.25) instead of geometric mean.
        # This ensures sweep density is sufficient even when actual signals have
        # depths closer to the lower bound of the prior (common for weak signals).
        depth_est: float = 0.0
        for key in ("dip_depth", "depth", "amplitude"):
            if key in phys:
                dlo, dhi = float(phys[key][0]), float(phys[key][1])
                if dhi > 0:
                    safe_dlo = max(dlo, 1e-12)
                    # Lower-quartile: more conservative than geometric mean
                    lq_depth = float(np.exp(0.75 * np.log(safe_dlo) + 0.25 * np.log(dhi)))
                    depth_est = max(depth_est, lq_depth)

        # Use the actual noise std when provided by the executor; otherwise
        # fall back to the default measurement noise (0.05).
        if noise_std is None or noise_std <= 0:
            noise_std = DEFAULT_MEASUREMENT_NOISE_STD

        # SNR-aware detectable half-width.
        snr_ratio = depth_est / noise_std if (depth_est > 0 and noise_std > 0) else 0.0
        if snr_ratio > 1.0:
            d_detect = linewidth_est * float(np.sqrt(snr_ratio - 1.0))
            feature_width = 2.0 * d_detect
        else:
            # depth ≤ noise: signal is undetectable; use the default sweep.
            return cls.DEFAULT_INITIAL_SWEEP_STEPS

        # If the model declares its minimum physical span, use it directly
        # as the feature width floor — the sweep must be dense enough to catch
        # signals even at their narrowest.  The SNR estimate remains the ceiling
        # (no point sampling finer than the detectable width).
        if signal_min_span is not None and signal_min_span > 0:
            feature_width = min(feature_width, float(signal_min_span)) if feature_width > 0 else float(signal_min_span)

        if feature_width <= 0:
            return cls.DEFAULT_INITIAL_SWEEP_STEPS

        # Required spacing so _MIN_SIGNAL_HITS points fall within one feature.
        # Apply 1.5x safety margin to account for:
        #   - Sobol sequence alignment jitter (points may not align perfectly)
        #   - Boundary effects near domain edges
        #   - Depth variation within the detectable width (signal not flat)
        safety_margin = 1.5
        required_spacing = feature_width / cls._MIN_SIGNAL_HITS
        needed = int(np.ceil(safety_margin * domain_width / required_spacing))

        return int(np.clip(needed, cls.DEFAULT_INITIAL_SWEEP_STEPS, 512))

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        **grid_config: object,
    ) -> SequentialBayesianLocator:
        """Generic factory for model-agnostic Bayesian locators.

        Subclasses for specific models (like NVCenterBayesianLocator) should
        override this to provide their own hard-coded belief setup.
        """
        if builder is None:
            raise ValueError(
                f"{cls.__name__} requires a 'builder' callable to create the AbstractMarginalDistribution."
            )
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            initial_sweep_builder=initial_sweep_builder,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
        )

    @property
    def scan_posterior(self) -> GridParameter:
        """Get the grid posterior for the scan parameter."""
        return self.belief.get_grid_param(self._scan_param)

    # ------------------------------------------------------------------
    # Extension point — the ONLY thing concrete subclasses must add
    # ------------------------------------------------------------------

    @abstractmethod
    def _acquire(self) -> float:
        """Choose the next measurement position from the current belief.

        Called once per step, *after* ``step_count`` has been incremented by
        ``next()``, so ``self.step_count`` is the index of the step being
        planned (1-based).

        Returns
        -------
        float
            Position in **physical units** to measure next. The base class
            will automatically normalize this to [0, 1] for the experiment.
        """

    # ------------------------------------------------------------------
    # Locator interface — implemented once for all Bayesian subclasses
    # ------------------------------------------------------------------

    def next(self) -> float:
        """Propose next measurement with staged initial-sweep warm-start."""
        self.step_count += 1

        if self.step_count == self.initial_sweep_steps + 1:
            self._set_acquisition_window_after_sweep()
            # Record actual step count when initial sweep phase completed (normal case)
            if self._initial_sweep_completed_at_step == 0:
                self._initial_sweep_completed_at_step = self.step_count - 1

        if self.step_count <= self.initial_sweep_steps:
            # Staged sweep: at 50% completion, analyze and potentially refocus
            mid_point = self.initial_sweep_steps // 2
            if (
                not self._sweep_refocus_done
                and mid_point > 0
                and self.step_count > mid_point
                and len(self._sweep_observations) >= mid_point
            ):
                self._sweep_refocus_done = True
                self._maybe_refocus_sweep(mid_point)

            # Early stopping: if signal is clearly detected with enough bracketing,
            # skip remaining initial sweep steps and go straight to window setting
            early_stopping_triggered = (
                self._sweep_refocus_done
                and len(self._sweep_observations) >= 6
                and self._check_initial_sweep_stop()
            )

            u_full = float(self._initial_sweep_points[self.step_count - 1])
            physical_value = self._full_domain_lo + u_full * (self._full_domain_hi - self._full_domain_lo)

            if early_stopping_triggered:
                # Record actual step count for phase tracking, then jump to end
                self._initial_sweep_completed_at_step = self.step_count
                self.step_count = self.initial_sweep_steps  # Jump after this point

            return self._to_experiment_normalized(physical_value)

        # ------------------------------------------------------------------
        # Secondary refinement sweep phase (when signal was found in initial sweep)
        # ------------------------------------------------------------------
        if self._secondary_sweep_active:
            # Check stopping condition first
            if self._check_secondary_sweep_stop():
                self._secondary_sweep_active = False
                self._refine_window_after_secondary_sweep()
                # Fall through to Bayesian
            elif self._secondary_sweep_steps_done < self._secondary_sweep_steps_total:
                self._secondary_sweep_steps_done += 1
                # _secondary_sweep_points are already in full-domain normalized
                # coordinates [0,1] (see _start_secondary_refinement_sweep).
                u_full = float(self._secondary_sweep_points[self._secondary_sweep_steps_done - 1])
                physical_value = self._full_domain_lo + u_full * (self._full_domain_hi - self._full_domain_lo)
                return self._to_experiment_normalized(physical_value)
            else:
                # Secondary sweep complete - refine window before Bayesian
                self._secondary_sweep_active = False
                self._refine_window_after_secondary_sweep()

        self.inference_step_count += 1
        physical_value = self._acquire()
        return self._to_experiment_normalized(physical_value)

    def _narrow_non_scan_params_from_posterior(self) -> None:
        """Periodically tighten non-scan parameter bounds using the current posterior.

        Computes a :data:`_POSTERIOR_CREDIBLE_LEVEL` credible interval from the
        live marginal posterior for every non-scan parameter and calls
        :meth:`narrow_scan_parameter_physical_bounds` to shrink the belief.
        Only narrows by at least :data:`_POSTERIOR_MIN_NARROWING_FRACTION` of the
        current prior width — never widens.
        """
        if not isinstance(self.belief, (UnitCubeGridMarginalDistribution, UnitCubeSMCMarginalDistribution)):
            return
        param_names = list(self.belief.model.parameter_names())
        for param in param_names:
            if param == self._scan_param:
                continue
            if param not in self.belief.physical_param_bounds:
                continue
            interval = _posterior_credible_interval(self.belief, param)
            if interval is None:
                continue
            new_lo, new_hi = interval
            cur_lo, cur_hi = self.belief.physical_param_bounds[param]
            cur_width = cur_hi - cur_lo
            if cur_width <= 0:
                continue
            new_lo = max(new_lo, cur_lo)
            new_hi = min(new_hi, cur_hi)
            if new_hi <= new_lo:
                continue
            if (cur_width - (new_hi - new_lo)) / cur_width < _POSTERIOR_MIN_NARROWING_FRACTION:
                continue
            try:
                self.belief.narrow_scan_parameter_physical_bounds(param, new_lo, new_hi)
            except Exception:  # noqa: BLE001
                pass

    def observe(self, obs: Observation) -> None:
        """Update belief and record sweep observations for the post-sweep window."""
        super().observe(obs)
        if self.step_count <= self.initial_sweep_steps:
            self._sweep_observations.append(obs)
        elif self._secondary_sweep_active:
            self._secondary_sweep_observations.append(obs)

    def done(self) -> bool:
        """Stop when converged (after warm-up) or step budget is exhausted."""
        if self.step_count < self.initial_sweep_steps:
            return False
        if self.inference_step_count >= self.max_steps:
            return True
        if self._target_params_converged():
            self._convergence_streak += 1
        else:
            self._convergence_streak = 0
        return self._convergence_streak >= self._convergence_patience_steps

    def _target_params_converged(self) -> bool:
        """Check convergence on configured target parameters.

        We use normalized uncertainties when available to keep thresholds like
        `0.01` meaningful across differently-scaled physical parameters.
        """
        if not self._convergence_params:
            return self.belief.converged(self.convergence_threshold)

        # Grid-style beliefs expose internal normalized marginals via `parameters`.
        if hasattr(self.belief, "parameters"):
            params = self.belief.parameters
            by_name = {getattr(p, "name", ""): p for p in params}
            if all(name in by_name for name in self._convergence_params):
                return all(
                    float(by_name[name].uncertainty()) < self.convergence_threshold for name in self._convergence_params
                )

        # SMC-style beliefs expose normalized per-dimension std via `_marginal_std`.
        if hasattr(self.belief, "_param_names") and hasattr(self.belief, "_marginal_std"):
            names = list(self.belief._param_names)
            if all(name in names for name in self._convergence_params):
                return all(
                    float(self.belief._marginal_std(names.index(name))) < self.convergence_threshold
                    for name in self._convergence_params
                )

        return self.belief.converged(self.convergence_threshold)

    def result(self) -> dict[str, float]:
        """Return posterior-mean estimates for all parameters."""
        return self.belief.estimates()

    def _acquisition_bounds(self) -> tuple[float, float]:
        """Physical bounds where :meth:`_acquire` searches (post-sweep window).

        When per-dip windows are active (for multi-dip signals like NV center),
        returns windows in round-robin order to focus measurements on each dip.
        """
        lo, hi = self._get_current_acquisition_bounds()
        return (min(lo, hi), max(lo, hi))

    def _set_acquisition_window_after_sweep(self) -> None:
        """Set the acquisition interval from informative sweep samples (frequency axis only).

        Only the scan (frequency) axis is narrowed here.  Non-scan parameters
        are left at their priors and will be tightened later by
        :meth:`_narrow_non_scan_params_from_posterior` once the posterior has
        accumulated enough information.
        """
        xs = np.array([float(o.x) for o in self._sweep_observations], dtype=float)
        ys = np.array([float(o.signal_value) for o in self._sweep_observations], dtype=float)
        span = _normalized_acquisition_interval_from_sweep(xs, ys)
        if span is None:
            # No signal detected in initial sweep. Try a fallback sweep with offset points
            # to avoid the aliasing problem where the Sobol gaps align with equal splits.
            if not self._sweep_fallback_done and self.initial_sweep_steps > 0:
                self._sweep_fallback_done = True
                # Regenerate sweep points with 0.5 offset (half-domain shift)
                self._initial_sweep_points = self._initial_sweep_builder(
                    self.initial_sweep_steps, offset=0.5
                )
                # Clear observations and reset step count to restart the sweep fresh
                self._sweep_observations.clear()
                self.step_count = 0
                self._sweep_refocus_done = False
                # Track total initial sweep steps including fallback for phase coloring
                self._initial_sweep_completed_at_step = self.initial_sweep_steps
                return  # Next call to next() will start the fallback sweep
            # Fallback already done or no sweep steps: use full domain
            self._acquisition_lo, self._acquisition_hi = self._scan_lo, self._scan_hi
            return
        lo_norm, hi_norm = span
        # Convert normalized interval to physical coordinates for acquisition bounds
        self._acquisition_lo = self._full_domain_lo + lo_norm * (self._full_domain_hi - self._full_domain_lo)
        self._acquisition_hi = self._full_domain_lo + hi_norm * (self._full_domain_hi - self._full_domain_lo)

        if isinstance(self.belief, (UnitCubeGridMarginalDistribution, UnitCubeSMCMarginalDistribution)):
            # Use the detected window directly - trust the sweep data over theory.
            # The merged interval from all detected signal segments ensures the
            # secondary sweep covers all dips (important for NV center with two peaks).

            self.belief.narrow_scan_parameter_physical_bounds(
                self._scan_param,
                self._acquisition_lo,
                self._acquisition_hi,
            )
            # Re-read scan axis bounds after narrowing (belief syncs them)
            phys_bounds = self.belief.physical_param_bounds
            slo, shi = phys_bounds[self._scan_param]
            self._acquisition_lo = min(slo, shi)
            self._acquisition_hi = max(slo, shi)

            # ------------------------------------------------------------------
            # Constrain split to fit within the narrowed window.
            # For NV center: dips are at center ± split, so both must satisfy
            #   center - split >= x_lo  and  center + split <= x_hi
            # This requires split <= min(center - x_lo, x_hi - center).
            # The tightest bound is split <= (x_hi - x_lo) / 2.
            # ------------------------------------------------------------------
            if "split" in phys_bounds:
                window_width = self._acquisition_hi - self._acquisition_lo
                if window_width > 0:
                    max_split_for_window = window_width / 2.0
                    split_lo, split_hi = phys_bounds["split"]
                    new_split_hi = min(float(split_hi), max_split_for_window)
                    if new_split_hi > float(split_lo):
                        self.belief.narrow_scan_parameter_physical_bounds(
                            "split",
                            float(split_lo),
                            new_split_hi,
                        )

            # Record all current bounds for the UI / downstream inspection.
            self._narrowed_param_bounds = {
                name: (float(lo), float(hi))
                for name, (lo, hi) in self.belief.physical_param_bounds.items()
            }

        # ------------------------------------------------------------------
        # Secondary refinement sweep: when signal found, do a finer characterization
        # before switching to Bayesian. This gives better initial posterior.
        # ------------------------------------------------------------------
        if span is not None and self.initial_sweep_steps > 0:
            self._start_secondary_refinement_sweep()

    def _start_secondary_refinement_sweep(self) -> None:
        """Start a secondary refinement sweep focused on actual signal region.

        Uses initial sweep observations to find where the signal actually is,
        then generates denser points concentrated in that region.
        Spacing is one "stage" finer than what the initial sweep achieved.
        Eliminates long-tail no-signal regions from the secondary sweep.
        """
        self._secondary_sweep_active = True
        self._secondary_sweep_steps_done = 0
        domain_width = self._full_domain_hi - self._full_domain_lo

        # Find tight signal region from initial sweep observations
        signal_lo_norm, signal_hi_norm = self._signal_region_from_initial_sweep()

        # Calculate actual spacing used in initial sweep within signal region
        # then use one stage finer (half the spacing) for secondary sweep
        signal_width_norm = signal_hi_norm - signal_lo_norm
        initial_sweep_spacing = self._initial_sweep_spacing_in_region(signal_lo_norm, signal_hi_norm)

        # One stage finer = half the spacing of initial sweep
        min_spacing = initial_sweep_spacing / 2.0

        # Also respect model signal span if available
        max_span = self._model_signal_max_span()
        if max_span is not None and domain_width > 0:
            span_norm = max_span / domain_width
            # Don't exceed signal span resolution
            min_spacing = max(min_spacing, span_norm / 16.0)

        min_spacing = max(min_spacing, 0.003)  # Hard floor at 0.3% of domain
        self._secondary_sweep_steps_total = max(10, min(50, int(signal_width_norm / min_spacing) + 4))

        # DEBUG: Log secondary sweep parameters
        import logging
        logging.getLogger("nvision").info(
            f"[SECONDARY SWEEP] signal_region=[{signal_lo_norm:.4f}, {signal_hi_norm:.4f}], "
            f"width={signal_width_norm:.4f}, initial_spacing={initial_sweep_spacing:.4f}, "
            f"min_spacing={min_spacing:.4f}, steps={self._secondary_sweep_steps_total}"
        )

        # Generate points concentrated in signal region
        base_points = self._initial_sweep_builder(self._secondary_sweep_steps_total)
        self._secondary_sweep_points = signal_lo_norm + base_points * signal_width_norm

        # DEBUG: Log generated points
        logging.getLogger("nvision").info(
            f"[SECONDARY SWEEP] base_points (first 5): {base_points[:5]}, "
            f"final_points (first 5): {self._secondary_sweep_points[:5]}"
        )

        self._secondary_sweep_observations.clear()

    def _signal_region_from_initial_sweep(self) -> tuple[float, float]:
        """Extract tight signal region from initial sweep observations.

        Returns normalized [lo, hi] bounds around where signal was actually detected.
        Uses merged interval from all detected segments to cover multi-dip signals.
        Falls back to acquisition window if no clear signal found.
        """
        domain_width = self._full_domain_hi - self._full_domain_lo
        if domain_width <= 0:
            return (0.0, 1.0)

        # Get initial sweep observations
        if len(self._sweep_observations) < 3:
            # Fallback to full acquisition window
            lo_norm = (self._acquisition_lo - self._full_domain_lo) / domain_width
            hi_norm = (self._acquisition_hi - self._full_domain_lo) / domain_width
            return (max(0.0, lo_norm), min(1.0, hi_norm))

        xs = np.array([float(o.x) for o in self._sweep_observations])
        ys = np.array([float(o.signal_value) for o in self._sweep_observations])

        # Observation.x is already in normalized [0,1] coordinates (see CoreExperiment.measure)
        xs_norm = xs

        # Use merged interval from ALL detected segments (not just best one)
        # This ensures secondary sweep covers all dips for multi-peak signals like NV center
        span = _normalized_acquisition_interval_from_sweep(xs_norm, ys)
        if span is not None:
            lo_norm, hi_norm = span
        else:
            # Fallback: use threshold-based detection
            baseline = float(np.percentile(ys, 75))
            threshold = baseline - 0.1 * (baseline - float(np.min(ys)))
            signal_mask = ys < threshold
            if np.any(signal_mask):
                lo_norm = float(np.min(xs_norm[signal_mask]))
                hi_norm = float(np.max(xs_norm[signal_mask]))
            else:
                lo_norm, hi_norm = 0.0, 1.0

        # Minimal padding - keep window tight around actual signal
        width = hi_norm - lo_norm
        padding = 0.02 * width
        lo_norm = max(0.0, lo_norm - padding)
        hi_norm = min(1.0, hi_norm + padding)

        return (lo_norm, hi_norm)

    def _initial_sweep_spacing_in_region(
        self, region_lo_norm: float, region_hi_norm: float
    ) -> float:
        """Calculate minimum spacing of initial sweep points within signal region.

        Returns the characteristic spacing achieved in the initial sweep,
        which is used to make the secondary sweep one "stage" finer.
        """
        if len(self._sweep_observations) < 2:
            # Default: typical Sobol spacing for initial sweep step count
            return 1.0 / max(1, self.initial_sweep_steps)

        # Get points within the signal region (o.x is already normalized [0,1])
        points_in_region = [
            float(o.x) for o in self._sweep_observations
            if region_lo_norm <= float(o.x) <= region_hi_norm
        ]

        if len(points_in_region) < 2:
            # No points in region yet - use full-domain spacing estimate
            xs = np.array([float(o.x) for o in self._sweep_observations])
            xs_sorted = np.sort(xs)
            spacings = np.diff(xs_sorted)
            return float(np.median(spacings)) if len(spacings) > 0 else 1.0 / self.initial_sweep_steps

        # Calculate minimum spacing between consecutive points in region
        points_sorted = np.sort(points_in_region)
        spacings = np.diff(points_sorted)
        min_spacing = float(np.min(spacings)) if len(spacings) > 0 else 1.0 / len(points_in_region)

        # For van der Corput/Sobol sequences, characteristic spacing is ~2x the min
        # (points cluster at certain resolutions)
        return max(min_spacing * 2.0, 1.0 / (2.0 * len(self._sweep_observations)))

    def _check_secondary_sweep_stop(self) -> bool:
        """Check stopping condition for secondary refinement sweep.

        Returns True if sweep should stop early (signal confirmed characterized).
        Stops once we have enough points around the dip to start Bayesian.
        """
        if len(self._secondary_sweep_observations) < 5:
            return False

        xs = np.array([float(o.x) for o in self._secondary_sweep_observations])
        ys = np.array([float(o.signal_value) for o in self._secondary_sweep_observations])
        background_est = float(np.median(np.sort(ys)[int(0.2 * len(ys)):]))
        min_signal = float(np.min(ys))

        # Same threshold logic as mid-sweep refocus
        if self._noise_max_dev is not None:
            dip_threshold = max(self._noise_max_dev, 0.04)
        else:
            iqr = float(np.percentile(ys, 75) - np.percentile(ys, 25))
            data_noise_scale = max(iqr / 1.35, self._noise_std, 1e-6)
            dip_threshold = max(2.5 * data_noise_scale, 0.04)

        # Signal not found - continue sweeping
        if background_est - min_signal < dip_threshold:
            return False

        # Signal confirmed - check if we have enough points around it
        # Stop if we have points on both sides of the minimum
        min_idx = int(np.argmin(ys))
        min_x = float(xs[min_idx])
        window_lo_norm = (self._acquisition_lo - self._full_domain_lo) / (self._full_domain_hi - self._full_domain_lo)
        window_hi_norm = (self._acquisition_hi - self._full_domain_lo) / (self._full_domain_hi - self._full_domain_lo)
        window_width = window_hi_norm - window_lo_norm

        has_left = np.any(xs < min_x - 0.1 * window_width)
        has_right = np.any(xs > min_x + 0.1 * window_width)
        # Stop early if we have points bracketing the dip
        return has_left and has_right

    def _check_initial_sweep_stop(self) -> bool:
        """Check if initial sweep can stop early (signal clearly detected).

        Returns True if we have enough points around the signal to characterize
        it and can skip remaining initial sweep steps.
        """
        if len(self._sweep_observations) < 6:
            return False

        xs = np.array([float(o.x) for o in self._sweep_observations])
        ys = np.array([float(o.signal_value) for o in self._sweep_observations])

        # Check for clear signal dip
        min_idx = int(np.argmin(ys))
        min_signal = float(ys[min_idx])
        baseline = float(np.percentile(ys, 75))

        # Must have significant dip (more than 5% below baseline)
        if baseline - min_signal < 0.05 * baseline:
            return False

        min_x = float(xs[min_idx])

        # Check if we have points bracketing the dip on both sides
        # Use 2% of domain as bracketing threshold
        bracket_threshold = 0.02 * (self._full_domain_hi - self._full_domain_lo)
        has_left = np.any(xs < min_x - bracket_threshold)
        has_right = np.any(xs > min_x + bracket_threshold)

        return has_left and has_right

    def _detect_dip_centers(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> list[tuple[float, float]]:
        """Detect individual dip centers and their depths.

        Returns list of (center_norm, depth_fraction) for each resolved dip.
        Uses contiguous segments below a threshold to identify separate dips.
        """
        if len(xs) < 3 or len(ys) < 3:
            return []

        # Sort by x
        order = np.argsort(xs)
        xs_sorted = xs[order]
        ys_sorted = ys[order]

        # Estimate background and dip depth
        background = float(np.percentile(ys_sorted, 75))
        dip_depth = background - float(np.min(ys_sorted))

        if dip_depth <= 0:
            return []

        # Threshold at 20% of dip depth - points below this are "in a dip"
        threshold = background - 0.2 * dip_depth
        below = ys_sorted < threshold

        if not np.any(below):
            return []

        # Find contiguous segments (individual dips)
        changes = np.diff(below.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if below[0]:
            starts = np.concatenate([[0], starts])
        if below[-1]:
            ends = np.concatenate([ends, [len(below)]])

        dips: list[tuple[float, float]] = []
        for s, e in zip(starts, ends, strict=False):
            seg_ys = ys_sorted[s:e]
            seg_xs = xs_sorted[s:e]
            min_idx = int(np.argmin(seg_ys))
            center_norm = float(seg_xs[min_idx])
            depth_frac = (background - float(seg_ys[min_idx])) / max(background, 1e-12)
            dips.append((center_norm, depth_frac))

        return dips

    def _per_dip_focus_windows(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> list[tuple[float, float]] | None:
        """Compute individual focus windows around each detected dip.

        For NV center with large splitting, creates separate tight windows
        around left, center, and right dips instead of one large window covering
        the entire span. This is more efficient for Bayesian acquisition.

        Returns list of (lo_norm, hi_norm) tuples, one per dip, or None if
        dips cannot be resolved separately.
        """
        dips = self._detect_dip_centers(xs, ys)

        if len(dips) < 2:
            # Single dip or none - use single window
            return None

        expected_dips = self._expected_dip_count_from_model()

        # For symmetry: group left/right dips if they have similar depth
        # This handles the asymmetric NV center case (left shallow, right deep)
        if expected_dips == 3 and len(dips) >= 2:
            # Check if first and last dips have depths in expected ratio
            # For k_np ~ 2-4, left should be ~1/k_np of right
            left_depth = dips[0][1]
            right_depth = dips[-1][1]
            depth_ratio = left_depth / max(right_depth, 1e-12)

            # k_np range is typically 2-4, so left/right ratio should be 0.25-0.5
            if 0.15 <= depth_ratio <= 0.6 and len(dips) == 3:
                # Asymmetric triplet: create windows around each dip
                # Group outer dips together (they're symmetric in position but not amplitude)
                left_center, _ = dips[0]
                right_center, _ = dips[-1]
                center_center, _ = dips[1]

                # Window covering both outer dips
                outer_span = right_center - left_center
                outer_lo = max(0.0, left_center - 0.12 * outer_span)
                outer_hi = min(1.0, right_center + 0.12 * outer_span)

                # Tight window around center dip
                min_span_norm = 0.025  # 2.5% of domain minimum
                center_lo = max(0.0, center_center - min_span_norm)
                center_hi = min(1.0, center_center + min_span_norm)

                return [(outer_lo, outer_hi), (center_lo, center_hi)]

        # Default: create individual window around each dip
        windows: list[tuple[float, float]] = []
        min_span_norm = 0.03  # 3% of domain

        for center_norm, _ in dips:
            lo = max(0.0, center_norm - min_span_norm)
            hi = min(1.0, center_norm + min_span_norm)
            windows.append((lo, hi))

        return windows if len(windows) >= 2 else None

    def _get_current_acquisition_bounds(self) -> tuple[float, float]:
        """Return the current acquisition window bounds.

        If per-dip windows are active, returns the current dip's window
        and advances to the next dip for subsequent calls (round-robin).
        Otherwise returns the single acquisition window.
        """
        if self._per_dip_windows is not None and len(self._per_dip_windows) > 0:
            # Round-robin through per-dip windows
            window = self._per_dip_windows[self._current_dip_window_idx]
            self._current_dip_window_idx = (self._current_dip_window_idx + 1) % len(self._per_dip_windows)
            return window
        return (self._acquisition_lo, self._acquisition_hi)

    def _expected_dip_count_from_model(self) -> int:
        """Return expected number of dips from the signal model.

        Delegates to the model's expected_dip_count() method which knows
        its own structure (1, 2, or 3 dips based on physics).
        """
        inner = self._inner_model()
        return inner.expected_dip_count()

    def _refine_window_after_secondary_sweep(self) -> None:
        """Tighten acquisition window using precise secondary sweep data.

        After the denser secondary sweep completes, use the accumulated
        observations to compute a tighter focus window around the actual
        peak/dip locations. This enhanced version uses knowledge of the
        expected signal shape (1 vs 2 dips from model split parameter) to
        apply more precise window narrowing.
        """
        if len(self._secondary_sweep_observations) < 5:
            return  # Not enough data to refine

        xs = np.array([float(o.x) for o in self._secondary_sweep_observations], dtype=float)
        ys = np.array([float(o.signal_value) for o in self._secondary_sweep_observations], dtype=float)

        expected_dips = self._expected_dip_count_from_model()
        domain_width = self._full_domain_hi - self._full_domain_lo

        # Choose windowing strategy based on expected signal shape
        if expected_dips == 1:
            # Single dip expected: use tight window around strongest dip
            span = _best_segment_from_sweep(xs, ys)
            if span is None:
                return
            lo_norm, hi_norm = span
        elif expected_dips == 2:
            # Two dips expected: use full interval but shrink padding
            span = _normalized_acquisition_interval_from_sweep(xs, ys)
            if span is None:
                return
            lo_norm, hi_norm = span
            # Reduce padding since we know the structure better
            current_width = hi_norm - lo_norm
            reduced_padding = 0.03  # 3% instead of default 10%
            lo_norm = max(0.0, lo_norm - reduced_padding * current_width)
            hi_norm = min(1.0, hi_norm + reduced_padding * current_width)
        elif expected_dips == 3:
            # Three dips expected (triplet): use per-dip focus windows for efficiency
            # Try to detect individual dips and create focused windows
            per_dip_windows = self._per_dip_focus_windows(xs, ys)

            if per_dip_windows is not None and len(per_dip_windows) >= 2:
                # Store per-dip windows for round-robin acquisition
                self._per_dip_windows = [
                    (self._full_domain_lo + lo * domain_width,
                     self._full_domain_lo + hi * domain_width)
                    for lo, hi in per_dip_windows
                ]
                # Use the full span of all windows as the main window
                all_lo = [w[0] for w in self._per_dip_windows]
                all_hi = [w[1] for w in self._per_dip_windows]
                new_lo = min(all_lo)
                new_hi = max(all_hi)
                # Update acquisition bounds to full span
                self._acquisition_lo = max(self._full_domain_lo, min(new_lo, new_hi))
                self._acquisition_hi = min(self._full_domain_hi, max(new_lo, new_hi))
                return  # Skip the rest - per-dip windows are set

            # Fallback: use standard detection but ensure minimum width for outer peaks
            span = _normalized_acquisition_interval_from_sweep(xs, ys)
            if span is None:
                return
            lo_norm, hi_norm = span
            # Ensure minimum width to capture outer dips of triplet
            min_width = 0.10  # At least 10% of domain for triplet
            current_width = hi_norm - lo_norm
            if current_width < min_width:
                mid = 0.5 * (lo_norm + hi_norm)
                lo_norm = max(0.0, mid - 0.5 * min_width)
                hi_norm = min(1.0, mid + 0.5 * min_width)
        else:
            # Fallback: use standard detection
            span = _normalized_acquisition_interval_from_sweep(xs, ys)
            if span is None:
                return
            lo_norm, hi_norm = span

        # Convert to physical coordinates
        new_lo = self._full_domain_lo + lo_norm * domain_width
        new_hi = self._full_domain_lo + hi_norm * domain_width

        # Apply additional tightening based on signal span knowledge
        max_span = self._model_signal_max_span()
        if max_span is not None and max_span > 0 and domain_width > 0:
            # Ensure window doesn't exceed expected signal span + small margin
            max_window_phys = max_span * 1.2  # 20% margin
            current_window_phys = abs(new_hi - new_lo)
            if current_window_phys > max_window_phys:
                # Shrink to max expected span centered on detected signal
                mid = 0.5 * (new_lo + new_hi)
                half_window = max_window_phys / 2
                new_lo = max(self._full_domain_lo, mid - half_window)
                new_hi = min(self._full_domain_hi, mid + half_window)

        # Constrain to full domain
        self._acquisition_lo = max(self._full_domain_lo, min(new_lo, new_hi))
        self._acquisition_hi = min(self._full_domain_hi, max(new_lo, new_hi))

        # Update belief with refined bounds
        if isinstance(self.belief, (UnitCubeGridMarginalDistribution, UnitCubeSMCMarginalDistribution)):
            self.belief.narrow_scan_parameter_physical_bounds(
                self._scan_param,
                self._acquisition_lo,
                self._acquisition_hi,
            )
            # Sync back from belief
            slo, shi = self.belief.physical_param_bounds[self._scan_param]
            self._acquisition_lo = min(slo, shi)
            self._acquisition_hi = max(slo, shi)

    def _maybe_refocus_sweep(self, mid_point: int) -> None:
        """Mid-sweep analysis: refocus if signal found, otherwise continue.

        At 50% of the sweep, analyze observations so far. If a signal dip is
        detected, narrow the remaining sweep to focus on that region. The anchor
        point is the global minimum signal value — the point most likely to be
        on the dip itself — rather than a local dip score estimate.
        """
        if len(self._sweep_observations) < 5:
            return  # Not enough data yet

        xs = np.array([float(o.x) for o in self._sweep_observations], dtype=float)
        ys = np.array([float(o.signal_value) for o in self._sweep_observations], dtype=float)

        # The global minimum signal value point IS on or near the dip by definition.
        # This is more reliable than local dip scoring with sparse samples.
        min_idx = int(np.argmin(ys))
        best_point_norm = float(xs[min_idx])
        min_signal = float(ys[min_idx])

        # Only proceed if the minimum is clearly below background.
        #
        # Threshold = expected maximum downward noise deviation for this sweep size.
        # When the executor provides noise_max_dev (from each noise model's own
        # EVT formula), we use it directly — it is exact for that noise distribution.
        # Fallback: IQR-based estimate from the sweep data itself, which adapts
        # to whatever noise is actually present.
        background_est = float(np.median(np.sort(ys)[int(0.2 * len(ys)):]))
        if self._noise_max_dev is not None:
            dip_threshold = max(self._noise_max_dev, 0.04)
        else:
            iqr = float(np.percentile(ys, 75) - np.percentile(ys, 25))
            data_noise_scale = max(iqr / 1.35, self._noise_std, 1e-6)
            dip_threshold = max(2.5 * data_noise_scale, 0.04)
        if background_est - min_signal < dip_threshold:
            return

        # Refocus window half-width: use the signal model's declared max span when
        # available. The model's signal_max_span() method is the authoritative value
        # — it already accounts for split, linewidth, and any multi-dip structure.
        # Fall back to deriving the span from individual parameter bounds when the
        # model does not provide it.
        domain_width = self._full_domain_hi - self._full_domain_lo
        max_span = self._model_signal_max_span()
        if max_span is not None and domain_width > 0:
            half_w = float(np.clip(max_span / 2.0 / domain_width, 0.05, 0.40))
        else:
            half_span_phys = 0.0
            phys_bounds: dict[str, tuple[float, float]] = getattr(
                self.belief, "physical_param_bounds", {}
            )
            for key in ("split",):
                if key in phys_bounds:
                    _, split_hi = phys_bounds[key]
                    half_span_phys = max(half_span_phys, 2.0 * float(split_hi))
            for key in ("linewidth", "fwhm_total", "fwhm_lorentz", "fwhm_gauss", "sigma"):
                if key in phys_bounds:
                    _, lw_hi = phys_bounds[key]
                    half_span_phys += float(lw_hi)
                    break  # one linewidth contribution is enough
            if domain_width > 0 and half_span_phys > 0:
                half_w = float(np.clip(half_span_phys / domain_width, 0.05, 0.40))
            else:
                half_w = 0.10  # safe default

        # Build window centered on the detected dip, sized by signal span.
        # When near domain boundaries, shrink half_w proportionally to keep
        # the window centered on the dip rather than creating an asymmetric window.
        left_space = best_point_norm  # distance to 0 boundary
        right_space = 1.0 - best_point_norm  # distance to 1 boundary
        half_w = min(half_w, left_space, right_space)
        lo_norm = best_point_norm - half_w
        hi_norm = best_point_norm + half_w

        # Convert normalized to physical for acquisition bounds
        lo_phys = self._full_domain_lo + lo_norm * (self._full_domain_hi - self._full_domain_lo)
        hi_phys = self._full_domain_lo + hi_norm * (self._full_domain_hi - self._full_domain_lo)
        self._acquisition_lo, self._acquisition_hi = lo_phys, hi_phys

        # Regenerate remaining sweep points - first point is the detected dip
        remaining = self.initial_sweep_steps - mid_point
        if remaining > 0:
            new_points: list[float] = []
            window_width = hi_norm - lo_norm

            # First point: directly at the detected dip (guaranteed on signal)
            new_points.append(best_point_norm)

            # Remaining points: Sobol in LOCAL [0,1] window space → convert to global
            if remaining > 1:
                sobol_pts = sobol_1d_sequence(remaining * 2)
                # dip location expressed in local window fraction [0,1]
                dip_local = (best_point_norm - lo_norm) / window_width if window_width > 0 else 0.5
                min_sep_local = 1.0 / (remaining * 2)
                for p in sobol_pts:
                    if abs(p - dip_local) > min_sep_local:
                        new_points.append(float(np.clip(lo_norm + p * window_width, lo_norm, hi_norm)))
                    if len(new_points) >= remaining:
                        break

            # Fill any shortfall with evenly spaced points in the window
            while len(new_points) < remaining:
                frac = len(new_points) / max(1, remaining - 1)
                new_points.append(float(np.clip(lo_norm + frac * window_width, lo_norm, hi_norm)))

            # Store back into sweep schedule
            for i in range(remaining):
                idx = mid_point + i
                if idx < len(self._initial_sweep_points):
                    self._initial_sweep_points[idx] = new_points[i]

    def secondary_sweep_count(self) -> int:
        """Number of secondary refinement sweep steps planned (for UI phase coloring)."""
        return self._secondary_sweep_steps_total

    def effective_initial_sweep_steps(self) -> int:
        """Effective initial sweep step count including any fallback sweep."""
        # If fallback occurred, _initial_sweep_completed_at_step tracks the original
        # sweep steps before reset. Total coarse steps = original + fallback.
        if self._initial_sweep_completed_at_step > 0 and self._sweep_fallback_done:
            return self._initial_sweep_completed_at_step + self.initial_sweep_steps
        # Normal case: initial sweep completed without fallback
        if self._initial_sweep_completed_at_step > 0:
            return self._initial_sweep_completed_at_step
        # Still in initial sweep - return current count
        return self.step_count

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        """Same physical interval as :meth:`_acquisition_bounds` for scan / manifest UI."""
        if self.initial_sweep_steps <= 0:
            return None
        lo, hi = self._acquisition_bounds()
        slo, shi = self._full_domain_lo, self._full_domain_hi
        if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(slo) and np.isfinite(shi)):
            return None
        if hi <= lo or shi <= slo:
            return None
        span = shi - slo
        if span <= 0:
            return None
        if (hi - lo) >= span * (1.0 - 1e-9):
            return None
        return (lo, hi)

    def narrowed_param_bounds(self) -> dict[str, tuple[float, float]]:
        """Physical bounds of non-scan parameters narrowed after the initial sweep.

        Returns an empty dict when no sweep has been completed or no parameters
        could be narrowed.

        Returns
        -------
        dict[str, tuple[float, float]]
            Mapping of parameter name → ``(lo, hi)`` in physical units after
            narrowing.  Only parameters that were genuinely tightened are included.
        """
        return dict(self._narrowed_param_bounds)

    # ------------------------------------------------------------------
    # Utility helpers available to all acquisition implementations
    # ------------------------------------------------------------------

    def _generate_candidates(self, num_candidates: int = 300) -> np.ndarray:
        """Generate grid from acquisition bounds (log-uniform for scale params)."""
        lo, hi = self._acquisition_bounds()
        is_scale = getattr(self.belief.model, "is_scale_parameter", lambda name: False)(self._scan_param)
        if is_scale and lo > 0 and hi > lo:
            return np.exp(np.linspace(np.log(lo), np.log(hi), num_candidates))
        return np.linspace(lo, hi, num_candidates)

    def _apply_parameter_weight_bias(
        self,
        utilities: np.ndarray,
        mu_preds: np.ndarray,
        sampled: object,
    ) -> np.ndarray:
        """Boost utilities toward measurements that are most informative about high-weight parameters.

        For each candidate ``x_i`` the **frequency-specificity** is the squared
        correlation between the signal predictions ``S(x_i, θ)`` and the
        frequency dimension of the posterior samples::

            R²_f(x_i) = Cov(S(x_i,θ), θ_f)² / [Var(S(x_i,θ)) · Var(θ_f)]

        This is in ``[0, 1]``.  A measurement where predictions are perfectly
        correlated with frequency uncertainty gets its utility multiplied by
        ``freq_weight``; a measurement uncorrelated with frequency is unchanged.

        Only parameters with weight > 1 contribute (default weight = 1 → no-op).

        Parameters
        ----------
        utilities : np.ndarray
            Shape ``(n_candidates,)`` — base acquisition utilities.
        mu_preds : np.ndarray
            Shape ``(n_candidates, n_samples)`` — signal predictions over posterior samples.
        sampled : ParameterValues[np.ndarray]
            Posterior parameter samples (unit-cube or physical; scale does not matter).

        Returns
        -------
        np.ndarray
            Biased utilities, same shape as input.
        """
        inner_model = getattr(self.belief.model, "inner", self.belief.model)
        if not hasattr(inner_model, "parameter_weights"):
            return utilities
        weights = inner_model.parameter_weights()

        result = utilities.copy()
        pred_var = np.var(mu_preds, axis=1)  # (n_candidates,)

        for param_name, weight in weights.items():
            if weight <= 1.0:
                continue
            try:
                param_particles = np.asarray(sampled[param_name], dtype=np.float64)
            except (KeyError, TypeError):
                continue
            p_var = float(np.var(param_particles))
            if p_var < 1e-30:
                continue

            p_mean = param_particles.mean()
            p_dev = param_particles - p_mean
            mu_mean = mu_preds.mean(axis=1)
            cov = ((mu_preds - mu_mean[:, None]) * p_dev[None, :]).mean(axis=1)
            r2 = cov**2 / (pred_var * p_var + 1e-30)
            r2 = np.clip(r2, 0.0, 1.0)

            result = result * (1.0 + (weight - 1.0) * r2)

        return result

    def _to_experiment_normalized(self, physical_value: float) -> float:
        """Map a physical scan position to ``[0, 1]`` for :meth:`CoreExperiment.measure`."""
        lo, hi = self._full_domain_lo, self._full_domain_hi
        return float(np.clip((physical_value - lo) / (hi - lo), 0.0, 1.0))

    def _normalize(self, param_name: str, physical_value: float) -> float:
        """Convert a physical parameter value to the normalised [0, 1] range.

        Uses the bounds stored in the belief's corresponding
        ``ParameterWithPosterior`` so subclasses never need to hard-code
        domain limits.

        Parameters
        ----------
        param_name : str
            Name of the parameter whose bounds define the normalisation.
        physical_value : float
            Value in physical units to convert.

        Returns
        -------
        float
            Clipped normalised value in [0, 1].
        """
        lo, hi = self.belief.parameter_bounds[param_name]
        return float(np.clip((physical_value - lo) / (hi - lo), 0.0, 1.0))

    def _denormalize(self, param_name: str, normalized_value: float) -> float:
        """Convert a normalised [0, 1] value back to physical units.

        Parameters
        ----------
        param_name : str
            Name of the parameter whose bounds define the mapping.
        normalized_value : float
            Value in [0, 1] to convert.

        Returns
        -------
        float
            Value in physical units.
        """
        lo, hi = self.belief.parameter_bounds[param_name]
        return float(lo + normalized_value * (hi - lo))
