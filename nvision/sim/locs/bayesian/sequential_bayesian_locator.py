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
from nvision.sim.locs.coarse.sobol_locator import sobol_1d_sequence, StagedSobolLocator


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

        # Create staged Sobol locator for initial sweep phase
        self._staged_sobol = StagedSobolLocator(
            belief=self.belief,
            max_steps=self.initial_sweep_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            scan_param=self._scan_param,
            domain_lo=self._full_domain_lo,
            domain_hi=self._full_domain_hi,
        )

        # Post-sweep interval in physical units where _acquire may search; starts at full scan.
        self._acquisition_lo, self._acquisition_hi = self._scan_lo, self._scan_hi
        # Non-scan parameter bounds narrowed after the sweep (empty = not yet set).
        self._narrowed_param_bounds: dict[str, tuple[float, float]] = {}
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
        safety_margin = 2.5
        required_spacing = feature_width / cls._MIN_SIGNAL_HITS
        needed = int(np.ceil(safety_margin * domain_width / required_spacing))

        return int(np.clip(needed, cls.DEFAULT_INITIAL_SWEEP_STEPS, 1023))

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

        # Phase 1: Use staged Sobol locator for initial sweep
        if not self._staged_sobol.done():
            value = self._staged_sobol.next()
            # Track if this is the last step of initial sweep
            if self._staged_sobol.done() and self._initial_sweep_completed_at_step == 0:
                self._initial_sweep_completed_at_step = self.step_count
                self._set_acquisition_window_from_staged_sobol()
            return self._to_experiment_normalized(value)

        # Phase 2: Bayesian acquisition
        if self.step_count == self.initial_sweep_steps + 1:
            # First step after initial sweep - ensure window is set
            if self._initial_sweep_completed_at_step == 0:
                self._initial_sweep_completed_at_step = self.step_count - 1
                self._set_acquisition_window_from_staged_sobol()

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
        if not self._staged_sobol.done():
            self._staged_sobol.observe(obs)

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

    def _set_acquisition_window_from_staged_sobol(self) -> None:
        """Set acquisition window from staged Sobol locator results.

        Copies the window bounds from the completed staged Sobol locator and
        narrows the belief's scan parameter bounds accordingly.
        """
        # Get window from staged Sobol locator
        self._acquisition_lo, self._acquisition_hi = self._staged_sobol.acquisition_window()

        if isinstance(self.belief, (UnitCubeGridMarginalDistribution, UnitCubeSMCMarginalDistribution)):
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

            # Constrain split to fit within narrowed window
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

    def effective_initial_sweep_steps(self) -> int:
        """Effective initial sweep step count including any fallback sweep."""
        # Return staged Sobol locator's effective step count
        if hasattr(self, '_staged_sobol'):
            return self._staged_sobol.effective_step_count()
        # Fallback if called before _staged_sobol is initialized
        if self._initial_sweep_completed_at_step > 0:
            return self._initial_sweep_completed_at_step
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

    def per_dip_windows(self) -> list[tuple[float, float]] | None:
        """Individual per-dip focus windows for multi-dip signals (e.g., NV center triplets).

        Returns a list of (lo, hi) tuples when per-dip targeting is active,
        or None when using a single acquisition window.
        """
        return self._per_dip_windows

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
        candidates: np.ndarray | None = None,
    ) -> np.ndarray:
        """Boost utilities toward measurements that are most informative about high-weight parameters.

        For each candidate ``x_i`` the **frequency-specificity** is the squared
        correlation between the signal predictions ``S(x_i, θ)`` and the
        frequency dimension of the posterior samples::

            R²_f(x_i) = Cov(S(x_i,θ), θ_f)² / [Var(S(x_i,θ)) · Var(θ_f)]

        This is in ``[0, 1]``.  A measurement where predictions are perfectly
        correlated with frequency uncertainty gets its utility multiplied by
        ``freq_weight``; a measurement uncorrelated with frequency is unchanged.

        Additionally, when candidates are provided, a **center-frequency proximity**
        bonus is applied: measurements near the posterior mean frequency receive
        higher weight, with a Gaussian falloff::

            center_boost(x_i) = 1.0 + center_freq_weight * exp(-0.5 * ((x_i - f_mean) / f_std)²)

        Only parameters with weight > 1 contribute (default weight = 1 → no-op).

        Parameters
        ----------
        utilities : np.ndarray
            Shape ``(n_candidates,)`` — base acquisition utilities.
        mu_preds : np.ndarray
            Shape ``(n_candidates, n_samples)`` — signal predictions over posterior samples.
        sampled : ParameterValues[np.ndarray]
            Posterior parameter samples (unit-cube or physical; scale does not matter).
        candidates : np.ndarray | None
            Shape ``(n_candidates,)`` — candidate positions in physical units.
            Required for center-frequency proximity weighting.

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

        # Center-frequency proximity weighting: boost utilities near posterior mean frequency
        if candidates is not None and "frequency" in weights:
            freq_weight = weights["frequency"]
            try:
                freq_particles = np.asarray(sampled["frequency"], dtype=np.float64)
                f_mean = float(np.mean(freq_particles))
                f_std = float(np.std(freq_particles))
                if f_std > 1e-12:
                    # Gaussian proximity factor: 1.0 at center, decays with distance
                    z = (candidates - f_mean) / f_std
                    proximity = np.exp(-0.5 * z * z)
                    # Boost: up to (freq_weight - 1.0) additional weight at center
                    center_boost = 1.0 + (freq_weight - 1.0) * proximity
                    result = result * center_boost
            except (KeyError, TypeError):
                pass

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
