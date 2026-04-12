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
    """Return ``(lo, hi)`` in normalized [0, 1] scan coordinates, or ``None`` to use the full domain."""
    if xs.size < 3 or ys.size < 3:
        return None

    info_quantile = 0.8
    padding_fraction = 0.1
    min_width_fraction = 0.15
    segment_peak_ratio = 0.6
    merge_gap_factor = 2.0

    center = float(np.median(ys))
    info = np.abs(ys - center)
    thr = float(np.quantile(info, info_quantile))
    keep = info >= thr
    if not np.any(keep):
        return None

    order = np.argsort(xs)
    xs_sorted = xs[order]
    info_sorted = info[order]
    keep_sorted = keep[order]
    x_keep = xs_sorted[keep_sorted]
    info_keep = info_sorted[keep_sorted]
    if x_keep.size == 0:
        return None

    diffs_all = np.diff(xs_sorted)
    positive_diffs = diffs_all[diffs_all > 0]
    median_dx = float(np.median(positive_diffs)) if positive_diffs.size else 0.0
    split_gap = max(3.0 * median_dx, 1e-6)
    seg_breaks = np.where(np.diff(x_keep) > split_gap)[0] + 1
    seg_xs = np.split(x_keep, seg_breaks)
    seg_infos = np.split(info_keep, seg_breaks)
    seg_peaks = np.array([float(np.max(s)) for s in seg_infos], dtype=float)
    if seg_peaks.size == 0:
        return None
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

    DEFAULT_INITIAL_SWEEP_STEPS = 32
    # Minimum number of sweep points that should land inside the expected
    # signal footprint so that :func:`_normalized_acquisition_interval_from_sweep`
    # can reliably identify the focus region.
    _MIN_SIGNAL_HITS = 10

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
        self._scan_param = scan_param or belief.parameters[0].name
        # Default convergence target is all model parameters.
        self._convergence_params: tuple[str, ...] = (
            tuple(convergence_params) if convergence_params is not None else tuple(self.belief.model.parameter_names())
        )
        self._convergence_patience_steps = max(1, int(convergence_patience_steps))
        self._convergence_streak = 0

        if initial_sweep_steps is None:
            initial_sweep_steps = self._sweep_steps_for_signal_coverage(belief, noise_std=noise_std)

        self.initial_sweep_steps = max(0, int(initial_sweep_steps))
        self._initial_sweep_builder = initial_sweep_builder or sobol_1d_sequence
        if self.initial_sweep_steps > 0:
            self._initial_sweep_points = self._initial_sweep_builder(self.initial_sweep_steps)
        else:
            self._initial_sweep_points = np.empty(0, dtype=float)

        self._scan_lo, self._scan_hi = self.belief.parameter_bounds[self._scan_param]
        # Full scan axis for :class:`~nvision.models.experiment.CoreExperiment` (never narrowed).
        # Belief / :meth:`_normalize` may use a tighter domain after the sweep; returned ``x`` must
        # stay normalized to this full range so ``measure()`` probes the intended frequency.
        self._full_domain_lo, self._full_domain_hi = float(self._scan_lo), float(self._scan_hi)
        # Post-sweep interval in physical units where _acquire may search; starts at full scan.
        self._acquisition_lo, self._acquisition_hi = self._scan_lo, self._scan_hi
        self._sweep_observations: list[Observation] = []
        # Non-scan parameter bounds narrowed after the sweep (empty = not yet set).
        self._narrowed_param_bounds: dict[str, tuple[float, float]] = {}

    @classmethod
    def _sweep_steps_for_signal_coverage(
        cls, belief: AbstractMarginalDistribution, *, noise_std: float | None = None
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
        scan_name: str = belief.parameters[0].name if belief.parameters else ""
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
        for key in ("linewidth", "fwhm_lorentz", "fwhm_gauss", "sigma"):
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
        # We use the geometric mean of the depth bounds as a representative
        # amplitude: the prior minimum may sit at the noise floor (undetectable),
        # while the maximum is optimistic.
        depth_est: float = 0.0
        for key in ("dip_depth", "depth", "amplitude"):
            if key in phys:
                dlo, dhi = float(phys[key][0]), float(phys[key][1])
                if dhi > 0:
                    depth_est = max(depth_est, float(np.sqrt(max(dlo, 1e-12) * dhi)))

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

        if feature_width <= 0:
            return cls.DEFAULT_INITIAL_SWEEP_STEPS

        # Required spacing so _MIN_SIGNAL_HITS points fall within one feature.
        required_spacing = feature_width / cls._MIN_SIGNAL_HITS
        needed = int(np.ceil(domain_width / required_spacing))

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
        """Propose next measurement with initial-sweep warm-start."""
        self.step_count += 1

        if self.step_count == self.initial_sweep_steps + 1:
            self._set_acquisition_window_after_sweep()

        if self.step_count <= self.initial_sweep_steps:
            u = float(self._initial_sweep_points[self.step_count - 1])
            is_scale = getattr(self.belief.model, "is_scale_parameter", lambda name: False)(self._scan_param)
            lo, hi = self._full_domain_lo, self._full_domain_hi
            if is_scale and lo > 0 and hi > lo:
                physical_value = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
            else:
                physical_value = lo + u * (hi - lo)
            return self._to_experiment_normalized(physical_value)

        self.inference_step_count += 1
        if (
            self.inference_step_count > 0
            and self.inference_step_count % _POSTERIOR_NARROWING_INTERVAL == 0
        ):
            self._narrow_non_scan_params_from_posterior()
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
                # Update narrowed_param_bounds so the UI reflects the live acquisition
                # window as it shrinks during inference. Only include the scan parameter
                # to keep the posterior plots clean.
                if param == self._scan_param:
                    actual_lo, actual_hi = self.belief.physical_param_bounds[param]
                    self._narrowed_param_bounds[param] = (
                        min(actual_lo, actual_hi),
                        max(actual_lo, actual_hi),
                    )
            except Exception:  # noqa: BLE001
                pass

    def narrowed_param_bounds(self) -> dict[str, tuple[float, float]] | None:
        """Return current narrowed parameter bounds for UI updates."""
        return getattr(self, "_narrowed_param_bounds", None)

    def observe(self, obs: Observation) -> None:
        """Update belief and record sweep observations for the post-sweep window."""
        super().observe(obs)
        if self.step_count <= self.initial_sweep_steps:
            self._sweep_observations.append(obs)

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
        """Physical bounds where :meth:`_acquire` searches (post-sweep window)."""
        lo, hi = float(self._acquisition_lo), float(self._acquisition_hi)
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
            self._acquisition_lo, self._acquisition_hi = self._scan_lo, self._scan_hi
            return
        lo_norm, hi_norm = span
        self._acquisition_lo = self._denormalize(self._scan_param, lo_norm)
        self._acquisition_hi = self._denormalize(self._scan_param, hi_norm)

        if isinstance(self.belief, (UnitCubeGridMarginalDistribution, UnitCubeSMCMarginalDistribution)):
            # ------------------------------------------------------------------
            # Split-based frequency window constraint.
            # The two NV dips sit at frequency ± split, so the scan window only
            # needs to reach 2×split_max away from the centre in each direction
            # (4×split_max total).  Apply this before narrowing the belief so
            # both constraints are intersected.
            # ------------------------------------------------------------------
            phys_bounds = self.belief.physical_param_bounds
            if "split" in phys_bounds and self._scan_param in phys_bounds:
                _, split_hi = phys_bounds["split"]
                if split_hi > 0:
                    max_half_span = 2.0 * split_hi
                    center = 0.5 * (self._acquisition_lo + self._acquisition_hi)
                    split_lo_cand = center - max_half_span
                    split_hi_cand = center + max_half_span
                    # Only narrow — never widen the current acquisition window.
                    self._acquisition_lo = max(self._acquisition_lo, split_lo_cand)
                    self._acquisition_hi = min(self._acquisition_hi, split_hi_cand)
                    # Clamp to the full scan domain.
                    self._acquisition_lo = max(self._acquisition_lo, self._full_domain_lo)
                    self._acquisition_hi = min(self._acquisition_hi, self._full_domain_hi)

            self.belief.narrow_scan_parameter_physical_bounds(
                self._scan_param,
                self._acquisition_lo,
                self._acquisition_hi,
            )
            # Re-read scan axis bounds after narrowing (belief syncs them)
            slo, shi = self.belief.physical_param_bounds[self._scan_param]
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
