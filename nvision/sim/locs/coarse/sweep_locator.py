"""Base class for sweeping locators with signal detection and windowing."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.locator import Locator
from nvision.models.observation import Observation, ObservationHistory
from nvision.sim.locs.refocus import infer_focus_window_physical as _refocus_infer_focus_window
from nvision.sim.locs.refocus.strategies import detect_dips as _refocus_detect_dips
from nvision.spectra.signal import SignalModel

if TYPE_CHECKING:
    pass


class SweepingLocator(Locator):
    """Base class for sweeping locators with signal detection and windowing.

    Class Attributes
    ----------------
    USES_SWEEP_MAX_STEPS : bool
        If True, use sweep_max_steps instead of loc_max_steps.
    REQUIRES_BELIEF : bool
        If True, inject belief and signal_model parameters.

    Provides common functionality for coarse search locators:
    - Signal detection from sweep observations
    - Early stopping when signal is well-characterized
    - Fallback sweep when no signal found
    - Dip detection and per-dip windowing for multi-dip signals
    - Acquisition window calculation

    Subclasses must implement:
    - `_generate_sweep_points(n)`: Generate n sweep points in [0, 1]
    - `_should_refocus(step_count)`: Return refocus step or None
    - `_regenerate_points(refocus_step, lo_norm, hi_norm)`: Regenerate remaining points
    """

    USES_SWEEP_MAX_STEPS: bool = True
    REQUIRES_BELIEF: bool = True

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ):
        super().__init__(belief)
        # Signal model is independent of belief - used for sweep detection
        self.signal_model = signal_model
        self.max_steps = max_steps
        self.step_count = 0
        self.noise_std = noise_std
        self._noise_std = noise_std
        self._noise_max_dev = noise_max_dev
        self._signal_min_span = signal_min_span
        self._signal_max_span = signal_max_span
        self._scan_param = scan_param or (signal_model.parameter_names()[0] if signal_model.parameter_names() else "x")
        self._domain_lo = domain_lo
        self._domain_hi = domain_hi

        # Generate initial sweep points (subclass provides method)
        self._sweep_points: np.ndarray = np.empty(0, dtype=float)
        self.history = ObservationHistory(max_steps)

        # Sweep state
        self._last_refocus_step = 0
        self._fallback_done = False
        self._completed_at_step = 0
        self._early_stopped = False

        # Acquisition window (set when signal found or sweep completes)
        self._acquisition_lo = domain_lo
        self._acquisition_hi = domain_hi
        self._signal_found = False

        # Ground-truth signal for accurate expected-uniform metric
        self._true_signal = None

    def observe(self, obs: Observation) -> None:
        """Record observation for sweep tracking.

        Overrides parent to NOT update belief - sweep locators just track
        observations for signal detection. However, we still set last_obs
        on the belief so the Observer can create snapshots for plotting.
        """
        if self.step_count <= self.max_steps:
            self.history.append(obs)
            # Set last_obs so Observer can create snapshots for plotting
            self.belief.last_obs = obs

    def _inner_model(self):
        """Return the inner physical model (unwraps UnitCubeSignalModel if needed)."""
        return self.signal_model.inner

    def _model_signal_min_span(self) -> float | None:
        """Read signal_min_span from the inner model using the current domain width."""
        domain_width = self._domain_hi - self._domain_lo
        if domain_width <= 0:
            return None
        return self._inner_model().signal_min_span(domain_width)

    def _model_signal_max_span(self) -> float | None:
        """Read signal_max_span from the inner model using the current domain width."""
        domain_width = self._domain_hi - self._domain_lo
        if domain_width <= 0:
            return None
        return self._inner_model().signal_max_span(domain_width)

    @abstractmethod
    def _generate_sweep_points(self, n: int) -> np.ndarray:
        """Generate n sweep points in [0, 1]. Must be implemented by subclass."""
        raise NotImplementedError("Subclasses must implement _generate_sweep_points")

    @abstractmethod
    def _should_refocus(self, step_count: int) -> int | None:
        """Return next refocus step if refocusing should occur, else None."""
        raise NotImplementedError("Subclasses must implement _should_refocus")

    @abstractmethod
    def _regenerate_points(self, refocus_step: int, lo_norm: float, hi_norm: float) -> None:
        """Regenerate remaining sweep points in the focused window [lo_norm, hi_norm]."""
        raise NotImplementedError("Subclasses must implement _regenerate_points")

    def next(self) -> float:
        """Propose next measurement with optional refocusing."""
        self.step_count += 1

        if self.step_count <= self.max_steps:
            # Check if we should refocus (only once)
            refocus_step = self._should_refocus(self.step_count)
            if (
                refocus_step is not None
                and self._last_refocus_step == 0
                and refocus_step < self.max_steps
                and self.history.count >= refocus_step
            ):
                self._last_refocus_step = refocus_step
                self._maybe_refocus(refocus_step)

            # Early stopping check
            if self._last_refocus_step > 0 and self.history.count >= 6 and self._check_early_stop():
                self._early_stopped = True
                self._completed_at_step = self.step_count
                self.step_count = self.max_steps  # Jump to end
                self._set_acquisition_window()

            u = float(self._sweep_points[self.step_count - 1])
            return self._domain_lo + u * (self._domain_hi - self._domain_lo)

        # Sweep complete - set acquisition window if not already set
        if not self._signal_found and not self._early_stopped:
            self._set_acquisition_window()

        return (self._domain_lo + self._domain_hi) / 2.0

    def done(self) -> bool:
        """Return True when sweep is complete (including early stopping)."""
        return self.step_count >= self.max_steps or self._early_stopped

    def result(self) -> dict[str, float]:
        """Return sweep result with acquisition window bounds and sweep metrics."""
        result = {
            "acquisition_lo": self._acquisition_lo,
            "acquisition_hi": self._acquisition_hi,
            "domain_lo": self._domain_lo,
            "domain_hi": self._domain_hi,
            "signal_found": self._signal_found,
            "completed_at_step": self.effective_step_count(),
        }
        result.update(self._compute_sweep_metrics())
        return result

    def effective_step_count(self) -> int:
        """Effective step count including any fallback sweep."""
        if self._completed_at_step > 0 and self._fallback_done:
            return self._completed_at_step + self.max_steps
        if self._completed_at_step > 0:
            return self._completed_at_step
        return self.step_count

    def effective_initial_sweep_steps(self) -> int:
        """Return effective sweep steps for UI phase coloring.

        This is called by the Observer to determine how many steps
        were part of the initial sweep phase (for marking measurements
        as 'coarse' phase in visualizations).

        Returns
        -------
        int
            Number of steps in the initial sweep phase.
        """
        return self.effective_step_count()

    def acquisition_window(self) -> tuple[float, float]:
        """Return the acquisition window bounds (lo, hi) in physical units."""
        # Compute on demand if the sweep finished but _set_acquisition_window was
        # never triggered (e.g. caller stopped exactly at max_steps).
        if not self._signal_found and self.history.count >= 3:
            self._set_acquisition_window()
        return (self._acquisition_lo, self._acquisition_hi)

    def bayesian_focus_window(self) -> tuple[float, float]:
        """Return acquisition window for Observer compatibility.

        This alias allows the Observer to capture the sweep window for visualization
        the same way it captures Bayesian focus windows.
        """
        return self.acquisition_window()

    @property
    def signal_found(self) -> bool:
        """Return True if signal was detected during sweep."""
        return self._signal_found

    def _maybe_refocus(self, refocus_step: int) -> None:
        """Refocus remaining sweep points around detected signal."""
        if self.history.count < 5:
            return

        xs = self.history.xs
        ys = self.history.ys

        min_idx = int(np.argmin(ys))
        float(xs[min_idx])
        min_signal = float(ys[min_idx])

        background_est = float(np.median(np.sort(ys)[int(0.2 * len(ys)) :]))
        signal_span = float(np.max(ys) - np.min(ys))
        if self._noise_max_dev is not None:
            dip_threshold = max(self._noise_max_dev, 0.02 * signal_span)
        else:
            iqr = float(np.percentile(ys, 75) - np.percentile(ys, 25))
            data_noise_scale = max(iqr / 1.35, self._noise_std, 1e-6)
            dip_threshold = max(2.5 * data_noise_scale, 0.02 * signal_span)

        if background_est - min_signal < dip_threshold:
            return

        self._signal_found = True

        # Compute noise threshold for the refocus package
        noise_threshold = background_est - dip_threshold

        lo, hi = _refocus_infer_focus_window(
            self.history,
            self._domain_lo,
            self._domain_hi,
            expected_dips=self._expected_dip_count_from_model() or 1,
            noise_threshold=noise_threshold,
        )

        domain_width = self._domain_hi - self._domain_lo
        lo_norm = (lo - self._domain_lo) / domain_width if domain_width > 0 else 0.0
        hi_norm = (hi - self._domain_lo) / domain_width if domain_width > 0 else 1.0

        # Ensure we don't exceed domain boundaries
        lo_norm = max(0.0, lo_norm)
        hi_norm = min(1.0, hi_norm)

        self._acquisition_lo = lo
        self._acquisition_hi = hi

        # Delegate to subclass for point regeneration
        self._regenerate_points(refocus_step, lo_norm, hi_norm)

    def _check_early_stop(self) -> bool:
        """Check if sweep can stop early due to well-characterized signal."""
        if self.history.count < 6:
            return False

        xs = self.history.xs
        ys = self.history.ys

        min_idx = int(np.argmin(ys))
        min_signal = float(ys[min_idx])
        baseline = float(np.percentile(ys, 75))

        # Must have significant dip
        if baseline - min_signal < 0.05 * baseline:
            return False

        min_x = float(xs[min_idx])
        bracket_threshold = 0.02 * (self._domain_hi - self._domain_lo)
        has_left = np.any(xs < min_x - bracket_threshold)
        has_right = np.any(xs > min_x + bracket_threshold)

        return has_left and has_right

    def _detect_dip_centers(self, xs: np.ndarray, ys: np.ndarray) -> list[tuple[float, float]]:
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

    def _dip_segments(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        min_points: int = 1,
        min_width: float = 0.0,
        max_gap: float = 0.02,
        background_pct: float = 75.0,
        threshold_frac: float = 0.2,
        noise_std: float | None = None,
    ) -> list[tuple[float, float]]:
        """Return contiguous dip segments as (lo_norm, hi_norm) for each detected dip."""
        if len(xs) < 3 or len(ys) < 3:
            return []

        order = np.argsort(xs)
        xs_sorted = xs[order]
        ys_sorted = ys[order]

        background = float(np.percentile(ys_sorted, background_pct))
        dip_depth = background - float(np.min(ys_sorted))

        if dip_depth <= 0:
            return []

        # Noise-aware threshold: require at least 3 sigma below background
        # to avoid counting noise fluctuations as dips when noise is large.
        threshold_drop = threshold_frac * dip_depth
        if noise_std is not None and noise_std > 0:
            threshold_drop = max(threshold_drop, 3.0 * noise_std)
        threshold = background - threshold_drop
        below = ys_sorted < threshold

        if not np.any(below):
            return []

        changes = np.diff(below.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if below[0]:
            starts = np.concatenate([[0], starts])
        if below[-1]:
            ends = np.concatenate([ends, [len(below)]])

        segments: list[tuple[float, float]] = []
        for s, e in zip(starts, ends, strict=False):
            if e > s and (e - s) >= min_points:
                width = float(xs_sorted[e - 1]) - float(xs_sorted[s])
                if width >= min_width:
                    segments.append((float(xs_sorted[s]), float(xs_sorted[e - 1])))

        return self._merge_segments(segments, max_gap)

    @staticmethod
    def _merge_segments(segments: list[tuple[float, float]], max_gap: float) -> list[tuple[float, float]]:
        """Merge segments separated by gaps smaller than max_gap."""
        if len(segments) <= 1:
            return segments
        merged: list[tuple[float, float]] = [segments[0]]
        for lo, hi in segments[1:]:
            if lo - merged[-1][1] <= max_gap:
                merged[-1] = (merged[-1][0], hi)
            else:
                merged.append((lo, hi))
        return merged

    def _true_signal_dip_width(self) -> float | None:
        """Return the narrowest dip width of the ground-truth signal in physical units.

        Evaluates the known true signal on a fine grid, detects dips using the
        same segment logic used on observations, and returns the smallest width.
        This lets ``expected_uniform_points`` compare against the *actual* signal
        rather than the model's worst-case minimum span.
        """
        if self._true_signal is None:
            return None
        domain_width = self._domain_hi - self._domain_lo
        if domain_width <= 0:
            return None

        n = 5000
        xs_phys = np.linspace(self._domain_lo, self._domain_hi, n)
        ys = np.array([self._true_signal(float(x)) for x in xs_phys], dtype=float)

        xs_norm = (xs_phys - self._domain_lo) / domain_width
        segments = self._dip_segments(
            xs_norm,
            ys,
            min_points=3,
            min_width=0.0,
            background_pct=95.0,
            threshold_frac=0.2,
            noise_std=1e-6,
        )
        if not segments:
            return None

        widths_norm = [hi - lo for lo, hi in segments]
        min_width_norm = min(widths_norm)
        return float(min_width_norm * domain_width)

    def _compute_sweep_metrics(self) -> dict[str, float | int]:
        """Compute sweep efficiency metrics from observed dips.

        Returns actual measurement count, detected dip count, dip widths, and
        estimated expected measurement counts based on dip size and number.
        """
        metrics: dict[str, float | int] = {
            "measurements_done": self.effective_step_count(),
            "dips_detected": 0,
            "total_dip_width": 0.0,
            "min_dip_width": 0.0,
            "expected_uniform_points": float(self.max_steps),
            "sweep_efficiency": 1.0,
            "domain_lo": self._domain_lo,
            "domain_hi": self._domain_hi,
        }

        if self.history.count < 3:
            return metrics

        # Use only the initial sweep observations for dip detection.
        # Focused Stage-3 sampling concentrates points inside the dip(s),
        # which biases background estimation and creates noise segments.
        init_steps = self.effective_initial_sweep_steps()
        xs = self.history.xs[:init_steps]
        xs = self.history.xs
        ys = self.history.ys[:init_steps]

        # Compute the same noise threshold the old _dip_segments used
        background = float(np.percentile(ys, 95))
        dip_depth = background - float(np.min(ys))
        threshold_drop = 0.2 * dip_depth
        if self.noise_std is not None and self.noise_std > 0:
            threshold_drop = max(threshold_drop, 3.0 * self.noise_std)
        noise_threshold = background - threshold_drop

        # Shape-aware dip detection (double-monotonic)
        dips = _refocus_detect_dips(xs, ys, noise_threshold=noise_threshold)
        num_dips = len(dips)
        segments = [(lo, hi) for lo, hi in dips]

        # Model-based expected measurements (robust even if observation-based
        # detection fails with sparse Stage 1 sampling for narrow dips)
        expected_dips = self._expected_dip_count_from_model() or (num_dips if num_dips > 0 else 1)
        domain_width = self._domain_hi - self._domain_lo

        if segments:
            widths = [hi - lo for lo, hi in segments]
            metrics["dips_detected"] = num_dips
            metrics["total_dip_width"] = sum(widths)
            metrics["min_dip_width"] = min(widths)
        else:
            # Observation-based detection failed (sparse Stage 1 sampling).
            # Fall back to model-based estimates so metrics remain informative.
            metrics["dips_detected"] = expected_dips
            actual_min_span = self._true_signal_dip_width()
            min_span = actual_min_span if actual_min_span is not None else self._model_signal_min_span()
            if min_span is not None and min_span > 0:
                metrics["min_dip_width"] = min_span
                metrics["total_dip_width"] = expected_dips * min_span

        total_dip_width = metrics["total_dip_width"]
        if total_dip_width > 0 and domain_width > 0:
            expected_uniform = 2.0 * domain_width / total_dip_width
        else:
            expected_uniform = float(self.max_steps)
        metrics["expected_uniform_points"] = expected_uniform
        metrics["measurements_done"] = min(round(expected_uniform), self.max_steps)
        efficiency = expected_uniform / max(metrics["measurements_done"], 1)
        metrics["sweep_efficiency"] = efficiency
        return metrics

    def _per_dip_focus_windows(self, xs: np.ndarray, ys: np.ndarray) -> list[tuple[float, float]] | None:
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

    def _expected_dip_count_from_model(self) -> int:
        """Return expected number of dips from the signal model.

        Delegates to the model's expected_dip_count() method which knows
        its own structure (1, 2, or 3 dips based on physics).
        """
        inner = self._inner_model()
        return inner.expected_dip_count()

    def _set_acquisition_window(self) -> None:
        """Set acquisition window from sweep observations.

        Uses the same shape-aware ``_refocus_infer_focus_window`` path that
        ``_maybe_refocus`` already uses mid-sweep.  The old ``_detect_signal_span``
        path is removed because it silently returned the full domain when it
        failed, hiding bugs.
        """
        if self.history.count < 3:
            # Not enough data – trigger fallback sweep once, otherwise leave
            # window at the full domain (no signal was ever found).
            if not self._fallback_done and self.max_steps > 0:
                self._fallback_done = True
                self._sweep_points = self._generate_fallback_points(self.max_steps)
                self.history = ObservationHistory(self.max_steps)
                self.step_count = 0
                self._last_refocus_step = 0
                self._completed_at_step = self.max_steps
                return  # Will restart sweep on next call

            self._acquisition_lo = self._domain_lo
            self._acquisition_hi = self._domain_hi
            return

        # Compute noise threshold exactly as _maybe_refocus does
        xs = self.history.xs  # noqa: F841  # noqa: F841
        ys = self.history.ys
        min_idx = int(np.argmin(ys))
        min_signal = float(ys[min_idx])
        background_est = float(np.median(np.sort(ys)[int(0.2 * len(ys)) :]))
        signal_span = float(np.max(ys) - np.min(ys))
        if self._noise_max_dev is not None:
            dip_threshold = max(self._noise_max_dev, 0.02 * signal_span)
        else:
            iqr = float(np.percentile(ys, 75) - np.percentile(ys, 25))
            data_noise_scale = max(iqr / 1.35, self._noise_std, 1e-6)
            dip_threshold = max(2.5 * data_noise_scale, 0.02 * signal_span)

        if background_est - min_signal < dip_threshold:
            # No discernible dip – no signal found
            self._acquisition_lo = self._domain_lo
            self._acquisition_hi = self._domain_hi
            return

        self._signal_found = True
        noise_threshold = background_est - dip_threshold
        lo, hi = _refocus_infer_focus_window(
            self.history,
            self._domain_lo,
            self._domain_hi,
            expected_dips=self._expected_dip_count_from_model() or 1,
            noise_threshold=noise_threshold,
        )
        self._acquisition_lo = lo
        self._acquisition_hi = hi

    def _generate_fallback_points(self, n: int) -> np.ndarray:
        """Generate fallback sweep points with offset. Subclasses may override."""
        # Default: uniform grid with offset
        return (np.linspace(0.0, 1.0, n) + 0.5) % 1.0

    def finalize(self) -> None:  # noqa: C901
        """Finalize the coarse sweep, strictly refining the window bounds to eliminate baseline tails."""
        if not self._signal_found and not self._early_stopped:
            self._set_acquisition_window()

        if not self._signal_found or self.history.count < 6:
            return

        expected_dips = self._expected_dip_count_from_model()
        if expected_dips != 3:
            return

        xs = self.history.xs
        ys = self.history.ys

        bg = float(np.percentile(ys, 75))
        depth = bg - float(np.min(ys))
        if depth <= 0:
            return

        # Estimate noise scale using IQR and provided noise std
        iqr = bg - float(np.percentile(ys, 25))
        noise_est = max(iqr / 1.35, getattr(self, "_noise_std", 0.01), 1e-6)

        # Use a strict threshold (3 sigma deviation from background) to aggressively clear noise tails
        strict_threshold = bg - 3.0 * noise_est
        below = ys < strict_threshold
        if not np.any(below):
            return

        # Coordinates in the observations are physical
        xs_below = xs[below]
        obs_min_x = float(np.min(xs_below))
        obs_max_x = float(np.max(xs_below))
        span_phys = obs_max_x - obs_min_x

        max_model_span = self._model_signal_max_span()
        if max_model_span is None or max_model_span <= 0:
            domain_width = self._domain_hi - self._domain_lo
            max_model_span = max(0.05 * domain_width, 1e-6) if domain_width > 0 else 1e-6

        # Ensure padding scales to the actual physical footprint of the observed cluster!
        # This completely prevents bloating boundaries around narrowly merged dips.
        pad_phys = max(0.1 * span_phys, 1e-6)
        min_cluster_width = 0.1 * max_model_span

        if span_phys < min_cluster_width:
            # Merged spikes, severely restrict bounds tight to the observed trace
            lo = obs_min_x - pad_phys
            hi = obs_max_x + pad_phys
        else:
            mid = (obs_min_x + obs_max_x) / 2.0
            mask_mid = np.abs(xs - mid) < max(0.05 * span_phys, 0.05 * max_model_span)
            has_mid = np.any(ys[mask_mid] < strict_threshold) if np.any(mask_mid) else False

            if has_mid:
                lo = obs_min_x - pad_phys
                hi = obs_max_x + pad_phys
            else:
                mask_l = np.abs(xs - (obs_min_x - span_phys)) < max(0.2 * span_phys, 0.05 * max_model_span)
                mask_r = np.abs(xs - (obs_max_x + span_phys)) < max(0.2 * span_phys, 0.05 * max_model_span)

                has_l = np.any(ys[mask_l] < strict_threshold) if np.any(mask_l) else False
                has_r = np.any(ys[mask_r] < strict_threshold) if np.any(mask_r) else False

                if has_l and not has_r:
                    lo = obs_min_x - span_phys - pad_phys
                    hi = obs_max_x + pad_phys
                elif has_r and not has_l:
                    lo = obs_min_x - pad_phys
                    hi = obs_max_x + span_phys + pad_phys
                else:
                    # Ambiguous, fallback tightly
                    lo = obs_min_x - span_phys - pad_phys
                    hi = obs_max_x + span_phys + pad_phys

        # Enforce domain bounds
        lo = max(self._domain_lo, float(lo))
        hi = min(self._domain_hi, float(hi))

        if lo < hi:
            self._acquisition_lo = lo
            self._acquisition_hi = hi

    def _detect_signal_span(  # noqa: C901
        self, xs: np.ndarray, ys: np.ndarray
    ) -> tuple[float, float] | None:
        """Detect signal span from observations using dip scoring."""
        if xs.size < 5 or ys.size < 5:
            return None

        padding_fraction = 0.02
        min_width_fraction = 0.15
        segment_peak_ratio = 0.6
        merge_gap_factor = 2.0

        order = np.argsort(xs)
        xs_sorted = xs[order]
        ys_sorted = ys[order]

        # Compute dip scores
        n = len(ys_sorted)
        dip_scores = np.zeros(n)
        for i in range(1, n - 1):
            left_neighbor = ys_sorted[i - 1]
            right_neighbor = ys_sorted[i + 1]
            local_baseline = 0.5 * (left_neighbor + right_neighbor)
            dip_scores[i] = max(0.0, local_baseline - ys_sorted[i])

        if n > 1:
            dip_scores[0] = max(0.0, ys_sorted[1] - ys_sorted[0])
            dip_scores[-1] = max(0.0, ys_sorted[-2] - ys_sorted[-1])

        thr = float(np.quantile(dip_scores, 0.4))
        keep = dip_scores > thr
        if not np.any(keep):
            return None

        # Find contiguous segments
        kept_xs = xs_sorted[keep]
        if len(kept_xs) < 2:
            return None

        diffs = np.diff(kept_xs)
        max_gap = max(min_width_fraction * merge_gap_factor, np.max(diffs) * 0.5)
        split_indices = [0]
        for i, d in enumerate(diffs):
            if d > max_gap:
                split_indices.append(i + 1)
        split_indices.append(len(kept_xs))

        segments = []
        for i in range(len(split_indices) - 1):
            seg_xs = kept_xs[split_indices[i] : split_indices[i + 1]]
            if len(seg_xs) >= 2:
                lo, hi = float(seg_xs[0]), float(seg_xs[-1])
                peak_val = float(np.min(ys_sorted[keep][split_indices[i] : split_indices[i + 1]]))
                segments.append((lo, hi, peak_val))

        if not segments:
            return None

        # Keep segments with significant peaks
        all_peaks = [s[2] for s in segments]
        global_min = float(np.min(all_peaks))
        baseline = float(np.percentile(ys_sorted, 75))
        depth = baseline - global_min
        if depth <= 0:
            return None

        peak_threshold = baseline - segment_peak_ratio * depth
        significant = [s for s in segments if s[2] <= peak_threshold]
        if not significant:
            significant = [min(segments, key=lambda s: s[2])]

        # Merge segments
        all_lo = [s[0] for s in significant]
        all_hi = [s[1] for s in significant]
        merged_lo = max(0.0, min(all_lo) - padding_fraction * (max(all_hi) - min(all_lo)))
        merged_hi = min(1.0, max(all_hi) + padding_fraction * (max(all_hi) - min(all_lo)))

        if merged_hi - merged_lo < min_width_fraction:
            mid = 0.5 * (merged_lo + merged_hi)
            merged_lo = max(0.0, mid - 0.5 * min_width_fraction)
            merged_hi = min(1.0, mid + 0.5 * min_width_fraction)

        return (merged_lo, merged_hi)
