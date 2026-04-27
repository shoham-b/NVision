"""Sobol-based coarse search locator with 3 stages.

1. 255 points (7th Sobol stage)
2. Further Sobol points monitoring for 2 dips using dynamic noise thresholding.
3. Windowed Sobol focusing on inferred target dips.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.locator import Locator
from nvision.models.observation import Observation, ObservationHistory
from nvision.sim.locs.coarse.sweep_locator import SweepingLocator, _infer_focus_window
from nvision.spectra.signal import SignalModel

if TYPE_CHECKING:
    pass


def sobol_1d_sequence(n: int, *, offset: float = 0.0) -> NDArray[np.float64]:
    """Minimal deterministic 1D low-discrepancy sequence over [0, 1].

    Uses a van der Corput base-2 sequence.
    """

    def vdc(k: int, base: int = 2) -> float:
        v = 0.0
        denom = 1.0
        while k:
            k, remainder = divmod(k, base)
            denom *= base
            v += remainder / denom
        return v

    points = np.array([vdc(i + 1) for i in range(n)], dtype=float)
    if offset != 0.0:
        points = (points + offset) % 1.0
    return points


def vdc_generator(base: int = 2) -> Iterator[float]:
    """Generator for van der Corput sequence without allocating fixed size array."""
    k = 1
    while True:
        num = k
        v = 0.0
        denom = 1.0
        while num:
            num, remainder = divmod(num, base)
            denom *= base
            v += remainder / denom
        yield v
        k += 1


class SobolSweepLocator(SweepingLocator):
    """Simple Sobol-based sweep locator without refocusing.

    Generates a van der Corput (Sobol-like) low-discrepancy sequence over the
    full domain, detects signal dips from the sweep data, and sets an
    acquisition window. Unlike ``StagedSobolSweepLocator``, there is no Stage 2
    thresholding or Stage 3 focused sampling — just a single sweep with
    signal detection handled by the inherited ``finalize()``.
    """

    @classmethod
    def create(
        cls,
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
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> SobolSweepLocator:
        """Factory method for creating a SobolSweepLocator."""
        if parameter_bounds is not None:
            param_name = scan_param or (
                signal_model.parameter_names()[0] if signal_model.parameter_names() else "peak_x"
            )
            if param_name in parameter_bounds:
                domain_lo, domain_hi = parameter_bounds[param_name]

        return cls(
            belief=belief,
            signal_model=signal_model,
            max_steps=max_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_min_span=signal_min_span,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )

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
        super().__init__(
            belief=belief,
            signal_model=signal_model,
            max_steps=max_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_min_span=signal_min_span,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )
        # Refocusing disabled — single sweep over full domain
        self._refocus_at = None
        self._sweep_points = self._generate_sweep_points(max_steps)

    def _generate_sweep_points(self, n: int) -> NDArray[np.float64]:
        """Generate n Sobol sequence points in [0, 1]."""
        if n <= 0:
            return np.array([], dtype=float)
        return sobol_1d_sequence(n)

    def _generate_fallback_points(self, n: int) -> NDArray[np.float64]:
        """Generate fallback Sobol points with 0.5 offset for coverage."""
        if n <= 0:
            return np.array([], dtype=float)
        return sobol_1d_sequence(n, offset=0.5)

    def _should_refocus(self, step_count: int) -> int | None:
        """No refocusing for simple Sobol sweep."""
        return None

    def _regenerate_points(self, refocus_step: int, lo_norm: float, hi_norm: float) -> None:
        """No-op — refocusing is disabled."""
        pass


class Stage1SobolLocator:
    """Stage 1: Collect exactly 255 points to establish a robust noise baseline."""

    def __init__(self, sobol_gen: Iterator[float], domain_lo: float, domain_hi: float):
        self._sobol_gen = sobol_gen
        self.domain_lo = domain_lo
        self.domain_hi = domain_hi
        self.points_collected = 0

    def next(self) -> float:
        u = next(self._sobol_gen)
        return self.domain_lo + u * (self.domain_hi - self.domain_lo)

    def observe(self, obs: Observation) -> None:
        self.points_collected += 1

    def done(self) -> bool:
        return self.points_collected >= 255

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        """Return None — simple sweep locators do not narrow the window."""
        return None



class Stage2SobolLocator:
    """Stage 2: Continue scanning within narrowed window. Stop upon finding 2 dip points."""

    def __init__(
        self,
        sobol_gen: Iterator[float],
        domain_lo: float,
        domain_hi: float,
        history: ObservationHistory,
        *,
        window_lo: float | None = None,
        window_hi: float | None = None,
    ):
        self._sobol_gen = sobol_gen
        self.domain_lo = domain_lo
        self.domain_hi = domain_hi
        self.history = history

        self.window_lo = window_lo if window_lo is not None else domain_lo
        self.window_hi = window_hi if window_hi is not None else domain_hi

        self._noise_threshold = -float("inf")
        self._done = False

        # Initialize threshold using the 255 observations inherited from Stage 1
        self._update_noise_threshold()
        self._check_for_dips()

    def next(self) -> float:
        u = next(self._sobol_gen)
        return self.window_lo + u * (self.window_hi - self.window_lo)

    def observe(self, obs: Observation) -> None:
        # Update dynamically every 128 new scans in Stage 2
        if self.history.count % 128 == 0:
            self._update_noise_threshold()

        self._check_for_dips()

    def done(self) -> bool:
        return self._done

    def _update_noise_threshold(self) -> None:
        if self.history.count == 0:
            return

        ys_valid = self.history.ys
        # Extract top 70% of measurements (discard the bottom 30% which may contain signal dips)
        p30_val = float(np.percentile(ys_valid, 30))
        noise_points = ys_valid[ys_valid >= p30_val]

        noise_median = float(np.median(noise_points))
        noise_std = float(np.std(noise_points))

        # Threshold is defined as 3 stds below the median of the noise points
        self._noise_threshold = noise_median - 3.0 * noise_std

    def _check_for_dips(self) -> None:
        ys_valid = self.history.ys

        below_idx = np.where(ys_valid < self._noise_threshold)[0]
        if len(below_idx) >= 2:
            self._done = True


class Stage3SobolLocator:
    """Stage 3: Infer structural bounds and exclusively generate points inside them."""

    def __init__(self, sobol_gen: Iterator[float], domain_lo: float, domain_hi: float, history: ObservationHistory):
        self._sobol_gen = sobol_gen
        self.domain_lo = domain_lo
        self.domain_hi = domain_hi
        self.history = history

        self.window_lo = domain_lo
        self.window_hi = domain_hi

        # Calculate exact bounds securely bypassing noise logic issues
        self._infer_bounds()

    def next(self) -> float:
        """Return the next Sobol point scaled directly to the focus window.

        The van der Corput sequence gives uniform low-discrepancy coverage
        of [0, 1]; scaling it linearly to ``[window_lo, window_hi]`` preserves
        that property inside the window without expensive rejection sampling.
        """
        u = next(self._sobol_gen)
        return self.window_lo + u * (self.window_hi - self.window_lo)

    def observe(self, obs: Observation) -> None:
        pass

    def done(self) -> bool:
        # When used inside SequentialBayesianLocator, the parent checks this and
        # takes over. When used standalone (StagedSobolSweepLocator), we continue
        # generating focused points; the parent handles max_steps via step_count.
        return False

    def _infer_bounds(self) -> None:
        self.window_lo, self.window_hi = _infer_focus_window(
            self.history, self.domain_lo, self.domain_hi
        )


class StagedSobolSweepLocator(Locator):
    """Orchestrator locator managing the 3-stage targeted Sobol methodology."""

    USES_SWEEP_MAX_STEPS: bool = True
    REQUIRES_BELIEF: bool = True

    @classmethod
    def create(
        cls,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int = 300,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> StagedSobolSweepLocator:
        if parameter_bounds is not None:
            param_name = scan_param or (
                signal_model.parameter_names()[0] if signal_model.parameter_names() else "peak_x"
            )
            if param_name in parameter_bounds:
                domain_lo, domain_hi = parameter_bounds[param_name]

        return cls(
            belief=belief,
            signal_model=signal_model,
            max_steps=max_steps,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
        )

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
    ):
        super().__init__(belief)
        self.signal_model = signal_model
        self.max_steps = max_steps
        self.domain_lo = domain_lo
        self.domain_hi = domain_hi
        self.noise_std = noise_std
        self.noise_max_dev = noise_max_dev
        self.signal_max_span = signal_max_span
        self.scan_param = scan_param

        self.step_count = 0
        self.history = ObservationHistory(self.max_steps)
        self._sobol_gen = vdc_generator()
        self._signal_found = False
        self._true_signal = None

        # Track initial sweep steps for Observer phase coloring
        self._initial_sweep_steps = 0
        self._stage1_end_step = 0

        self._stage1 = Stage1SobolLocator(self._sobol_gen, self.domain_lo, self.domain_hi)
        self._stage2: Stage2SobolLocator | None = None
        self._stage3: Stage3SobolLocator | None = None

        self._active_locator: Stage1SobolLocator | Stage2SobolLocator | Stage3SobolLocator = self._stage1

    def next(self) -> float:
        self.step_count += 1

        if self.done():
            return (self.domain_lo + self.domain_hi) / 2.0

        return self._active_locator.next()

    def done(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        return bool(self._active_locator is self._stage3 and self._stage3.done())

    def observe(self, obs: Observation) -> None:
        if self.step_count > self.max_steps:
            return

        self.history.append(obs)
        # Set last_obs so Observer can create snapshots for plotting
        self.belief.last_obs = obs
        self._active_locator.observe(obs)

        if self._active_locator is self._stage1 and self._active_locator.done():
            self._stage1_end_step = self.step_count
            win_lo, win_hi = _infer_focus_window(self.history, self.domain_lo, self.domain_hi)
            self._stage2 = Stage2SobolLocator(
                self._sobol_gen, self.domain_lo, self.domain_hi, self.history,
                window_lo=win_lo, window_hi=win_hi,
            )
            self._active_locator = self._stage2

            # Cascade: Stage 2 might be instantaneously complete if dips found in Stage 1 data!
            if self._active_locator.done():
                self._transition_to_stage3()

        elif self._active_locator is self._stage2 and self._active_locator.done():
            self._transition_to_stage3()

    def _transition_to_stage3(self) -> None:
        self._initial_sweep_steps = self._stage1_end_step
        self._stage3 = Stage3SobolLocator(self._sobol_gen, self.domain_lo, self.domain_hi, self.history)
        self._active_locator = self._stage3
        self._signal_found = True

    def effective_initial_sweep_steps(self) -> int:
        """Return number of steps in Stage 1 (coarse phase).

        Called by the Observer to determine how many measurements belong to
        the 'coarse' phase for plot coloring.
        """
        if self._stage1_end_step > 0:
            return self._stage1_end_step
        # If we never reached Stage 2, all completed steps were Stage 1
        return self.step_count

    def secondary_sweep_count(self) -> int:
        """Return number of steps in Stage 2 (secondary phase).

        Called by the Observer for secondary phase coloring.
        """
        if self._initial_sweep_steps > 0:
            return self._initial_sweep_steps - self._stage1_end_step
        return 0

    def finalize(self) -> None:
        """Infer the focus window from collected data if stages didn't complete."""
        if self._stage3 is None and self.history.count > 0:
            lo, hi = _infer_focus_window(self.history, self.domain_lo, self.domain_hi)
            if hi > lo and (hi - lo) < (self.domain_hi - self.domain_lo):
                self._signal_found = True
                self._inferred_lo = lo
                self._inferred_hi = hi

    def acquisition_window(self) -> tuple[float, float]:
        if self._stage3 is not None:
            return (self._stage3.window_lo, self._stage3.window_hi)
        # Fall back to inference from collected data (set by finalize or observe)
        if hasattr(self, "_inferred_lo"):
            return (self._inferred_lo, self._inferred_hi)
        # Last resort: infer directly from history
        if self.history.count > 0:
            return _infer_focus_window(self.history, self.domain_lo, self.domain_hi)
        return (self.domain_lo, self.domain_hi)

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        """Return the narrowed focus window from any available stage or inference."""
        if self._stage3 is not None:
            return (self._stage3.window_lo, self._stage3.window_hi)
        if self._stage2 is not None:
            return (self._stage2.window_lo, self._stage2.window_hi)
        if hasattr(self, "_inferred_lo"):
            return (self._inferred_lo, self._inferred_hi)
        # Infer from history if we have data
        if self.history.count > 0:
            lo, hi = _infer_focus_window(self.history, self.domain_lo, self.domain_hi)
            if hi > lo and (hi - lo) < (self.domain_hi - self.domain_lo):
                return (lo, hi)
        return None

    def _model_expected_measurements(
        self, num_dips: int, total_dip_width: float | None = None, domain_width: float | None = None
    ) -> dict[str, float | int]:
        """Compute expected measurement counts from model bounds."""
        if domain_width is None:
            domain_width = self.domain_hi - self.domain_lo
        if total_dip_width is not None and total_dip_width > 0 and domain_width > 0:
            expected_uniform = 2.0 * domain_width / total_dip_width
        else:
            min_span = self.signal_model.signal_min_span(domain_width)
            if min_span is not None and min_span > 0 and domain_width > 0:
                expected_uniform = 6.0 * domain_width / min_span
            else:
                expected_uniform = float(self.max_steps)
        expected_focused = num_dips * 5.0 + 20.0 if num_dips > 0 else expected_uniform
        measurements_done = min(int(round(expected_uniform)), self.max_steps)
        efficiency = expected_uniform / max(measurements_done, 1)
        return {
            "measurements_done": measurements_done,
            "expected_uniform_points": expected_uniform,
            "expected_focused_points": expected_focused,
            "sweep_efficiency": efficiency,
        }

    def _detect_dip_segments(
        self, xs: np.ndarray, ys: np.ndarray, noise_std: float | None = None, min_width: float = 0.005
    ) -> list[tuple[float, float]]:
        """Find contiguous dip segments in sorted x/y data."""
        background = float(np.percentile(ys, 95))
        dip_depth = background - float(np.min(ys))
        if dip_depth <= 0:
            return []
        threshold_drop = 0.2 * dip_depth
        if noise_std is not None and noise_std > 0:
            threshold_drop = max(threshold_drop, 3.0 * noise_std)
        threshold = background - threshold_drop
        below = ys < threshold
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
            if e > s and (e - s) >= 3:
                width = float(xs[e - 1]) - float(xs[s])
                if width >= min_width:
                    segments.append((float(xs[s]), float(xs[e - 1])))

        return self._merge_segments(segments, max_gap=0.02)

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

    def _true_signal_segments(self) -> list[tuple[float, float]] | None:
        """Return detected dip segments on the ground-truth signal (normalized)."""
        if self._true_signal is None:
            return None
        domain_width = self.domain_hi - self.domain_lo
        if domain_width <= 0:
            return None
        n = 5000
        xs_phys = np.linspace(self.domain_lo, self.domain_hi, n)
        try:
            ys = np.array([self._true_signal(float(x)) for x in xs_phys], dtype=float)
        except Exception:
            return None
        xs_norm = (xs_phys - self.domain_lo) / domain_width
        return self._detect_dip_segments(xs_norm, ys, noise_std=1e-6, min_width=0.0)

    def _true_signal_dip_width(self) -> float | None:
        """Return the narrowest dip width of the ground-truth signal in physical units."""
        segments = self._true_signal_segments()
        if not segments:
            return None
        domain_width = self.domain_hi - self.domain_lo
        widths_norm = [hi - lo for lo, hi in segments]
        return float(min(widths_norm) * domain_width)

    def _true_signal_total_dip_width(self) -> float | None:
        """Return the total dip width of the ground-truth signal in physical units."""
        segments = self._true_signal_segments()
        if not segments:
            return None
        domain_width = self.domain_hi - self.domain_lo
        return float(sum(hi - lo for lo, hi in segments) * domain_width)

    def _true_signal_span(self) -> float | None:
        """Return total signal span (last dip end - first dip start) in physical units."""
        segments = self._true_signal_segments()
        if not segments:
            return None
        domain_width = self.domain_hi - self.domain_lo
        return float((segments[-1][1] - segments[0][0]) * domain_width)

    def _true_signal_dips_merged(self) -> bool | None:
        """Check whether detected dips are close enough to be considered one combined range.

        Merged means the total span is not much larger than the sum of individual
        dip widths (gaps are small relative to the dips themselves).
        """
        segments = self._true_signal_segments()
        if not segments:
            return None
        if len(segments) == 1:
            return True
        total_width = sum(hi - lo for lo, hi in segments)
        total_span = segments[-1][1] - segments[0][0]
        # Merged if span is less than 1.5x the total width (gaps are small)
        return total_span <= total_width * 1.5

    def _true_signal_dip_count(self) -> int | None:
        """Return the actual number of dips in the ground-truth signal."""
        segments = self._true_signal_segments()
        if segments is None:
            return None
        return len(segments)

    def _compute_sweep_metrics(self) -> dict[str, float | int]:
        """Compute sweep efficiency metrics from observed dips."""
        metrics: dict[str, float | int] = {
            "measurements_done": self.step_count,
            "dips_detected": 0,
            "total_dip_width": 0.0,
            "min_dip_width": 0.0,
            "expected_uniform_points": float(self.max_steps),
            "expected_focused_points": 0.0,
            "sweep_efficiency": 1.0,
        }

        if self.history.count < 3:
            return metrics

        init_steps = self.effective_initial_sweep_steps()
        xs = self.history.xs[:init_steps]
        ys = self.history.ys[:init_steps]
        order = np.argsort(xs)
        segments = self._detect_dip_segments(xs[order], ys[order], noise_std=self.noise_std)

        # Prefer actual ground-truth dip count when available
        true_dip_count = self._true_signal_dip_count()
        expected_dips = true_dip_count if true_dip_count is not None else (
            self.signal_model.expected_dip_count() or (len(segments) if segments else 1)
        )
        domain_width = self.domain_hi - self.domain_lo
        min_span = self.signal_model.signal_min_span(domain_width)

        if segments:
            widths = [hi - lo for lo, hi in segments]
            metrics["dips_detected"] = len(segments)
            metrics["total_dip_width"] = sum(widths)
            metrics["min_dip_width"] = min(widths)
        else:
            metrics["dips_detected"] = expected_dips
            if min_span is not None and min_span > 0:
                metrics["min_dip_width"] = min_span
                metrics["total_dip_width"] = expected_dips * min_span

        # Override with actual signal dip width from ground truth when available
        true_min = self._true_signal_dip_width()
        true_total = self._true_signal_total_dip_width()
        true_span = self._true_signal_span()
        merged = self._true_signal_dips_merged()
        if true_min is not None:
            metrics["min_dip_width"] = true_min
        if true_total is not None:
            metrics["total_dip_width"] = true_total
        if true_span is not None:
            metrics["total_signal_span"] = true_span
        if merged is not None:
            metrics["dips_merged"] = merged

        # Use span for expected_uniform when dips are merged (one combined range),
        # otherwise fall back to total dip width
        effective_width = metrics.get("total_signal_span") if merged else metrics.get("total_dip_width")
        metrics.update(self._model_expected_measurements(expected_dips, effective_width, domain_width))
        return metrics

    def result(self) -> dict[str, float | bool]:
        lo, hi = self.acquisition_window()
        result = {
            "acquisition_lo": lo,
            "acquisition_hi": hi,
            "domain_lo": self.domain_lo,
            "domain_hi": self.domain_hi,
            "signal_found": self._signal_found,
            "completed_at_step": self.step_count,
        }
        result.update(self._compute_sweep_metrics())
        return result
