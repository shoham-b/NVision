"""Sobol-based coarse search locator.

Uses a low-discrepancy Sobol sequence over [0, 1] for needle-in-a-haystack style
global search on mostly-zero signals.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter
from nvision.models.locator import Locator
from nvision.models.observation import Observation
from nvision.sim.locs.coarse.numba_kernels import gaussian_peak_posterior_update
from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.signal import ParamSpec, SignalModel

if TYPE_CHECKING:
    from nvision.belief.unit_cube_grid_marginal import UnitCubeGridMarginalDistribution
    from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution


def sobol_1d_sequence(n: int, *, offset: float = 0.0) -> NDArray[np.float64]:
    """Minimal deterministic 1D low-discrepancy sequence over [0, 1].


    Uses a van der Corput base-2 sequence as a stand-in for a true Sobol
    sequence.  The optional ``offset`` shifts all points by a fixed amount
    (mod 1) so that a second sweep can avoid the same gap pattern when
    the first sweep finds no signal.
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


@dataclass(frozen=True)
class _BlackBoxParams:
    peak_x: float


@dataclass(frozen=True)
class _BlackBoxSampleParams:
    peak_x: np.ndarray


@dataclass(frozen=True)
class _BlackBoxUncertaintyParams:
    peak_x: float


class _BlackBoxSpec(ParamSpec[_BlackBoxParams, _BlackBoxSampleParams, _BlackBoxUncertaintyParams]):
    @property
    def names(self) -> tuple[str, ...]:
        return ("peak_x",)

    @property
    def dim(self) -> int:
        return 1

    def unpack_params(self, values) -> _BlackBoxParams:
        (v,) = values
        return _BlackBoxParams(float(v))

    def pack_params(self, params: _BlackBoxParams) -> tuple[float, ...]:
        return (float(params.peak_x),)

    def unpack_uncertainty(self, values) -> _BlackBoxUncertaintyParams:
        (v,) = values
        return _BlackBoxUncertaintyParams(float(v))

    def pack_uncertainty(self, u: _BlackBoxUncertaintyParams) -> tuple[float, ...]:
        return (float(u.peak_x),)

    def unpack_samples(self, arrays_in_order) -> _BlackBoxSampleParams:
        (px,) = arrays_in_order
        return _BlackBoxSampleParams(peak_x=np.asarray(px, dtype=FLOAT_DTYPE))

    def pack_samples(self, samples: _BlackBoxSampleParams) -> tuple[np.ndarray, ...]:
        return (np.asarray(samples.peak_x, dtype=FLOAT_DTYPE),)


class BlackBoxSignalModel(SignalModel[_BlackBoxParams, _BlackBoxSampleParams, _BlackBoxUncertaintyParams]):
    """Signal model placeholder for Sobol search (we only measure)."""

    _SPEC = _BlackBoxSpec()

    @property
    def spec(self) -> _BlackBoxSpec:
        return self._SPEC

    def compute(self, x: float, params: _BlackBoxParams) -> float:
        return 0.0

    def compute_vectorized_samples(self, x: float, samples: _BlackBoxSampleParams) -> np.ndarray:
        # Not used by coarse locators' observe() path; exists for compatibility.
        return np.zeros_like(samples.peak_x, dtype=FLOAT_DTYPE)


class SobolLocator(Locator):
    """Coarse locator using a 1D Sobol sequence over [0, 1]."""

    def __init__(self, belief: AbstractMarginalDistribution, max_steps: int = 64):
        super().__init__(belief)
        self.max_steps = max_steps
        self.step_count = 0
        self._points = self._sobol_1d(max_steps)
        self.best_signal = -np.inf
        self.best_x = 0.5

    @classmethod
    def create(cls, max_steps: int = 64, n_grid: int = 128, **kwargs) -> SobolLocator:
        model = BlackBoxSignalModel()
        belief = GridMarginalDistribution(
            model=model,
            parameters=[
                GridParameter(
                    name="peak_x",
                    bounds=(0.0, 1.0),
                    grid=np.linspace(0.0, 1.0, n_grid),
                    posterior=np.ones(n_grid) / n_grid,
                ),
            ],
        )
        return cls(belief, max_steps)

    def _sobol_1d(self, n: int) -> NDArray[np.float64]:
        """Minimal 1D Sobol sequence generator over [0, 1]."""
        return sobol_1d_sequence(n)

    def next(self) -> float:
        x = self._points[self.step_count]
        self.step_count += 1
        return float(x)

    def done(self) -> bool:
        return self.step_count >= self.max_steps

    def result(self) -> dict[str, float]:
        return {
            "peak_x": self.best_x,
            "peak_signal": self.best_signal,
        }

    def observe(self, obs) -> None:
        if obs.signal_value > self.best_signal:
            self.best_signal = obs.signal_value
            self.best_x = obs.x
        peak_param = self.belief.get_grid_param("peak_x")
        updated_posterior, total = gaussian_peak_posterior_update(
            peak_param.grid,
            peak_param.posterior,
            float(obs.x),
            float(obs.signal_value),
        )
        if total > 1e-10:
            peak_param.posterior = updated_posterior
            peak_param.value = peak_param.mean()
        self.belief.last_obs = obs


class SweepingLocator(Locator):
    """Base class for sweeping locators with signal detection and windowing.

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

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        max_steps: int,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ):
        super().__init__(belief)
        self.max_steps = max_steps
        self.step_count = 0
        self._noise_std = noise_std
        self._noise_max_dev = noise_max_dev
        self._signal_max_span = signal_max_span
        self._scan_param = scan_param or (belief.model.parameter_names()[0] if belief.model.parameter_names() else "x")
        self._domain_lo = domain_lo
        self._domain_hi = domain_hi

        # Generate initial sweep points (subclass provides method)
        self._sweep_points: np.ndarray = np.empty(0, dtype=float)
        self._sweep_observations: list[Observation] = []

        # Sweep state
        self._last_refocus_step = 0
        self._fallback_done = False
        self._completed_at_step = 0
        self._early_stopped = False

        # Acquisition window (set when signal found or sweep completes)
        self._acquisition_lo = domain_lo
        self._acquisition_hi = domain_hi
        self._signal_found = False

    def _inner_model(self):
        """Return the inner physical model (unwraps UnitCubeSignalModel if needed)."""
        return getattr(self.belief.model, "inner", self.belief.model)

    def _model_signal_min_span(self) -> float | None:
        """Read signal_min_span from the inner model using the current domain width."""
        domain_width = self._domain_hi - self._domain_lo
        if domain_width <= 0:
            return None
        m = getattr(self._inner_model(), "signal_min_span", None)
        if callable(m):
            return m(domain_width)
        return None

    def _model_signal_max_span(self) -> float | None:
        """Read signal_max_span from the inner model using the current domain width."""
        domain_width = self._domain_hi - self._domain_lo
        if domain_width <= 0:
            return None
        m = getattr(self._inner_model(), "signal_max_span", None)
        if callable(m):
            return m(domain_width)
        return None

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
            # Check if we should refocus
            refocus_step = self._should_refocus(self.step_count)
            if (
                refocus_step is not None
                and refocus_step < self.max_steps
                and len(self._sweep_observations) >= refocus_step
            ):
                self._last_refocus_step = refocus_step
                self._maybe_refocus(refocus_step)

            # Early stopping check
            if self._last_refocus_step > 0 and len(self._sweep_observations) >= 6:
                if self._check_early_stop():
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

    def observe(self, obs: Observation) -> None:
        """Record observation."""
        if self.step_count <= self.max_steps:
            self._sweep_observations.append(obs)

    def result(self) -> dict[str, float]:
        """Return sweep result with acquisition window bounds."""
        return {
            "acquisition_lo": self._acquisition_lo,
            "acquisition_hi": self._acquisition_hi,
            "signal_found": self._signal_found,
            "completed_at_step": self.effective_step_count(),
        }

    def effective_step_count(self) -> int:
        """Effective step count including any fallback sweep."""
        if self._completed_at_step > 0 and self._fallback_done:
            return self._completed_at_step + self.max_steps
        if self._completed_at_step > 0:
            return self._completed_at_step
        return self.step_count

    def acquisition_window(self) -> tuple[float, float]:
        """Return the acquisition window bounds (lo, hi) in physical units."""
        return (self._acquisition_lo, self._acquisition_hi)

    @property
    def signal_found(self) -> bool:
        """Return True if signal was detected during sweep."""
        return self._signal_found

    def _maybe_refocus(self, refocus_step: int) -> None:
        """Refocus remaining sweep points around detected signal."""
        if len(self._sweep_observations) < 5:
            return

        xs = np.array([float(o.x) for o in self._sweep_observations], dtype=float)
        ys = np.array([float(o.signal_value) for o in self._sweep_observations], dtype=float)

        min_idx = int(np.argmin(ys))
        best_point_norm = float(xs[min_idx])
        min_signal = float(ys[min_idx])

        background_est = float(np.median(np.sort(ys)[int(0.2 * len(ys)):]))
        if self._noise_max_dev is not None:
            dip_threshold = max(self._noise_max_dev, 0.04)
        else:
            iqr = float(np.percentile(ys, 75) - np.percentile(ys, 25))
            data_noise_scale = max(iqr / 1.35, self._noise_std, 1e-6)
            dip_threshold = max(2.5 * data_noise_scale, 0.04)

        if background_est - min_signal < dip_threshold:
            return

        self._signal_found = True

        # Calculate window half-width based on observed signal extent in data
        # Find where signal drops significantly below background (dip region)
        half_depth = (background_est + min_signal) / 2.0  # Midpoint between background and dip
        signal_indices = np.where(ys < half_depth)[0]

        # Domain width needed for coordinate conversions
        domain_width = self._domain_hi - self._domain_lo

        if len(signal_indices) >= 2:
            # Signal width from data: distance from min to furthest significant point
            signal_xs = xs[signal_indices]
            left_extent = best_point_norm - float(np.min(signal_xs))
            right_extent = float(np.max(signal_xs)) - best_point_norm
            # Half-width includes observed extent plus 50% margin on each side
            half_w = max(left_extent, right_extent) * 1.5
        else:
            # Fallback: use model guidance or reasonable default
            max_span = self._signal_max_span if self._signal_max_span is not None else self._model_signal_max_span()
            if max_span is not None and domain_width > 0:
                half_w = float(max_span / 2.0 / domain_width)
            else:
                half_w = 0.15  # 15% default

        # Clamp to reasonable bounds: at least 5% of domain, at most 40%
        half_w = float(np.clip(half_w, 0.05, 0.40))

        # Ensure we don't exceed domain boundaries
        left_space = best_point_norm
        right_space = 1.0 - best_point_norm
        half_w = min(half_w, left_space, right_space)
        lo_norm = best_point_norm - half_w
        hi_norm = best_point_norm + half_w

        self._acquisition_lo = self._domain_lo + lo_norm * domain_width
        self._acquisition_hi = self._domain_lo + hi_norm * domain_width

        # Delegate to subclass for point regeneration
        self._regenerate_points(refocus_step, lo_norm, hi_norm)

    def _check_early_stop(self) -> bool:
        """Check if sweep can stop early due to well-characterized signal."""
        if len(self._sweep_observations) < 6:
            return False

        xs = np.array([float(o.x) for o in self._sweep_observations])
        ys = np.array([float(o.signal_value) for o in self._sweep_observations])

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

    def _expected_dip_count_from_model(self) -> int:
        """Return expected number of dips from the signal model.

        Delegates to the model's expected_dip_count() method which knows
        its own structure (1, 2, or 3 dips based on physics).
        """
        inner = self._inner_model()
        return inner.expected_dip_count()

    def _set_acquisition_window(self) -> None:
        """Set acquisition window from sweep observations."""
        if len(self._sweep_observations) < 3:
            # No signal found - check if we should do fallback sweep
            if not self._fallback_done and self.max_steps > 0:
                self._fallback_done = True
                self._sweep_points = sobol_1d_sequence(self.max_steps, offset=0.5)
                self._sweep_observations.clear()
                self.step_count = 0
                self._last_refocus_step = 0
                self._completed_at_step = self.max_steps
                return  # Will restart sweep on next call

            # No fallback or already done
            self._acquisition_lo = self._domain_lo
            self._acquisition_hi = self._domain_hi
            return

        # Try to detect signal region
        xs = np.array([float(o.x) for o in self._sweep_observations], dtype=float)
        ys = np.array([float(o.signal_value) for o in self._sweep_observations], dtype=float)

        span = self._detect_signal_span(xs, ys)
        if span is None:
            if not self._fallback_done and self.max_steps > 0:
                self._fallback_done = True
                self._sweep_points = sobol_1d_sequence(self.max_steps, offset=0.5)
                self._sweep_observations.clear()
                self.step_count = 0
                self._last_refocus_step = 0
                self._completed_at_step = self.max_steps
                return

            self._acquisition_lo = self._domain_lo
            self._acquisition_hi = self._domain_hi
            return

        lo_norm, hi_norm = span
        domain_width = self._domain_hi - self._domain_lo
        self._acquisition_lo = self._domain_lo + lo_norm * domain_width
        self._acquisition_hi = self._domain_lo + hi_norm * domain_width
        self._signal_found = True

    def _detect_signal_span(
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
            seg_xs = kept_xs[split_indices[i]:split_indices[i + 1]]
            if len(seg_xs) >= 2:
                lo, hi = float(seg_xs[0]), float(seg_xs[-1])
                peak_val = float(np.min(ys_sorted[keep][split_indices[i]:split_indices[i + 1]]))
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


class StagedSobolLocator(SweepingLocator):
    """Staged Sobol locator with powers-of-2 refocusing.

    Performs an initial sweep using a Sobol sequence, refocusing the remaining
    sweep points at powers of 2 (2, 4, 8, 16...) when a signal is detected.
    The powers-of-2 refocusing aligns with Sobol sequence stages for optimal
    coverage.
    """

    @classmethod
    def create(
        cls,
        belief: AbstractMarginalDistribution | None = None,
        max_steps: int = 24,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ) -> StagedSobolLocator:
        """Factory method to create a StagedSobolLocator.

        When belief is None, creates a dummy belief for compatibility.
        In practice, this is called with a real belief from SequentialBayesianLocator.
        """
        if belief is None:
            # Create minimal belief for factory pattern compliance
            model = BlackBoxSignalModel()
            belief = GridMarginalDistribution(
                model=model,
                parameters=[
                    GridParameter(
                        name="peak_x",
                        bounds=(0.0, 1.0),
                        grid=np.linspace(0.0, 1.0, 128),
                        posterior=np.ones(128) / 128,
                    ),
                ],
            )
        return cls(
            belief=belief,
            max_steps=max_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        max_steps: int,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ):
        super().__init__(
            belief=belief,
            max_steps=max_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )
        # Generate initial Sobol points
        self._sweep_points = self._generate_sweep_points(max_steps)

    def _generate_sweep_points(self, n: int) -> np.ndarray:
        """Generate n Sobol sequence points in [0, 1]."""
        return sobol_1d_sequence(n)

    def _should_refocus(self, step_count: int) -> int | None:
        """Return next binary-digit refocus step if refocusing should occur.

        Refocusing happens when the Sobol sequence reaches deeper binary digits,
        making the space denser. First refocus at 2^8=256 (8th binary digit),
        then at each additional digit (512, 1024, 2048...).
        """
        # First refocus at 2^8 = 256 (8th binary digit)
        if self._last_refocus_step == 0:
            next_refocus = 256
        else:
            next_refocus = self._last_refocus_step * 2

        if step_count >= next_refocus and next_refocus <= self.max_steps:
            return next_refocus
        return None

    def _regenerate_points(self, refocus_step: int, lo_norm: float, hi_norm: float) -> None:
        """Regenerate remaining sweep points from deeper Sobol stages.

        Instead of generating points only within the window, we generate a deeper
        Sobol sequence (2x the total steps) and keep only those points that fall
        within the focused window. This preserves the low-discrepancy property
        as if we continued the global sweep, but concentrated in the region of interest.
        """
        remaining = self.max_steps - refocus_step
        if remaining <= 0:
            return

        window_width = hi_norm - lo_norm
        if window_width <= 0:
            return

        # Generate deeper Sobol sequence (2x total for selection)
        # This simulates continuing to the next "stage" of the global sequence
        deep_stage_size = self.max_steps * 2
        deep_sobol = sobol_1d_sequence(deep_stage_size)

        # Collect points that fall within the focused window
        window_points: list[float] = []
        for p in deep_sobol:
            if lo_norm <= p <= hi_norm:
                window_points.append(float(p))
            if len(window_points) >= remaining:
                break

        # Fill shortfall with evenly spaced points in the window
        while len(window_points) < remaining:
            frac = len(window_points) / max(1, remaining - 1)
            window_points.append(float(np.clip(lo_norm + frac * window_width, lo_norm, hi_norm)))

        # Store back into sweep schedule
        for i in range(remaining):
            idx = refocus_step + i
            if idx < len(self._sweep_points):
                self._sweep_points[idx] = window_points[i]
