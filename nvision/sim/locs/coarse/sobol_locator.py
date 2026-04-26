"""Sobol-based coarse search locator with 3 stages.

1. 127 points (7th Sobol stage)
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
from nvision.sim.locs.coarse.sweep_locator import SweepingLocator
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
    """Stage 1: Collect exactly 127 points to establish a robust noise baseline."""

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
        return self.points_collected >= 127


class Stage2SobolLocator:
    """Stage 2: Continue scanning while thresholding. Stop upon finding 2 dip points."""

    def __init__(self, sobol_gen: Iterator[float], domain_lo: float, domain_hi: float, history: ObservationHistory):
        self._sobol_gen = sobol_gen
        self.domain_lo = domain_lo
        self.domain_hi = domain_hi
        self.history = history

        self._noise_threshold = -float("inf")
        self._done = False

        # Initialize threshold using the 127 observations inherited from Stage 1
        self._update_noise_threshold()
        self._check_for_dips()

    def next(self) -> float:
        u = next(self._sobol_gen)
        return self.domain_lo + u * (self.domain_hi - self.domain_lo)

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

        # Threshold is defined as 2 stds below the median of the noise points
        self._noise_threshold = noise_median - 2.0 * noise_std

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
        ys_valid = self.history.ys
        xs_valid = self.history.xs

        # Extract top 70% of measurements (discard the bottom 30% which may contain signal dips)
        p30_val = float(np.percentile(ys_valid, 30))
        noise_points = ys_valid[ys_valid >= p30_val]

        noise_median = float(np.median(noise_points))
        noise_std = float(np.std(noise_points))

        # Threshold is defined as 2 stds below the median of the noise points
        noise_threshold = noise_median - 2.0 * noise_std

        below_idx = np.where(ys_valid < noise_threshold)[0]
        if len(below_idx) < 2:
            return  # Safety fallback to domain bounds

        dip_xs = xs_valid[below_idx]
        x_min = float(np.min(dip_xs))
        x_max = float(np.max(dip_xs))
        d = x_max - x_min

        domain_width = self.domain_hi - self.domain_lo
        tol = max(0.015 * domain_width, 1e-4)

        def check_empty(spot: float) -> bool:
            mask = np.abs(xs_valid - spot) < tol
            nearby_ys = ys_valid[mask]
            return len(nearby_ys) > 0 and float(np.min(nearby_ys)) > noise_threshold

        empty_right = check_empty(x_max + d)
        empty_left = check_empty(x_min - d)
        empty_mid = check_empty(x_min + d / 2.0)

        best_left = x_min - d
        best_right = x_max + d

        if empty_mid:
            if empty_right:
                best_left = x_min - d
                best_right = x_max
            elif empty_left:
                best_left = x_min
                best_right = x_max + d
        else:
            if empty_left and empty_right:
                best_left = x_min
                best_right = x_max

        # Ensure minimal padding ensures bounds contain structures
        pad = d * 0.3 if d > 0 else tol * 3
        self.window_lo = max(self.domain_lo, best_left - pad)
        self.window_hi = min(self.domain_hi, best_right + pad)


class StagedSobolSweepLocator(Locator):
    """Orchestrator locator managing the 3-stage targeted Sobol methodology."""

    USES_SWEEP_MAX_STEPS: bool = True
    REQUIRES_BELIEF: bool = True

    @classmethod
    def create(
        cls,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int = 24,
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

        # Track initial sweep steps for Observer phase coloring
        self._initial_sweep_steps = 0

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
            self._stage2 = Stage2SobolLocator(self._sobol_gen, self.domain_lo, self.domain_hi, self.history)
            self._active_locator = self._stage2

            # Cascade: Stage 2 might be instantaneously complete if dips found in Stage 1 data!
            if self._active_locator.done():
                self._transition_to_stage3()

        elif self._active_locator is self._stage2 and self._active_locator.done():
            self._transition_to_stage3()

    def _transition_to_stage3(self) -> None:
        self._initial_sweep_steps = self.step_count
        self._stage3 = Stage3SobolLocator(self._sobol_gen, self.domain_lo, self.domain_hi, self.history)
        self._active_locator = self._stage3
        self._signal_found = True

    def effective_initial_sweep_steps(self) -> int:
        """Return number of steps in the initial sweep (Stage 1 + Stage 2).

        Called by the Observer to determine how many measurements belong to
        the 'coarse' phase for plot coloring.
        """
        if self._initial_sweep_steps > 0:
            return self._initial_sweep_steps
        # If we never reached Stage 3, all completed steps were initial sweep
        return self.step_count

    def finalize(self) -> None:
        pass

    def acquisition_window(self) -> tuple[float, float]:
        if self._stage3 is not None:
            return (self._stage3.window_lo, self._stage3.window_hi)
        return (self.domain_lo, self.domain_hi)

    def result(self) -> dict[str, float | bool]:
        lo, hi = self.acquisition_window()
        return {
            "acquisition_lo": lo,
            "acquisition_hi": hi,
            "signal_found": self._signal_found,
            "completed_at_step": self.step_count,
        }
