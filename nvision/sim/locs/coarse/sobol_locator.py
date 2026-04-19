"""Sobol-based coarse search locator.

Uses a low-discrepancy Sobol sequence over [0, 1] for needle-in-a-haystack style
global search on mostly-zero signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.sim.locs.coarse.sweep_locator import SweepingLocator
from nvision.spectra.signal import SignalModel

if TYPE_CHECKING:
    pass


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
    ) -> StagedSobolLocator:
        """Factory method to create a StagedSobolLocator.

        Parameters
        ----------
        belief : AbstractMarginalDistribution
            Belief for Locator parent class (not used for sweep detection)
        signal_model : SignalModel
            The signal model to sweep over (used for sweep detection)
        max_steps : int
            Number of sweep points
        noise_std : float
            Measurement noise standard deviation
        noise_max_dev, signal_min_span, signal_max_span : optional
            Detection parameters
        scan_param : str | None
            Parameter name to scan (defaults to first model parameter)
        domain_lo, domain_hi : float
            Sweep domain bounds
        parameter_bounds : dict | None
            Override domain bounds from parameter name
        """
        # Use parameter_bounds to override domain if provided
        if parameter_bounds is not None:
            # Extract bounds for the scan parameter or first parameter
            param_name = scan_param or (signal_model.parameter_names()[0] if signal_model.parameter_names() else "peak_x")
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

    def _generate_fallback_points(self, n: int) -> np.ndarray:
        """Generate fallback sweep points with offset."""
        return sobol_1d_sequence(n, offset=0.5)
