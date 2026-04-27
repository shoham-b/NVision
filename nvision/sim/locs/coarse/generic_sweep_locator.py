"""Generic uniform sweep locator — a concrete SweepingLocator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.sim.locs.coarse.sweep_locator import SweepingLocator
from nvision.spectra.signal import SignalModel

if TYPE_CHECKING:
    pass


class GenericSweepLocator(SweepingLocator):
    """Uniform grid sweep locator without refocusing.

    A concrete implementation of SweepingLocator that uses a pure uniform grid
    for the entire sweep. Unlike StagedSobolSweepLocator, this locator does not
    refocus - it maintains uniform sampling density across the full domain.

    Parameters
    ----------
    belief : AbstractMarginalDistribution
        Belief distribution (required by Locator parent class).
    signal_model : SignalModel
        Signal model for sweep detection.
    max_steps : int
        Maximum number of sweep steps.
    noise_std : float, default 0.01
        Estimated measurement noise standard deviation.
    noise_max_dev : float | None, default None
        Pre-computed maximum noise deviation for thresholding.
    signal_min_span : float | None, default None
        Minimum expected signal span for density calculation.
    signal_max_span : float | None, default None
        Maximum expected signal span for window sizing.
    scan_param : str | None, default None
        Parameter name being scanned.
    domain_lo : float, default 0.0
        Domain lower bound in physical units.
    domain_hi : float, default 1.0
        Domain upper bound in physical units.
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
    ) -> GenericSweepLocator:
        """Factory method for creating a GenericSweepLocator.

        Parameters
        ----------
        belief : AbstractMarginalDistribution
            Belief distribution.
        signal_model : SignalModel
            Signal model for sweep detection.
        max_steps : int
            Maximum number of sweep steps.
        noise_std : float, default 0.01
            Estimated measurement noise standard deviation.
        noise_max_dev : float | None, default None
            Pre-computed maximum noise deviation.
        signal_min_span : float | None, default None
            Minimum expected signal span.
        signal_max_span : float | None, default None
            Maximum expected signal span.
        scan_param : str | None, default None
            Parameter name being scanned.
        domain_lo : float, default 0.0
            Domain lower bound.
        domain_hi : float, default 1.0
            Domain upper bound.
        parameter_bounds : dict[str, tuple[float, float]] | None, default None
            Parameter bounds (used to override domain if scan_param is provided).

        Returns
        -------
        GenericSweepLocator
            Configured sweep locator.
        """
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

        # Refocus configuration: refocus at half the sweep to focus remaining
        # points around any detected signal.
        self._refocus_at = max(20, max_steps // 2)

        # Generate initial sweep points
        self._sweep_points = self._generate_sweep_points(max_steps)

    def _generate_sweep_points(self, n: int) -> NDArray[np.float64]:
        """Generate n uniformly spaced sweep points in [0, 1].

        Parameters
        ----------
        n : int
            Number of points to generate.

        Returns
        -------
        NDArray[np.float64]
            Array of n uniform grid points in [0, 1].
        """
        if n <= 0:
            return np.array([], dtype=float)
        return np.linspace(0.0, 1.0, n, dtype=float)

    def _generate_fallback_points(self, n: int) -> NDArray[np.float64]:
        """Generate fallback sweep points with 0.5 offset for coverage.

        Parameters
        ----------
        n : int
            Number of points to generate.

        Returns
        -------
        NDArray[np.float64]
            Offset uniform grid points.
        """
        if n <= 0:
            return np.array([], dtype=float)
        # Offset by half a step to cover gaps from first sweep
        return (np.linspace(0.0, 1.0, n, dtype=float) + 0.5 / n) % 1.0

    def _should_refocus(self, step_count: int) -> int | None:
        """Return refocus step if we just reached the refocus point.

        Triggers once after the initial coarse sweep completes, when enough
        observations exist to detect a signal and refocus the remaining points.

        Parameters
        ----------
        step_count : int
            Current step count (1-indexed).

        Returns
        -------
        int | None
            Refocus step count, or None if already refocused or disabled.
        """
        if self._refocus_at is None or self._last_refocus_step > 0:
            return None
        if step_count > self._refocus_at and self.history.count >= self._refocus_at:
            return self._refocus_at
        return None

    def _regenerate_points(self, refocus_step: int, lo_norm: float, hi_norm: float) -> None:
        """Regenerate remaining sweep points in the focused window.

        Replaces the remaining points in the sweep with denser sampling
        within the detected signal window.

        Parameters
        ----------
        refocus_step : int
            The step at which refocusing occurred.
        lo_norm : float
            Lower bound of focus window in normalized [0, 1] coordinates.
        hi_norm : float
            Upper bound of focus window in normalized [0, 1] coordinates.
        """
        remaining = self.max_steps - refocus_step
        if remaining <= 0:
            return

        # Generate denser points in the focused window
        new_points = np.linspace(lo_norm, hi_norm, remaining, dtype=float)

        # Replace remaining points in the sweep array
        self._sweep_points[refocus_step:] = new_points

    def expected_sweep_steps(self, coverage_factor: float = 3.0) -> int:
        """Calculate expected number of sweep steps needed.

        Formula: (domain_width / signal_span) * num_dips * coverage_factor

        This estimates how many measurement points are needed to adequately
        sample the signal features across the full domain.

        Parameters
        ----------
        coverage_factor : float, default 3.0
            Samples per dip span for adequate coverage (higher = denser sampling).

        Returns
        -------
        int
            Expected number of sweep steps, clamped to reasonable bounds.
        """
        domain_width = self._domain_hi - self._domain_lo
        if domain_width <= 0:
            return self.max_steps

        # Get signal span from model
        signal_span = self._signal_max_span or self._model_signal_max_span()
        if signal_span is None or signal_span <= 0:
            signal_span = domain_width * 0.1  # Default: 10% of domain

        # Get expected dip count from model
        num_dips = self.signal_model.expected_dip_count()

        # Calculate: (domain / span) * dips * coverage
        # domain/span = how many signal spans fit in the domain
        # multiplied by dips and coverage = total samples needed
        expected = (domain_width / signal_span) * num_dips * coverage_factor

        # Clamp to reasonable bounds
        min_steps = max(10, num_dips * 3)  # At least 3 points per dip
        max_steps = min(500, int(domain_width / (signal_span / (num_dips * 10))))
        return max(min_steps, min(int(expected), max_steps, self.max_steps))

    def finalize(self) -> None:
        """Finalize the sweep, trimming acquisition window if needed.

        Delegates to parent class finalize for baseline tail trimming.
        """
        super().finalize()

    def effective_initial_sweep_steps(self) -> int:
        """Return effective initial sweep steps for UI phase coloring.

        For GenericSweepLocator, ALL measurements are part of the sweep
        (there is no separate inference/locator phase), so this returns
        the total step count.

        Returns
        -------
        int
            Total measurements taken (all are sweep measurements).
        """
        return self.effective_step_count()

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        """Return None — uniform sweep locators do not narrow the window."""
        return None
