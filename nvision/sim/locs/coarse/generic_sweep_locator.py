"""Generic uniform sweep locator — a concrete SweepingLocator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.sim.locs.coarse.sweep_locator import SweepingLocator
from nvision.spectra.signal import SignalModel

if TYPE_CHECKING:
    pass


class GenericSweepLocator(SweepingLocator):
    """Uniform grid sweep locator with parabolic (r²) fit.

    Sweeps the full domain uniformly, then fits a quadratic to the region
    around the minimum to estimate the dip center. No refocusing or early
    stopping — every allocated step is used.

    Parameters
    ----------
    belief : AbstractMarginalDistribution
        Belief distribution (required by Locator parent class).
    signal_model : SignalModel
        Signal model for sweep detection.
    max_steps : int
        Number of sweep steps (use all of them).
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
        **kwargs: Any,
    ) -> GenericSweepLocator:
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

        # No refocusing — sweep the entire domain.
        self._refocus_at = None

        # Fit results set by finalize() — reported via result().
        self._freq_estimate_phys: float | None = None
        self._freq_uncert_phys: float | None = None

        # Buffer observations so we can batch-update the belief once at finalize()
        # instead of paying a full Bayesian update on every step.
        self._pending_obs: list = []

        # Generate sweep points across the full domain.
        self._sweep_points = self._generate_sweep_points(max_steps)

    def _generate_sweep_points(self, n: int) -> NDArray[np.float64]:
        if n <= 0:
            return np.array([], dtype=float)
        return np.linspace(0.0, 1.0, n, dtype=float)

    def _generate_fallback_points(self, n: int) -> NDArray[np.float64]:
        if n <= 0:
            return np.array([], dtype=float)
        return (np.linspace(0.0, 1.0, n, dtype=float) + 0.5 / n) % 1.0

    def observe(self, obs) -> None:
        """Record observation without updating the belief on every step.

        The belief is only used for visualization, not for choosing the next
        point (all sweep positions are predetermined). We buffer all observations
        and do a single belief update in finalize() to avoid paying the full
        Bayesian posterior cost N times.
        """
        if self.step_count <= self.max_steps:
            self.history.append(obs)
            self.belief.last_obs = obs
            self._pending_obs.append(obs)

    def _should_refocus(self, step_count: int) -> int | None:
        return None

    def _regenerate_points(self, refocus_step: int, lo_norm: float, hi_norm: float) -> None:
        pass

    def _check_early_stop(self) -> bool:
        return False

    def finalize(self) -> None:
        """Fit a parabola to the dip region, set the acquisition window and frequency estimate."""
        # Flush deferred belief updates — one pass at the end instead of N passes
        # during the sweep (belief is not used for acquisition decisions).
        for obs in self._pending_obs:
            self.belief.update(obs)
        self._pending_obs.clear()

        if self.history.count < 3:
            self._acquisition_lo = self._domain_lo
            self._acquisition_hi = self._domain_hi
            return

        xs = self.history.xs  # normalized [0, 1]
        ys = self.history.ys
        domain_width = self._domain_hi - self._domain_lo

        # Locate the minimum and define a fitting window around it.
        min_idx = int(np.argmin(ys))
        x_min_norm = float(xs[min_idx])

        # Use the signal max-span (or 20 % of domain) to set fitting half-width.
        signal_max_span = self._signal_max_span or self._model_signal_max_span()
        if signal_max_span is not None and domain_width > 0:
            half_width_norm = min(0.5, 1.5 * signal_max_span / domain_width)
        else:
            half_width_norm = 0.2

        lo_fit = max(0.0, x_min_norm - half_width_norm)
        hi_fit = min(1.0, x_min_norm + half_width_norm)
        mask = (xs >= lo_fit) & (xs <= hi_fit)

        if mask.sum() >= 3:
            xs_fit = xs[mask]
            ys_fit = ys[mask]
        else:
            xs_fit = xs
            ys_fit = ys

        # Quadratic (r²) fit: y = a*x² + b*x + c, with covariance for uncertainty.
        try:
            coeffs, cov = np.polyfit(xs_fit, ys_fit, 2, cov=True)
        except (np.linalg.LinAlgError, ValueError):
            coeffs = np.polyfit(xs_fit, ys_fit, 2)
            cov = None

        a, b, _c = coeffs

        if a > 0:
            x0_norm = float(-b / (2.0 * a))
            x0_norm = float(np.clip(x0_norm, 0.0, 1.0))

            # Propagate fit covariance to get σ(x0) in normalized units.
            if cov is not None:
                var_a = float(cov[0, 0])
                var_b = float(cov[1, 1])
                cov_ab = float(cov[0, 1])
                # x0 = -b/(2a)  →  ∂x0/∂a = b/(2a²),  ∂x0/∂b = -1/(2a)
                d_a = b / (2.0 * a**2)
                d_b = -1.0 / (2.0 * a)
                var_x0 = d_a**2 * var_a + d_b**2 * var_b + 2.0 * d_a * d_b * cov_ab
                sigma_x0_norm = float(np.sqrt(max(var_x0, 0.0)))
            else:
                sigma_x0_norm = half_width_norm * 0.1  # rough fallback
        else:
            x0_norm = x_min_norm
            sigma_x0_norm = half_width_norm * 0.5

        # Convert to physical units and store for result().
        if domain_width > 0:
            self._freq_estimate_phys = self._domain_lo + x0_norm * domain_width
            self._freq_uncert_phys = sigma_x0_norm * domain_width
        else:
            self._freq_estimate_phys = self._domain_lo
            self._freq_uncert_phys = None

        # Set acquisition window as ± half_width around the fitted center.
        acq_lo_norm = max(0.0, x0_norm - half_width_norm)
        acq_hi_norm = min(1.0, x0_norm + half_width_norm)

        self._signal_found = True
        self._acquisition_lo = self._domain_lo + acq_lo_norm * domain_width
        self._acquisition_hi = self._domain_lo + acq_hi_norm * domain_width

    def result(self) -> dict[str, float]:
        """Return the parabolic fit estimate alongside the standard sweep result."""
        res = super().result()
        if self._freq_estimate_phys is not None:
            res["frequency"] = self._freq_estimate_phys
        if self._freq_uncert_phys is not None:
            res["uncert"] = self._freq_uncert_phys
        return res

    def effective_initial_sweep_steps(self) -> int:
        return self.effective_step_count()

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        return None
