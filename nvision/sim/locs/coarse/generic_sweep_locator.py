"""Generic uniform sweep locator — a concrete SweepingLocator implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.sim.locs.coarse.curve_fit_estimator import CurveFitEstimator
from nvision.sim.locs.coarse.sweep_locator import SweepingLocator
from nvision.spectra.signal import SignalModel

if TYPE_CHECKING:
    pass

# Deferred observations are flushed to belief.batch_update() in chunks of this
# size at finalize().  batch_update evaluates the model once per chunk via the
# vectorized _many kernel instead of once per observation, while still letting
# the filter resample between chunks (same rationale as SOBOL_BATCH_CHUNK_SIZE).
NVISION_SWEEP_BATCH_CHUNK_SIZE: int = int(os.getenv("NVISION_SWEEP_BATCH_CHUNK_SIZE", "200"))


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
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
        **kwargs: Any,
    ) -> GenericSweepLocator:
        domain_lo = kwargs.get("domain_lo")
        domain_hi = kwargs.get("domain_hi")
        
        if domain_lo is None or domain_hi is None:
            if parameter_bounds is not None:
                # "frequency" is always the probe x-axis for NV-center models even
                # when fixed (not inferred) and therefore absent from
                # signal_model.parameter_names() -- same landmine as
                # SweepingLocator.__init__'s own scan_param default.
                if scan_param:
                    param_name = scan_param
                elif "frequency" in parameter_bounds:
                    param_name = "frequency"
                else:
                    param_name = signal_model.parameter_names()[0] if signal_model.parameter_names() else "peak_x"
                if param_name in parameter_bounds:
                    if domain_lo is None:
                        domain_lo = parameter_bounds[param_name][0]
                    if domain_hi is None:
                        domain_hi = parameter_bounds[param_name][1]
        
        domain_lo = domain_lo if domain_lo is not None else 0.0
        domain_hi = domain_hi if domain_hi is not None else 1.0

        inst = cls(
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
        if parameter_bounds is not None:
            inst._parameter_bounds = dict(parameter_bounds)
        return inst

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

        # Fit results set by finalize() — reported via result().
        self._freq_estimate_phys: float | None = None
        self._freq_uncert_phys: float | None = None

        # Full fitted physical parameter vector from the model fit (set by
        # _fit_model).  Used to draw the actual fit in the visualization
        # instead of the SMC belief marginal mode.
        self._fit_params_phys: dict[str, float] | None = None

        # Buffer observations so we can batch-update the belief once at finalize()
        # instead of paying a full Bayesian update on every step.
        self._pending_obs: list = []

        # Parameter bounds stored by create() when provided as an argument.
        # Used by _fit_model() when signal_model has no param_bounds_phys.
        self._parameter_bounds: dict[str, tuple[float, float]] | None = None

        # Generate sweep points across the full domain.
        self._sweep_points = self._generate_sweep_points(max_steps)

    def _generate_sweep_points(self, n: int) -> NDArray[np.float64]:
        if n <= 0:
            return np.array([], dtype=float)
        return np.linspace(0.0, 1.0, n, dtype=float)

    def _expected_dip_count_from_model(self) -> int:
        """Return expected number of dips from the signal model.

        Delegates to the model's expected_dip_count() method, which knows its
        own structure (1, 2, 3, or 6 dips based on physics).
        """
        return self._inner_model().expected_dip_count()

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

    def _flush_pending_obs(self) -> None:
        """Flush deferred belief updates in vectorized chunks.

        One batched posterior update per chunk instead of a full
        per-observation update N times (the belief is not used for
        acquisition decisions during the sweep).
        """
        if not self._pending_obs:
            return
            
        mapped_obs = []
        is_unit_cube = type(self.belief.model).__name__ == "UnitCubeSignalModel"
        
        if is_unit_cube:
            mapped_obs = self._pending_obs
        else:
            from nvision.models.observation import Observation
            width = self._domain_hi - self._domain_lo
            for o in self._pending_obs:
                x_phys = self._domain_lo + o.x * width
                mapped_obs.append(Observation(
                    x=x_phys,
                    signal_value=o.signal_value,
                    noise_std=o.noise_std,
                    frequency_noise_model=o.frequency_noise_model,
                ))

        if hasattr(self.belief, "batch_update"):
            chunk = NVISION_SWEEP_BATCH_CHUNK_SIZE
            for i in range(0, len(mapped_obs), chunk):
                self.belief.batch_update(mapped_obs[i : i + chunk])
        else:
            for obs in mapped_obs:
                self.belief.update(obs)
        self._pending_obs.clear()

    def _fit_model(self, xs_norm: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
        """Fit the physical model to sweep data via least squares.

        Returns (freq_phys, uncert_phys). Delegates to :class:`CurveFitEstimator`
        (shared with ``StagedSobolSweepLocator``/``SimpleSobolBayesianLocator`` so
        every point-collecting locator reports the same actual least-squares fit
        as its final estimate, instead of each having its own — or no — fit
        logic). Raises if the model can't be fit; a `GenericSweepLocator` sweep
        is only useful *because* of this fit, so a failure here is a real
        problem to surface, not a case to silently paper over with a cruder
        estimate.
        """
        estimator = CurveFitEstimator(
            signal_model=self.signal_model,
            belief=self.belief,
            scan_param=self._scan_param,
            domain_lo=self._domain_lo,
            domain_hi=self._domain_hi,
            noise_std=self._noise_std,
            parameter_bounds=self._parameter_bounds,
            signal_min_span=self._signal_min_span,
        )
        fit_result = estimator.fit(xs_norm, ys)
        self._fit_params_phys = estimator.fit_params_phys
        return fit_result.freq_phys, fit_result.uncert_phys

    def finalize(self) -> None:
        """Fit the physical model to the sweep and report the center frequency.

        Order matters: the fit must run BEFORE the belief flush. The SMC
        belief's batch update can resample and auto-narrow the frequency
        bounds *in place* on the shared ``signal_model.param_bounds_phys``
        (see ``UnitCubeSMCMarginalDistribution.narrow_scan_parameter_physical_bounds``)
        — and a batch-updated belief is collapsed, so it narrows to the wrong
        window using the global RNG. Fitting first keeps the fit anchored to
        the original physical bounds; the flush only exists so visualizations
        can show a belief.

        Raises if the fit fails — see ``_fit_model``.
        """
        domain_width = self._domain_hi - self._domain_lo
        freq_phys, uncert_phys = self._fit_model(self.history.xs, self.history.ys)

        self._flush_pending_obs()

        self._freq_estimate_phys = freq_phys
        self._freq_uncert_phys = uncert_phys
        self._signal_found = True
        signal_max_span = self._signal_max_span or self._model_signal_max_span()
        half_width_phys = (
            min(domain_width / 2, 1.5 * signal_max_span)
            if signal_max_span is not None
            else domain_width * 0.2
        )
        self._acquisition_lo = max(self._domain_lo, freq_phys - half_width_phys)
        self._acquisition_hi = min(self._domain_hi, freq_phys + half_width_phys)

    def result(self) -> dict[str, float]:
        """Return dip-center estimate, uncertainty, and fit-derived sweep metrics."""
        res = super().result()
        if self._freq_estimate_phys is not None:
            res["frequency"] = self._freq_estimate_phys
        if self._freq_uncert_phys is not None:
            res["uncert"] = self._freq_uncert_phys
        res.update(self._compute_sweep_metrics())
        return res

    def _compute_sweep_metrics(self) -> dict[str, float | int]:
        """Sweep efficiency metrics derived from the model fit.

        ``finalize()`` always produces a full least-squares fit (or raises),
        so the dip count and width are known exactly from the model and the
        fit — no need to re-detect dips from noisy sweep y-values the way a
        locator without a fit (e.g. ``StagedSobolSweepLocator``) has to.
        """
        domain_width = self._domain_hi - self._domain_lo
        expected_dips = self._expected_dip_count_from_model()

        dip_width = None
        if self._fit_params_phys is not None:
            dip_width = self._fit_params_phys.get("linewidth") or self._fit_params_phys.get("homogeneous_linewidth")
        dip_width = dip_width or self._model_signal_min_span()

        total_dip_width = expected_dips * dip_width if dip_width is not None and dip_width > 0 else 0.0
        if total_dip_width > 0 and domain_width > 0:
            expected_uniform = 2.0 * domain_width / total_dip_width
        else:
            expected_uniform = float(self.max_steps)
        measurements_done = min(round(expected_uniform), self.max_steps)

        return {
            "measurements_done": measurements_done,
            "dips_detected": expected_dips,
            "total_dip_width": total_dip_width,
            "min_dip_width": dip_width or 0.0,
            "expected_uniform_points": expected_uniform,
            "sweep_efficiency": expected_uniform / max(measurements_done, 1),
        }

    def fit_mode_estimates(self) -> dict[str, float] | None:
        """Full fitted physical parameters from the model fit, or None if unavailable.

        This is the actual least-squares fit of the physical model to the sweep
        data — used by the visualization in preference to the SMC belief marginal
        mode, which is not resampled during the sweep and can collapse.
        """
        return dict(self._fit_params_phys) if self._fit_params_phys is not None else None

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        return None
