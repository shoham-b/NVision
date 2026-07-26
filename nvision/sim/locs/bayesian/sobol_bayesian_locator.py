"""Simple Sobol Sweep Locator with Bayesian Convergence."""

from __future__ import annotations

import logging
import os

from nvision.belief.smc_marginal import _inverse_sum_squares
from nvision.models.observation import Observation, ObservationHistory
from nvision.sim.defaults import NVISION_CONVERGENCE_THRESHOLD
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator
from nvision.sim.locs.coarse.curve_fit_estimator import CurveFitEstimator

log = logging.getLogger(__name__)

# Observations are buffered and dispatched to belief.batch_update() every
# SOBOL_BATCH_CHUNK_SIZE steps.  batch_update uses compute_vectorized_many
# (the _many kernel, parallel over probe positions) instead of N sequential
# calls to compute_vectorized (the _one kernel, where thread overhead ≫ work
# at practical particle counts).  Larger chunks amortise thread startup over
# more evaluations; 200 gives ~12x speedup vs sequential at 10k particles
# while still resampling 4–5 times per 920-step run.
SOBOL_BATCH_CHUNK_SIZE: int = int(os.getenv("NVISION_SOBOL_BATCH_CHUNK_SIZE", "200"))


def van_der_corput(n: int, base: int = 2) -> float:
    """Compute the n-th number of the van der Corput sequence in base."""
    vdc = 0.0
    denom = 1.0
    val = n
    while val > 0:
        denom *= base
        val, remainder = divmod(val, base)
        vdc += remainder / denom
    return vdc


class SimpleSobolBayesianLocator(SequentialBayesianLocator):
    """Deterministic van der Corput base-2 sequence scaled to physical bounds.

    Checks Bayesian uncertainty convergence at each step.
    """

    # Tells the executor to inject belief + signal_model automatically,
    # matching the contract of all other Bayesian locators.
    REQUIRES_BELIEF = True

    def __init__(
        self,
        belief,
        max_steps: int = 150,
        convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
        scan_param: str | None = None,
        noise_std: float = 0.02,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        batch_chunk_size: int = SOBOL_BATCH_CHUNK_SIZE,
        signal_model=None,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        super().__init__(
            belief,
            max_steps,
            convergence_threshold,
            scan_param,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
        )
        if hasattr(self.belief, "auto_resample"):
            self.belief.auto_resample = False
        self._is_converged = False
        self._obs_buffer: list[Observation] = []
        self._batch_chunk_size: int = batch_chunk_size
        # Signal model + full observation history, kept only so finalize() can
        # run the same least-squares fit GenericSweepLocator uses (see
        # CurveFitEstimator) as the final estimate -- independent of the van der
        # Corput acquisition / SMC convergence logic above, which stays as-is.
        self.signal_model = signal_model
        self._parameter_bounds = dict(parameter_bounds) if parameter_bounds is not None else None
        self.history = ObservationHistory(max_steps)
        self._fit_params_phys: dict[str, float] | None = None

    @classmethod
    def create(
        cls,
        builder=None,
        max_steps: int = 150,
        convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
        scan_param: str | None = None,
        parameter_bounds=None,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        signal_model=None,
        **grid_config,
    ):
        if builder is None:
            raise ValueError(f"{cls.__name__} requires a 'builder' callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            noise_std=noise_std if noise_std is not None else 0.02,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            signal_model=signal_model,
            parameter_bounds=parameter_bounds,
        )

    def _acquire(self) -> float:
        n = self.inference_step_count
        val = van_der_corput(n, base=2)
        lo, hi = self._scan_lo, self._scan_hi
        return float(lo + val * (hi - lo))

    def _observe_acquisition(self, obs: Observation) -> None:
        """Buffer the observation; belief update is deferred to _flush_buffer."""
        self._obs_buffer.append(obs)
        self.belief.last_obs = obs
        if self.history.count < self.history.max_steps:
            self.history.append(obs)

    def _flush_buffer(self, check_convergence: bool = True) -> None:
        """Dispatch buffered observations to belief.batch_update and resample/check convergence."""
        if not self._obs_buffer:
            return
        self.belief.batch_update(self._obs_buffer)
        self._obs_buffer.clear()
        self._check_and_resample(check_convergence=check_convergence)

    def _check_and_resample(self, check_convergence: bool = True) -> None:
        if not hasattr(self.belief, "_weights"):
            return

        ess = _inverse_sum_squares(self.belief._weights)
        ess_threshold = getattr(self.belief, "ess_threshold", 0.0) * getattr(self.belief, "num_particles", 0)
        if ess < ess_threshold:
            if hasattr(self.belief, "_resample"):
                self.belief._resample()

        if check_convergence:
            # One uncertainty pass shared by the milestone and convergence checks
            # (each belief.uncertainty() call is a full O(particles x params) pass).
            physical_uncertainties = self.belief.uncertainty()
            self._check_convergence_milestones(physical_uncertainties)
            if self._target_params_converged(physical_uncertainties):
                self._is_converged = True

    def done(self) -> bool:
        """Flush pending observations when the chunk is full or budget is exhausted."""
        exhausted = self.inference_step_count >= self.max_steps
        if len(self._obs_buffer) >= self._batch_chunk_size or (self._obs_buffer and exhausted):
            self._flush_buffer(check_convergence=True)
        return self._acquisition_done()

    def finalize(self) -> None:
        """Fit the physical model to every point collected this run, via the
        same least-squares fit ``GenericSweepLocator`` uses (``CurveFitEstimator``).

        The SMC belief driving acquisition above can resample repeatedly on
        sparse, sequentially-revealed data and collapse onto a plausible-but-
        wrong mode (particle impoverishment) -- confidently reporting a tiny
        uncertainty next to a large true error. The fit is a straight
        least-squares read of the actual collected points, independent of that
        resampling history, so it doesn't share that failure mode. Best-effort:
        a fit failure here just leaves the SMC belief's own posterior as the
        final estimate (this locator worked before the fit existed), not a
        raised exception.
        """
        if self.signal_model is None or self.history.count < 2:
            return
        domain_lo, domain_hi = self._full_domain_lo, self._full_domain_hi
        domain_width = domain_hi - domain_lo
        if domain_width <= 0:
            return
        xs_norm = self.history.xs
        ys = self.history.ys
        estimator = CurveFitEstimator(
            signal_model=self.signal_model,
            belief=self.belief,
            scan_param=self._scan_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
            noise_std=self._noise_std,
            parameter_bounds=self._parameter_bounds,
        )
        try:
            estimator.fit(xs_norm, ys)
        except Exception:
            log.warning("SimpleSobolBayesianLocator: curve fit failed, falling back to SMC belief estimate", exc_info=True)
            return
        self._fit_params_phys = estimator.fit_params_phys

    def fit_mode_estimates(self) -> dict[str, float] | None:
        """Full fitted physical parameters from the model fit, or None if unavailable.

        Mirrors ``GenericSweepLocator.fit_mode_estimates()`` -- used in
        preference to the SMC belief marginal mode, which can collapse onto a
        wrong mode under this locator's resampling schedule.
        """
        return dict(self._fit_params_phys) if self._fit_params_phys is not None else None
