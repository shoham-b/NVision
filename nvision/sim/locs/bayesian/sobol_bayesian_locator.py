"""Simple Sobol Sweep Locator with Bayesian Convergence."""

from __future__ import annotations

from nvision.belief.smc_marginal import _inverse_sum_squares
from nvision.models.observation import Observation
from nvision.sim.defaults import NVISION_CONVERGENCE_THRESHOLD, PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


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
        self.freq_converged_step: int | None = None

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
        )

    def _acquire(self) -> float:
        n = self.inference_step_count
        val = van_der_corput(n, base=2)
        lo, hi = self._scan_lo, self._scan_hi
        return float(lo + val * (hi - lo))

    def _observe_acquisition(self, obs: Observation) -> None:
        super()._observe_acquisition(obs)
        self._check_and_resample(check_convergence=True)

    def _check_and_resample(self, check_convergence: bool = True) -> None:
        if not hasattr(self.belief, "_weights"):
            return

        # Track frequency convergence at every step (absolute threshold from env).
        if self.freq_converged_step is None:
            freq_threshold = PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS.get("frequency")
            if freq_threshold is not None:
                physical_uncertainties = self.belief.uncertainty()
                if "frequency" in physical_uncertainties:
                    unc = float(physical_uncertainties["frequency"])
                    if unc < freq_threshold:
                        self.freq_converged_step = self.step_count

        ess = _inverse_sum_squares(self.belief._weights)
        ess_threshold = getattr(self.belief, "ess_threshold", 0.0) * getattr(self.belief, "num_particles", 0)
        if ess < ess_threshold:
            if hasattr(self.belief, "_resample"):
                self.belief._resample()

        if check_convergence and self._target_params_converged():
            self._is_converged = True

    def _acquisition_done(self) -> bool:
        if self._is_converged:
            return True
        return super()._acquisition_done()
