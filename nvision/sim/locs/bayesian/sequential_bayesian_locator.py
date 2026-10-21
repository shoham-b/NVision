"""Abstract base class for all Bayesian locators."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.belief.focus_window import FocusRegion
from nvision.belief.smc_marginal import SMCMarginalDistribution, _inverse_sum_squares
from nvision.metrics.milestones import (  # noqa: F401  (re-exported for callers and tests)
    DRIVE_RESPONSE_PARAMS,
    belief_target_params_converged,
    resolve_primary_param,
    saturation_voigt_derived_bounds,
    saturation_voigt_derived_sigmas,
)
from nvision.models.locator import Locator
from nvision.models.observation import Observation
from nvision.sim.defaults import (
    NVISION_CONVERGENCE_THRESHOLD,
)

# Minimum number of consecutive converged checks before declaring convergence.
# Prevents false early stops on the first measurement, especially with no noise.
NVISION_CONVERGENCE_PATIENCE: int = int(os.getenv("NVISION_CONVERGENCE_PATIENCE", "8"))


class ConvergenceStopRules:
    """The stopping rules shared by every locator that infers with an SMC belief (SBED family and the sweeps).

    Needs ``self.belief`` and ``self.step_count``; call :meth:`_init_stop_rules` once, then
    :meth:`_update_stop_rules` after each observation and read ``_is_converged``.
    """

    def _init_stop_rules(self, convergence_threshold: float, convergence_patience_steps: int) -> None:
        """Set the state the stopping rules keep (requires ``self.belief``)."""
        self.convergence_threshold = convergence_threshold
        self._convergence_patience_steps = max(1, int(convergence_patience_steps))
        self._convergence_streak = 0
        # The primary parameter (see resolve_primary_param) --
        # zeeman_shift/hyperfine_shift when the model has one (the actual free/scientific-interest quantity
        # once zero_field_freq is fixed by default), else zero_field_freq itself for legacy free-zero_field_freq
        # configs. See resolve_primary_param.
        self._primary_param: str | None = resolve_primary_param(self.belief.model.parameter_names())
        self._is_converged: bool = False

    def _update_stop_rules(self) -> None:
        """Run the stopping rules once, after the belief has absorbed the latest measurement.

        Sets ``_is_converged`` once every target parameter's uncertainty has stayed under its ceiling for
        ``_convergence_patience_steps`` consecutive checks, *and* the data show a dip (the focus is found,
        ``belief.current_dips()`` -- the rule the stage metrics use). The belief's uncertainty is overconfident, so
        without the dip requirement a run could stop having never located the resonance. This is the only
        stopping rule besides the step budget.
        """
        # The raw belief uncertainty -- the same measure the convergence milestones are read from
        # (nvision/metrics/milestones.py), so the stop rule and the recorded milestones can never disagree.
        uncertainties = self.belief.uncertainty()
        # The dip scan only runs once the uncertainty is already under its ceiling. Only the SMC belief models dips
        # (a grid belief has no focus to find), so the requirement applies to it alone.
        if self._target_params_converged(uncertainties) and (
            not isinstance(self.belief, SMCMarginalDistribution) or self.belief.current_dips()
        ):
            self._convergence_streak += 1
            if self._convergence_streak >= self._convergence_patience_steps:
                self._is_converged = True
        else:
            self._convergence_streak = 0

    def _target_params_converged(self, physical_uncertainties=None) -> bool:
        """Whether every target parameter's uncertainty is under its ceiling (see :func:`target_params_converged`).

        ``physical_uncertainties`` may be passed in to avoid recomputing the full O(particles x params) pass.
        """
        return belief_target_params_converged(self.belief, physical_uncertainties, self.convergence_threshold)


class SequentialBayesianLocator(ConvergenceStopRules, Locator):
    """Shared Bayesian loop infrastructure for all acquisition strategies.

    Handles the mechanics common to every Bayesian locator:
    - incrementing the step counter
    - convergence-based stopping
    - extracting results from the belief posterior

    Subclasses must implement:
    - ``create(**config)`` — build the model-specific ``BeliefSignal`` with priors.
    - ``_acquire()``       — select the next measurement position from the current belief.

    The only behavioral difference between Bayesian strategies is *how* the
    next measurement position is chosen.  All other wiring is identical and
    lives here so it never needs to be repeated.

    Parameters
    ----------
    belief : AbstractMarginalDistribution
        Initial belief (usually a flat / uniform prior over all parameters).
    max_steps : int
        Hard upper bound on Bayesian inference steps.
    convergence_threshold : float
        Relative uncertainty threshold (fraction of bound width) below which
        we consider parameters converged and stop early.  Default ``0.01`` = 1 %.
    center_param : str | None
        The parameter we are proposing measurements along. Defaults to the
        first parameter in the belief.
    """

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        max_steps: int = 450,
        convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
        center_param: str | None = None,
        convergence_patience_steps: int = 8,
    ) -> None:
        super().__init__(belief)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        # Total measurement count
        self.step_count: int = 0
        # Bayesian acquisition count
        self.inference_step_count: int = 0
        # "zero_field_freq" is always the drive-frequency axis even when it's a fixed (not
        # inferred) model parameter and therefore absent from parameter_names().
        self._center_param = center_param or (
            "zero_field_freq"
            if "zero_field_freq" in belief.physical_param_bounds
            else belief.model.parameter_names()[0]
        )
        self._init_stop_rules(convergence_threshold, convergence_patience_steps)

        self._true_signal = None

        # Set domain bounds for acquisition.
        lo_phys, hi_phys = self.belief.drive_freq_bounds_phys
        self._drive_freq_min_phys, self._drive_freq_max_phys = float(lo_phys), float(hi_phys)
        # Which sections of the drive-frequency axis candidates may be drawn from. Owned here (never by the
        # belief): narrowing it changes only where we scan, not the belief's parameter bounds or particles.
        self._focus = FocusRegion.full(self._drive_freq_min_phys, self._drive_freq_max_phys)

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        max_steps: int = 150,
        convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
        center_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        convergence_patience_steps: int = 8,
        **grid_config: object,
    ) -> SequentialBayesianLocator:
        """Generic factory for model-agnostic Bayesian locators.

        Subclasses for specific models (like NVCenterBayesianLocator) should
        override this to provide their own hard-coded belief setup.
        """
        if builder is None:
            raise ValueError(
                f"{cls.__name__} requires a 'builder' callable to create the AbstractMarginalDistribution."
            )
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            center_param=center_param,
            convergence_patience_steps=convergence_patience_steps,
        )

    # ------------------------------------------------------------------
    # SMC lifecycle hooks — subclasses may override any stage
    # ------------------------------------------------------------------

    def _acquire(self) -> float:
        """Acquisition logic must be implemented by subclasses.

        Returns
        -------
        float
            Position in **physical units** to measure next. The base class
            will automatically normalize this to [0, 1] for the experiment.
        """
        raise NotImplementedError("Subclasses must implement _acquire()")

    def _observe_acquisition(self, obs: Observation) -> None:
        """Route acquisition observations to the belief immediately."""
        self.belief.update(obs)
        self._check_convergence_milestones()

    def _acquisition_done(self) -> bool:
        """Return whether the Bayesian acquisition phase should stop.

        Stops when:
        1. Target parameters have stayed converged for ``convergence_patience_steps``
           (``_target_params_converged``,
           both of which require several *consecutive* or empirically-grounded
           confirmations, not a single snapshot).
        2. The absolute ``max_steps`` budget is exhausted.

        This used to also stop early whenever a dynamic budget estimate
        (``n_req = step_count * (crlb_zero_field_freq()/threshold)**2``, scaled by
        ``NVISION_ZERO_FIELD_FREQ_CRLB_SAFETY_FACTOR``) said the target precision was
        unreachable, or already reached, within budget. Both the "unreachable"
        and "reached early" branches trusted a single-step snapshot of
        ``crlb_zero_field_freq()`` -- which is derived from the belief's *current*
        (possibly still-wrong, possibly still-degenerate) parameter estimates,
        not the true signal. Verified two concrete failure modes against a
        width-degenerate Voigt signal (hwhm_lorentz/sigma_inhom don't
        converge on their own): (a) a pessimistic snapshot early on quit after a
        single measurement even though zero_field_freq kept improving for the entire
        step budget when forced to keep running; (b) the belief locked onto a
        wrong mode whose *own* narrow-looking CRLB snapshot satisfied the
        "budget reached" branch at a fraction of max_steps, reporting a
        confidently wrong zero_field_freq tens of MHz off. Both are the same
        underlying risk the convergence-patience streak
        already guards against for the "declare
        converged" path -- removing this second, snapshot-only stopping path
        rather than trying to further patch its triggering condition.
        """
        return self._is_converged or self.inference_step_count >= self.max_steps

    # ------------------------------------------------------------------
    # Locator interface — thin orchestrators that delegate to hooks above
    # ------------------------------------------------------------------

    def next(self) -> float:
        """Propose the next drive frequency as ``drive_freq_unit`` (``_acquire`` works in Hz)."""
        self.step_count += 1
        self.inference_step_count += 1

        physical_value = self._acquire()
        return self._to_experiment_normalized(physical_value)

    def observe(self, obs: Observation) -> None:
        """Route observation to the appropriate lifecycle hook."""
        self._observe_acquisition(obs)

    def done(self) -> bool:
        """Stop when converged or step budget is exhausted."""
        return self._acquisition_done()

    def result(self) -> dict[str, float]:
        """Return posterior-mean estimates for all parameters."""
        return dict(self.belief.estimates())

    def _resample_if_degenerate(self) -> bool:
        """Resample when the effective sample size has fallen below the belief's threshold; True if it did."""
        belief = self.belief
        if _inverse_sum_squares(belief._weights) < belief.ess_threshold * belief.num_particles:
            belief._resample()
            return True
        return False

    # ------------------------------------------------------------------
    # Utility helpers available to all acquisition implementations
    # ------------------------------------------------------------------

    def _to_experiment_normalized(self, physical_value: float) -> float:
        """Map a physical scan position to ``[0, 1]`` for :meth:`CoreExperiment.measure`."""
        lo, hi = self._drive_freq_min_phys, self._drive_freq_max_phys
        tol = 1e-7 * (hi - lo)
        if physical_value < lo - tol or physical_value > hi + tol:
            raise ValueError(
                f"Proposed physical scan position {physical_value} is outside the full domain bounds {(lo, hi)}"
            )
        unit_val = (physical_value - lo) / (hi - lo)
        return float(np.clip(unit_val, 0.0, 1.0))
