"""Observer pattern for tracking localization runs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.locator import Locator
from nvision.models.observation import Observation
from nvision.spectra.signal import TrueSignal


@dataclass
class StepSnapshot:
    """Snapshot of locator state at one step.

    Attributes
    ----------
    obs : Observation
        The observation taken at this step
    belief : BeliefSignal
        Snapshot of posterior at this step
    true_signal : TrueSignal
        Ground truth for error computation
    narrowed_param_bounds : dict[str, tuple[float, float]] | None
        Current narrowed parameter bounds at this step (for dynamic UI updates)
    """

    obs: Observation
    belief: AbstractMarginalDistribution
    true_signal: TrueSignal
    narrowed_param_bounds: dict[str, tuple[float, float]] | None = None


@dataclass
class RunResult:
    """Full result of one localization run with trajectory data.

    Exposes convergence trajectories as first-class methods.
    Single source of truth between Observer and viz/gui.

    Attributes
    ----------
    snapshots : list[StepSnapshot]
        Snapshot at each step of the run
    true_signal : TrueSignal
        Ground truth signal
    focus_window : tuple[float, float] | None
        Physical ``(lo, hi)`` interval where Bayesian acquisition searches after the
        initial sweep — same as the locator's acquisition bounds when the locator
        implements ``bayesian_focus_window()`` (see ``SequentialBayesianLocator``).
    per_dip_windows : list[tuple[float, float]] | None
        Individual per-dip focus windows for multi-dip signals (e.g., NV center triplets).
        Each tuple is ``(lo, hi)`` in physical units. None when using single window.
    """

    snapshots: list[StepSnapshot]
    true_signal: TrueSignal
    focus_window: tuple[float, float] | None = None
    per_dip_windows: list[tuple[float, float]] | None = None
    narrowed_param_bounds: dict[str, tuple[float, float]] | None = None
    sweep_steps: int = 0
    secondary_sweep_steps: int = 0

    def uncertainty_trajectory(self, param: str) -> list[float]:
        """Get uncertainty (std) trajectory for parameter.

        Parameters
        ----------
        param : str
            Parameter name

        Returns
        -------
        list[float]
            Uncertainty at each step
        """
        return [s.belief.uncertainty()[param] for s in self.snapshots]

    def estimate_trajectory(self, param: str) -> list[float]:
        """Get parameter estimate trajectory.

        Parameters
        ----------
        param : str
            Parameter name

        Returns
        -------
        list[float]
            Estimate (posterior mean) at each step
        """
        return [s.belief.estimates()[param] for s in self.snapshots]

    def error_trajectory(self, param: str) -> list[float]:
        """Get absolute error trajectory for parameter.

        Parameters
        ----------
        param : str
            Parameter name

        Returns
        -------
        list[float]
            Absolute error at each step
        """
        true_value = self.true_signal.get_param_value(param)
        return [abs(s.belief.estimates()[param] - true_value) for s in self.snapshots]

    def entropy_trajectory(self) -> list[float]:
        """Get total entropy trajectory across all parameters.

        Returns
        -------
        list[float]
            Total entropy at each step
        """
        return [s.belief.entropy() for s in self.snapshots]

    def signal_trajectory(self, x: float) -> list[float]:
        """Get predicted signal trajectory at position x.

        Parameters
        ----------
        x : float
            Position to evaluate

        Returns
        -------
        list[float]
            Predicted signal value at x for each step
        """
        return [s.belief(x) for s in self.snapshots]

    def convergence_step(self, param: str, threshold: float) -> int | None:
        """Find first step where parameter converged below threshold.

        Parameters
        ----------
        param : str
            Parameter name
        threshold : float
            Uncertainty threshold for convergence

        Returns
        -------
        int | None
            Step index where converged, or None if never converged
        """
        for i, s in enumerate(self.snapshots):
            if s.belief.uncertainty()[param] < threshold:
                return i
        return None

    def final_estimates(self) -> dict[str, float]:
        """Get final parameter estimates.

        Returns
        -------
        dict[str, float]
            Final parameter estimates from last snapshot
        """
        if not self.snapshots:
            return {}
        return self.snapshots[-1].belief.estimates()

    def num_steps(self) -> int:
        """Get total number of steps in this run.

        Returns
        -------
        int
            Number of observations taken
        """
        return len(self.snapshots)


class Observer:
    """Watches a localization run and builds trajectory data.

    Gets both BeliefSignal and TrueSignal at each step.
    Builds RunResult with full trajectory data for downstream analysis.

    Attributes
    ----------
    true_signal : TrueSignal
        Ground truth signal
    snapshots : list[StepSnapshot]
        Accumulated snapshots during run
    x_min : float
        Physical domain minimum (for error computation)
    x_max : float
        Physical domain maximum (for error computation)
    """

    def __init__(self, true_signal: TrueSignal, x_min: float, x_max: float):
        """Initialize observer.

        Parameters
        ----------
        true_signal : TrueSignal
            Ground truth signal for error computation
        x_min : float
            Physical domain minimum
        x_max : float
            Physical domain maximum
        """
        self.true_signal = true_signal
        self.x_min = x_min
        self.x_max = x_max
        self.snapshots: list[StepSnapshot] = []

    def watch(self, runner: Iterator[Locator]) -> RunResult:
        """Watch a run and accumulate snapshots.

        Parameters
        ----------
        runner : Iterator[Locator]
            Generator yielding locator state at each step

        Returns
        -------
        RunResult
            Complete result with trajectory data
        """
        self.snapshots = []
        self.last_locator: Locator | None = None

        last_locator: Locator | None = None
        for locator in runner:
            last_locator = locator
            # Snapshot the current state
            # Make deep copy of belief to preserve state at this step
            if locator.belief.last_obs is not None:
                # Capture current narrowed bounds if the locator supports it
                bounds_getter = getattr(locator, "narrowed_param_bounds", None)
                current_bounds = None
                if callable(bounds_getter):
                    nb = bounds_getter()
                    if nb:
                        current_bounds = nb

                snapshot = StepSnapshot(
                    obs=locator.belief.last_obs,
                    belief=locator.belief.copy(),
                    true_signal=self.true_signal,
                    narrowed_param_bounds=current_bounds,
                )
                self.snapshots.append(snapshot)

        self.last_locator = last_locator
        focus_window: tuple[float, float] | None = None
        per_dip_windows: list[tuple[float, float]] | None = None
        narrowed_param_bounds: dict[str, tuple[float, float]] | None = None
        sweep_steps = 0
        secondary_sweep_steps = 0
        if last_locator is not None:
            # Use duck typing with hasattr for optional locator capabilities
            if hasattr(last_locator, "bayesian_focus_window"):
                focus_window = last_locator.bayesian_focus_window()
            if hasattr(last_locator, "per_dip_windows"):
                per_dip_windows = last_locator.per_dip_windows()
            if hasattr(last_locator, "narrowed_param_bounds"):
                nb = last_locator.narrowed_param_bounds()
                if nb:
                    narrowed_param_bounds = nb
            # Capture sweep step counts for phase coloring in UI
            # Use effective_initial_sweep_steps to account for any fallback sweep
            if hasattr(last_locator, "effective_initial_sweep_steps"):
                sweep_steps = last_locator.effective_initial_sweep_steps()
            elif hasattr(last_locator, "initial_sweep_steps"):
                sweep_steps = last_locator.initial_sweep_steps
            if hasattr(last_locator, "secondary_sweep_count"):
                secondary_sweep_steps = last_locator.secondary_sweep_count()

        return RunResult(
            snapshots=self.snapshots,
            true_signal=self.true_signal,
            focus_window=focus_window,
            per_dip_windows=per_dip_windows,
            narrowed_param_bounds=narrowed_param_bounds,
            sweep_steps=sweep_steps,
            secondary_sweep_steps=secondary_sweep_steps,
        )
