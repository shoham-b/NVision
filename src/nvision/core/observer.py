"""Observer pattern for tracking localization runs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from nvision.core.locator import Locator
from nvision.core.observation import Observation
from nvision.core.signal import BeliefSignal, TrueSignal


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
    """

    obs: Observation
    belief: BeliefSignal
    true_signal: TrueSignal


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
    """

    snapshots: list[StepSnapshot]
    true_signal: TrueSignal

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
        true_param = self.true_signal.get_param(param)
        return [abs(s.belief.estimates()[param] - true_param.value) for s in self.snapshots]

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

        for locator in runner:
            # Snapshot the current state
            # Make deep copy of belief to preserve state at this step
            if locator.belief.last_obs is not None:
                snapshot = StepSnapshot(
                    obs=locator.belief.last_obs,
                    belief=locator.belief.copy(),
                    true_signal=self.true_signal,
                )
                self.snapshots.append(snapshot)

        return RunResult(
            snapshots=self.snapshots,
            true_signal=self.true_signal,
        )
