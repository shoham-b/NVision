"""Common types and data structures for the metrics package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ParameterConvergence:
    """Convergence status for a specific parameter in a repeat.

    Attributes:
        parameter: Name of the parameter.
        converged: Whether the parameter converged.
        convergence_step: The step at which it converged (None if not converged).
        final_error: Absolute error at the end of the run.
        final_uncertainty: Uncertainty (std) at the end of the run.
    """
    parameter: str
    converged: bool
    convergence_step: Optional[int] = None
    final_error: float = 0.0
    final_uncertainty: float = 0.0


@dataclass(frozen=True)
class RepeatMetrics:
    """Metrics for a single localization repeat.

    Attributes:
        repeat_id: Identifier for the repeat.
        total_steps: Total number of measurements taken.
        parameter_convergence: Detailed convergence info for each parameter.
        stop_reason: Reason why the locator stopped.
    """
    repeat_id: int
    total_steps: int
    parameter_convergence: Dict[str, ParameterConvergence]
    stop_reason: str


@dataclass(frozen=True)
class StrategyMetrics:
    """Aggregated metrics for a strategy across multiple repeats.

    Attributes:
        strategy_name: Name of the strategy.
        n_repeats: Total number of repeats analyzed.
        absolute_errors: List of absolute errors for each parameter across all repeats.
        convergence_steps: List of steps to convergence for each parameter across all repeats.
        convergence_rate: Fraction of repeats where all parameters converged.
        parameter_convergence_rates: Fraction of repeats where each specific parameter converged.
        mean_steps_to_convergence: Average steps to convergence for successful repeats.
    """
    strategy_name: str
    n_repeats: int
    # Parameter name -> List of values (one per repeat)
    absolute_errors: Dict[str, List[float]] = field(default_factory=dict)
    convergence_steps: Dict[str, List[int]] = field(default_factory=dict)

    # Summary statistics
    parameter_convergence_rates: Dict[str, float] = field(default_factory=dict)
    mean_steps_to_convergence: Dict[str, float] = field(default_factory=dict)
    total_convergence_rate: float = 0.0
