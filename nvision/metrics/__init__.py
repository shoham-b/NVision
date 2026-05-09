"""NVision metrics package for strategy performance analysis.

This package provides tools to calculate, aggregate, and analyze metrics
across multiple localization repeats, including convergence tracking
and error distribution analysis.
"""

from nvision.metrics.calculator import (
    calculate_strategy_metrics,
    compute_error_histogram,
)
from nvision.metrics.convergence import (
    analyze_run_convergence,
    get_convergence_summary,
)
from nvision.metrics.types import (
    ParameterConvergence,
    RepeatMetrics,
    StrategyMetrics,
)

__all__ = [
    "calculate_strategy_metrics",
    "compute_error_histogram",
    "analyze_run_convergence",
    "get_convergence_summary",
    "ParameterConvergence",
    "RepeatMetrics",
    "StrategyMetrics",
]
