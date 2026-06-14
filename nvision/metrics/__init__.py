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
from nvision.metrics.series import (
    effective_convergence_threshold,
    extract_step_series,
)
from nvision.metrics.types import (
    ParameterConvergence,
    RepeatMetrics,
    StrategyMetrics,
)

__all__ = [
    "ParameterConvergence",
    "RepeatMetrics",
    "StrategyMetrics",
    "analyze_run_convergence",
    "calculate_strategy_metrics",
    "compute_error_histogram",
    "effective_convergence_threshold",
    "extract_step_series",
    "get_convergence_summary",
]
