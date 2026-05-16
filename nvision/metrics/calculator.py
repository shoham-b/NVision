"""Main calculator for aggregating metrics across multiple repeats."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from nvision.metrics.convergence import analyze_run_convergence
from nvision.metrics.types import ParameterConvergence, RepeatMetrics, StrategyMetrics
from nvision.models.observer import RunResult

log = logging.getLogger(__name__)


def calculate_strategy_metrics(
    strategy_name: str,
    run_results: Sequence[RunResult],
    uncertainty_threshold: float = 0.01,
    relative: bool = True
) -> StrategyMetrics:
    """Calculate aggregated metrics for a strategy from a sequence of run results.

    Args:
        strategy_name: Name of the strategy being analyzed.
        run_results: Sequence of RunResult objects (one per repeat).
        uncertainty_threshold: Threshold for convergence detection.
        relative: Whether the threshold is relative to parameter range.

    Returns:
        Aggregated StrategyMetrics.
    """
    n_repeats = len(run_results)
    if n_repeats == 0:
        return StrategyMetrics(strategy_name=strategy_name, n_repeats=0)

    # Temporary storage for aggregation
    all_abs_errors: Dict[str, List[float]] = {}
    all_conv_steps: Dict[str, List[int]] = {}
    convergence_counts: Dict[str, int] = {}
    total_conv_steps: Dict[str, int] = {}
    total_fully_converged = 0

    param_names = set()
    if run_results[0].snapshots:
        param_names = set(run_results[0].snapshots[0].belief.estimates().keys())

    for i, run in enumerate(run_results):
        conv_map = analyze_run_convergence(run, uncertainty_threshold, relative)

        fully_converged = True
        for param, conv_info in conv_map.items():
            # Initialize dicts if needed
            if param not in all_abs_errors:
                all_abs_errors[param] = []
                all_conv_steps[param] = []
                convergence_counts[param] = 0
                total_conv_steps[param] = 0

            all_abs_errors[param].append(conv_info.final_error)

            if conv_info.converged:
                all_conv_steps[param].append(conv_info.convergence_step)
                convergence_counts[param] += 1
                total_conv_steps[param] += conv_info.convergence_step
            else:
                fully_converged = False

        if fully_converged and conv_map:
            total_fully_converged += 1

    # Calculate summary stats
    param_conv_rates = {p: count / n_repeats for p, count in convergence_counts.items()}
    mean_conv_steps = {
        p: total_conv_steps[p] / convergence_counts[p] if convergence_counts[p] > 0 else float('nan')
        for p in convergence_counts
    }

    return StrategyMetrics(
        strategy_name=strategy_name,
        n_repeats=n_repeats,
        absolute_errors=all_abs_errors,
        convergence_steps=all_conv_steps,
        parameter_convergence_rates=param_conv_rates,
        mean_steps_to_convergence=mean_conv_steps,
        total_convergence_rate=total_fully_converged / n_repeats
    )


def compute_error_histogram(
    errors: List[float],
    bins: int = 20,
    density: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram of errors for density estimation.

    Returns:
        tuple (hist, bin_edges)
    """
    if not errors:
        return np.array([]), np.array([])
    return np.histogram(errors, bins=bins, density=density)
