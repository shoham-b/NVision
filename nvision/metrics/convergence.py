"""Logic for detecting and analyzing convergence in localization runs."""

from __future__ import annotations

from typing import Dict, List, Optional

from nvision.metrics.types import ParameterConvergence
from nvision.models.observer import RunResult


def analyze_run_convergence(
    run_result: RunResult,
    uncertainty_threshold: float = 0.01,
    relative: bool = True
) -> Dict[str, ParameterConvergence]:
    """Analyze convergence for all parameters in a single run.

    Args:
        run_result: The result of a single localization run.
        uncertainty_threshold: Threshold below which a parameter is considered converged.
        relative: If True, the threshold is relative to the parameter's initial range.
                 If False, the threshold is absolute.

    Returns:
        A dictionary mapping parameter names to their convergence status.
    """
    param_convergence = {}
    true_signal = run_result.true_signal
    final_snapshot = run_result.snapshots[-1] if run_result.snapshots else None

    if final_snapshot is None:
        return {}

    # Get final estimates and uncertainties
    final_estimates = final_snapshot.belief.estimates()
    final_uncertainties = final_snapshot.belief.uncertainty()

    # Get parameter bounds for relative threshold calculation
    bounds = final_snapshot.belief.physical_param_bounds

    for param in final_estimates:
        true_value = true_signal.get_param_value(param)
        final_err = abs(final_estimates[param] - true_value)
        final_uncert = final_uncertainties[param]

        # Determine effective threshold
        if relative and param in bounds:
            lo, hi = bounds[param]
            effective_threshold = uncertainty_threshold * (hi - lo)
        else:
            effective_threshold = uncertainty_threshold

        # Find convergence step
        conv_step = None
        for i, snapshot in enumerate(run_result.snapshots):
            uncert = snapshot.belief.uncertainty()[param]
            if uncert < effective_threshold:
                # Check if it stayed below threshold (optional: could add stability check)
                conv_step = i
                break

        param_convergence[param] = ParameterConvergence(
            parameter=param,
            converged=conv_step is not None,
            convergence_step=conv_step,
            final_error=final_err,
            final_uncertainty=final_uncert
        )

    return param_convergence


def get_convergence_summary(convergence_map: Dict[str, ParameterConvergence]) -> Dict[str, bool]:
    """Get a simple summary of which parameters converged."""
    return {p: c.converged for p, c in convergence_map.items()}
