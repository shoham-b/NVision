"""Logic for detecting parameter-specific milestones in localization runs."""

from __future__ import annotations

import math
from typing import Any

from nvision.models.observer import RunResult
from nvision.sim.defaults import NVISION_FREQ_CONVERGENCE_THRESHOLD


def detect_milestone(run_result: RunResult, param: str, threshold: float = 0.01, relative: bool = True) -> int | None:
    """Find the first step where a parameter's uncertainty drops below a threshold.

    Args:
        run_result: Full trajectory of the run.
        param: Parameter name (e.g., 'frequency').
        threshold: Uncertainty threshold.
        relative: If True, threshold is relative to initial parameter range.

    Returns:
        Step index (0-indexed) or None if never converged.
    """
    if not run_result.snapshots:
        return None

    if param == "frequency":
        # For frequency, convergence threshold is set via environment variable
        threshold = NVISION_FREQ_CONVERGENCE_THRESHOLD

    elif relative:
        # Get bounds for relative threshold
        bounds = run_result.snapshots[0].belief.physical_param_bounds.get(param)
        if bounds:
            lo, hi = bounds
            threshold = threshold * (hi - lo)

    for i, snapshot in enumerate(run_result.snapshots):
        uncert = snapshot.belief.uncertainty().get(param)
        if uncert is not None and uncert < threshold:
            return i

    return None


def extract_milestone_metrics(
    run_result: RunResult, step_idx: int, fb_param: str = "frequency", fc_param: str = "split"
) -> dict[str, Any]:
    """Extract estimates and errors at a specific step milestone.

    Returns a dictionary of metrics at that step.
    """
    if step_idx >= len(run_result.snapshots):
        return {}

    snapshot = run_result.snapshots[step_idx]
    estimates = snapshot.belief.estimates()
    uncertainties = snapshot.belief.uncertainty()

    true_fb = run_result.true_signal.get_param_value(fb_param)
    true_fc = run_result.true_signal.get_param_value(fc_param)

    est_fb = estimates.get(fb_param, math.nan)
    est_fc = estimates.get(fc_param, math.nan)

    # Calculate overall uncertainty (mean of all parameters)
    overall_uncert = float(sum(uncertainties.values()) / len(uncertainties)) if uncertainties else math.nan

    return {
        "step": step_idx + 1,  # 1-indexed for display
        "est_fb": est_fb,
        "est_fc": est_fc,
        "err_fb": abs(est_fb - true_fb) if not math.isnan(est_fb) else math.nan,
        "err_fc": abs(est_fc - true_fc) if not math.isnan(est_fc) else math.nan,
        "overall_uncert": overall_uncert,
    }


def calculate_zeeman_metrics(
    run_result: RunResult, threshold: float = 0.01, fb_param: str = "frequency", fc_param: str = "split"
) -> dict[str, Any]:
    """Compare the fb milestone to the final state.

    Returns aggregated metrics for the repeat.
    """
    # 1. FB Milestone
    fb_idx = detect_milestone(run_result, fb_param, threshold)

    metrics: dict[str, Any] = {}

    if fb_idx is not None:
        ms = extract_milestone_metrics(run_result, fb_idx, fb_param, fc_param)
        metrics.update(
            {
                "steps_to_fb": ms["step"],
                "err_fb_at_milestone": ms["err_fb"],
                "err_fc_at_milestone": ms["err_fc"],
                "fc_at_milestone": ms["est_fc"],
                "overall_uncert_at_milestone": ms["overall_uncert"],
            }
        )
    else:
        metrics.update(
            {
                "steps_to_fb": None,
                "err_fb_at_milestone": None,
                "err_fc_at_milestone": None,
                "fc_at_milestone": None,
                "overall_uncert_at_milestone": None,
            }
        )

    # 2. Final state
    final_idx = len(run_result.snapshots) - 1
    if final_idx >= 0:
        fs = extract_milestone_metrics(run_result, final_idx, fb_param, fc_param)
        metrics.update(
            {
                "final_err_fb": fs["err_fb"],
                "final_err_fc": fs["err_fc"],
                "final_overall_uncert": fs["overall_uncert"],
                "final_steps": fs["step"],
            }
        )

        # 3. Deltas
        if fb_idx is not None:
            metrics["err_fb_diff"] = metrics["err_fb_at_milestone"] - fs["err_fb"]
            metrics["err_fc_diff"] = metrics["err_fc_at_milestone"] - fs["err_fc"]

    return metrics
