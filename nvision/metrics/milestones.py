"""Logic for detecting parameter-specific milestones in localization runs."""

from __future__ import annotations

import math
from typing import Any

from nvision.models.observer import RunResult
from nvision.sim.defaults import NVISION_CONVERGENCE_THRESHOLD, PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS


def detect_milestone(
    run_result: RunResult,
    param: str,
    threshold: float = NVISION_CONVERGENCE_THRESHOLD,
    relative: bool = True,
) -> int | None:
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

    if param in PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS:
        threshold = PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS[param]
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


def default_fc_param(run_result: RunResult) -> str:
    """Splitting parameter for the fc milestone: ``zeeman_split`` when the model has it, else ``split``.

    The default NV model is Zeeman-split (parameter ``zeeman_split``); hyperfine models
    expose ``split``. Hardcoding ``split`` silently yields all-NaN fc metrics for Zeeman runs.
    """
    try:
        params = run_result.true_signal.parameter_values()
    except Exception:
        return "split"
    return "zeeman_split" if "zeeman_split" in params else "split"


def extract_milestone_metrics(
    run_result: RunResult,
    step_idx: int,
    fb_param: str = "frequency",
    fc_param: str = "split",
    override_estimates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract estimates and errors at a specific step milestone.

    ``override_estimates``, when given, is merged on top of the belief's raw
    estimates before ``fb_param``/``fc_param`` are read. Pass
    ``run_result.fit_mode_estimates`` here for the final-state milestone: sweep
    locators (``GenericSweepLocator``) defer their belief update to a batch flush
    that can collapse to a garbage marginal (see ``executor.py``), so the actual
    least-squares fit is the only trustworthy estimate at that step.

    Returns a dictionary of metrics at that step.
    """
    if step_idx >= len(run_result.snapshots):
        return {}

    snapshot = run_result.snapshots[step_idx]
    estimates = snapshot.belief.estimates()
    if override_estimates:
        estimates = {**estimates, **override_estimates}
    uncertainties = snapshot.belief.reported_uncertainty()

    param_values = run_result.true_signal.parameter_values()
    true_fb = param_values.get(fb_param, math.nan)
    true_fc = param_values.get(fc_param, math.nan)

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
        "uncert_fb": uncertainties.get(fb_param, math.nan),
        "overall_uncert": overall_uncert,
    }


def calculate_zeeman_metrics(
    run_result: RunResult,
    threshold: float = NVISION_CONVERGENCE_THRESHOLD,
    fb_param: str = "frequency",
    fc_param: str | None = None,
) -> dict[str, Any]:
    """Compare the fb milestone to the final state.

    ``fc_param`` defaults to the model's splitting parameter (``zeeman_split`` when
    present, else ``split``). Returns aggregated metrics for the repeat.
    """
    if fc_param is None:
        fc_param = default_fc_param(run_result)

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
                "fb_at_milestone": ms["est_fb"],
                "fc_at_milestone": ms["est_fc"],
                "uncert_fb_at_milestone": ms["uncert_fb"],
                "overall_uncert_at_milestone": ms["overall_uncert"],
            }
        )
    else:
        metrics.update(
            {
                "steps_to_fb": None,
                "err_fb_at_milestone": None,
                "err_fc_at_milestone": None,
                "fb_at_milestone": None,
                "fc_at_milestone": None,
                "uncert_fb_at_milestone": None,
                "overall_uncert_at_milestone": None,
            }
        )

    # 2. Final state
    final_idx = len(run_result.snapshots) - 1
    if final_idx >= 0:
        fs = extract_milestone_metrics(
            run_result, final_idx, fb_param, fc_param, override_estimates=run_result.fit_mode_estimates
        )
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
