"""Logic for detecting parameter-specific milestones in localization runs."""

from __future__ import annotations

import math
from collections.abc import Iterable
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
        param: Parameter name (e.g., 'center_freq').
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


def default_split_param(run_result: RunResult) -> str:
    """Splitting parameter for the split-error metrics: ``zeeman_split`` when the model has it, else ``split``.

    The default NV model is Zeeman-split (parameter ``zeeman_split``); hyperfine models
    expose ``split``. Hardcoding ``split`` silently yields all-NaN split metrics for Zeeman runs.
    """
    try:
        params = run_result.true_signal.parameter_values()
    except Exception:
        return "split"
    return "zeeman_split" if "zeeman_split" in params else "split"


_PRIMARY_PARAM_PREFERENCE = ("zeeman_split", "split", "center_freq")


def resolve_primary_param(available_params: Iterable[str]) -> str | None:
    """The parameter whose convergence defines the locator's primary milestone.

    Prefers the model's splitting parameter (the actual free/scientific-interest
    quantity once center_freq is fixed by default -- see ``with_fixed_center_freq`` in
    nvision/spectra/nv_center.py), falling back to ``"center_freq"`` itself for legacy
    free-center_freq configurations. ``None`` if the model has neither (milestone
    tracking stays permanently unset, matching historical behavior for such models).
    """
    available = set(available_params)
    for candidate in _PRIMARY_PARAM_PREFERENCE:
        if candidate in available:
            return candidate
    return None


def extract_milestone_metrics(
    run_result: RunResult,
    step_idx: int,
    primary_param: str = "center_freq",
    split_param: str = "split",
    override_estimates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract estimates and errors at a specific step milestone.

    ``override_estimates``, when given, is merged on top of the belief's raw
    estimates before ``primary_param``/``split_param`` are read. Pass
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
    true_primary = param_values.get(primary_param, math.nan)
    true_split = param_values.get(split_param, math.nan)

    est_primary = estimates.get(primary_param, math.nan)
    est_split = estimates.get(split_param, math.nan)

    # Calculate overall uncertainty (mean of all parameters)
    overall_uncert = float(sum(uncertainties.values()) / len(uncertainties)) if uncertainties else math.nan

    return {
        "step": step_idx + 1,  # 1-indexed for display
        "est_primary": est_primary,
        "est_split": est_split,
        "err_primary": abs(est_primary - true_primary) if not math.isnan(est_primary) else math.nan,
        "err_split": abs(est_split - true_split) if not math.isnan(est_split) else math.nan,
        "uncert_primary": uncertainties.get(primary_param, math.nan),
        "overall_uncert": overall_uncert,
    }


def calculate_all_converged_metrics(
    run_result: RunResult,
    all_converged_step: int | None,
    primary_param: str = "center_freq",
    split_param: str | None = None,
) -> dict[str, Any]:
    """Error/uncertainty of the primary parameter at the all-converged milestone.

    ``all_converged_step`` is tracked live by the locator (1-indexed measurement
    count, see ``SequentialBayesianLocator._check_convergence_milestones``) rather
    than re-detected here, since "all converged" depends on every tracked
    parameter's uncertainty, not just ``primary_param`` -- unlike the primary milestone in
    ``calculate_zeeman_metrics``, which re-derives its own step via
    ``detect_milestone``.
    """
    if split_param is None:
        split_param = default_split_param(run_result)

    step_idx = all_converged_step - 1 if all_converged_step is not None else -1
    if step_idx < 0 or step_idx >= len(run_result.snapshots):
        return {
            "err_primary_at_all_converged": None,
            "uncert_primary_at_all_converged": None,
        }

    ms = extract_milestone_metrics(run_result, step_idx, primary_param, split_param)
    return {
        "err_primary_at_all_converged": ms["err_primary"],
        "uncert_primary_at_all_converged": ms["uncert_primary"],
    }


def calculate_zeeman_metrics(
    run_result: RunResult,
    threshold: float = NVISION_CONVERGENCE_THRESHOLD,
    primary_param: str = "center_freq",
    split_param: str | None = None,
) -> dict[str, Any]:
    """Compare the primary-parameter milestone to the final state.

    ``split_param`` defaults to the model's splitting parameter (``zeeman_split`` when
    present, else ``split``). Returns aggregated metrics for the repeat.
    """
    if split_param is None:
        split_param = default_split_param(run_result)

    # 1. Primary-parameter milestone
    primary_idx = detect_milestone(run_result, primary_param, threshold)

    metrics: dict[str, Any] = {}

    if primary_idx is not None:
        ms = extract_milestone_metrics(run_result, primary_idx, primary_param, split_param)
        metrics.update(
            {
                "steps_to_primary": ms["step"],
                "err_primary_at_milestone": ms["err_primary"],
                "err_split_at_milestone": ms["err_split"],
                "primary_at_milestone": ms["est_primary"],
                "split_at_milestone": ms["est_split"],
                "uncert_primary_at_milestone": ms["uncert_primary"],
                "overall_uncert_at_milestone": ms["overall_uncert"],
            }
        )
    else:
        metrics.update(
            {
                "steps_to_primary": None,
                "err_primary_at_milestone": None,
                "err_split_at_milestone": None,
                "primary_at_milestone": None,
                "split_at_milestone": None,
                "uncert_primary_at_milestone": None,
                "overall_uncert_at_milestone": None,
            }
        )

    # 2. Final state
    final_idx = len(run_result.snapshots) - 1
    if final_idx >= 0:
        fs = extract_milestone_metrics(
            run_result, final_idx, primary_param, split_param, override_estimates=run_result.fit_mode_estimates
        )
        metrics.update(
            {
                "final_err_primary": fs["err_primary"],
                "final_err_split": fs["err_split"],
                "final_overall_uncert": fs["overall_uncert"],
                "final_steps": fs["step"],
            }
        )

        # 3. Deltas
        if primary_idx is not None:
            metrics["err_primary_diff"] = metrics["err_primary_at_milestone"] - fs["err_primary"]
            metrics["err_split_diff"] = metrics["err_split_at_milestone"] - fs["err_split"]

    return metrics
