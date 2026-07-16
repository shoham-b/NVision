"""Per-repeat metrics extraction."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Any

import polars as pl

from nvision.models.experiment import CoreExperiment
from nvision.models.observer import RunResult
from nvision.tools.math import _first_finite, _maybe_finite, _promote_uncert

log = logging.getLogger(__name__)


def _truth_positions(experiment: CoreExperiment) -> list[float]:
    """Extract ground truth peak positions from a CoreExperiment."""
    values = experiment.true_signal.parameter_values()
    return [value for name, value in values.items() if "frequency" in name or "position" in name]


def generate_attempt_metrics(  # noqa: C901
    n_repeats: int,
    attempt_idx_in_combo: int,
    gen_name: str,
    noise_name: str,
    strat_name: str,
    repeat_stop_reasons: list[str],
    repeat_start_times: list[float],
    repeat_timestamps: list[str],
    current_scan: CoreExperiment,
    final_history_df: pl.DataFrame,
    finalize_results: pl.DataFrame,
    strat_obj: Any,
    max_steps: int | None = None,
    seed: int | None = None,
    run_result: RunResult | None = None,
    generator_obj: Any = None,
) -> tuple[dict[str, Any], dict[str, Any], pl.DataFrame]:
    """Calculate and format metrics for a single repeat.

    Returns
    -------
    tuple
        ``(entry_base, main_result_row, current_history_df)``
    """
    from nvision.metrics.milestones import calculate_zeeman_metrics

    if not final_history_df.is_empty():
        current_history_df = final_history_df.filter(pl.col("repeat_id") == attempt_idx_in_combo).drop("repeat_id")
    else:
        current_history_df = pl.DataFrame(
            {
                "x": pl.Series("x", [], dtype=pl.Float64),
                "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
            }
        )

    if current_history_df.is_empty():
        log.info(
            "No measurements recorded for repeat %s (reason=%s); generating baseline scan plot.",
            attempt_idx_in_combo + 1,
            repeat_stop_reasons[attempt_idx_in_combo],
        )

    finalize_row = finalize_results.filter(pl.col("repeat_id") == attempt_idx_in_combo)
    # Prefer per-repeat duration stored in `locator_results.csv` metadata (written by the executor).
    # Fall back to the legacy `repeat_start_times` timing for backward compatibility.
    duration_ms_value: float | None = None
    if not finalize_row.is_empty() and "duration_ms" in finalize_row.columns:
        duration_ms_value = float(finalize_row.get_column("duration_ms")[0])
    if finalize_row.is_empty() or current_history_df.is_empty():
        estimate: dict[str, float] = {"x_hat": math.nan, "uncert": math.inf}
        measurements = 0
    else:
        estimate_dict = finalize_row.drop("repeat_id").row(0, named=True)
        estimate = {k: float(v) for k, v in estimate_dict.items() if isinstance(v, int | float)}
        measurements = current_history_df.height

    attempt_metrics = _scan_attempt_metrics(_truth_positions(current_scan), estimate)
    metrics_serialized = {key: _maybe_finite(value) for key, value in attempt_metrics.items()}
    metrics_serialized["measurements"] = _maybe_finite(measurements)

    # Forward sweep-locator diagnostic metrics when present
    for _sweep_key in (
        "dips_detected",
        "total_dip_width",
        "min_dip_width",
        "expected_uniform_points",
        "expected_focused_points",
        "sweep_efficiency",
        "measurements_done",
        "acquisition_lo",
        "acquisition_hi",
        "sobol_baseline_steps",
        "sobol_freq_steps",
        "sobol_conv_diff",
    ):
        if _sweep_key in estimate:
            metrics_serialized[_sweep_key] = _maybe_finite(estimate[_sweep_key])

    exp_uniform = metrics_serialized.get("expected_uniform_points")
    meas = metrics_serialized.get("measurements")
    if exp_uniform is not None and meas is not None:
        metrics_serialized["sobol_difference"] = _maybe_finite(exp_uniform - meas)

    # Milestone metrics
    if run_result:
        milestone_data = calculate_zeeman_metrics(run_result)
        for k, v in milestone_data.items():
            metrics_serialized[k] = _maybe_finite(v)

    # Initialise Sobol convergence variables before they are populated from finalize_row
    sobol_freq_uncert_at_conv: float | None = None
    sobol_freq_err_at_conv: float | None = None

    sweep_steps: int | None = None
    locator_steps: int | None = None
    sobol_baseline_steps: int | None = None
    sobol_freq_steps: int | None = None
    sobol_conv_diff: int | None = None
    if not finalize_row.is_empty():
        if "sweep_steps" in finalize_row.columns:
            val = finalize_row.get_column("sweep_steps")[0]
            if val is not None:
                sweep_steps = int(val)
        if "locator_steps" in finalize_row.columns:
            val = finalize_row.get_column("locator_steps")[0]
            if val is not None:
                locator_steps = int(val)
        if "sobol_baseline_steps" in finalize_row.columns:
            val = finalize_row.get_column("sobol_baseline_steps")[0]
            if val is not None:
                sobol_baseline_steps = int(val)
        if "sobol_freq_steps" in finalize_row.columns:
            val = finalize_row.get_column("sobol_freq_steps")[0]
            if val is not None:
                sobol_freq_steps = int(val)
        if "sobol_conv_diff" in finalize_row.columns:
            val = finalize_row.get_column("sobol_conv_diff")[0]
            if val is not None:
                sobol_conv_diff = int(val)
        if "sobol_freq_uncert_at_conv" in finalize_row.columns:
            val = finalize_row.get_column("sobol_freq_uncert_at_conv")[0]
            if val is not None:
                sobol_freq_uncert_at_conv = float(val)
        if "sobol_freq_err_at_conv" in finalize_row.columns:
            val = finalize_row.get_column("sobol_freq_err_at_conv")[0]
            if val is not None:
                sobol_freq_err_at_conv = float(val)

    freq_converged_step: int | None = None
    all_converged_step: int | None = None
    theory_step_budget: int | None = None
    if not finalize_row.is_empty():
        if "freq_converged_step" in finalize_row.columns:
            val = finalize_row.get_column("freq_converged_step")[0]
            if val is not None:
                freq_converged_step = int(val)
        if "all_converged_step" in finalize_row.columns:
            val = finalize_row.get_column("all_converged_step")[0]
            if val is not None:
                all_converged_step = int(val)
        if "theory_step_budget" in finalize_row.columns:
            val = finalize_row.get_column("theory_step_budget")[0]
            if val is not None:
                theory_step_budget = int(val)

    is_converged = False
    if not finalize_row.is_empty() and "converged" in finalize_row.columns:
        val = finalize_row.get_column("converged")[0]
        if val is not None:
            is_converged = bool(val)

    _stop_reason = repeat_stop_reasons[attempt_idx_in_combo]
    if freq_converged_step is not None or is_converged:
        failure_reason: str | None = None
    elif strat_name in ("SimpleSweep", "SimpleSobol", "StagedSobolSweep", "StagedSobolSweepLocator", "GenericSweepLocator"):
        failure_reason = None
    elif _stop_reason == "infeasible_crlb":
        failure_reason = "infeasible_crlb"
    elif _stop_reason == "repeat_timeout":
        failure_reason = "timeout"
    elif theory_step_budget is not None and locator_steps is not None and locator_steps > theory_step_budget:
        failure_reason = "theory_budget"
    elif locator_steps is not None and max_steps is not None and locator_steps < max_steps:
        failure_reason = None
    else:
        failure_reason = "max_steps"

    if sobol_conv_diff is None and sobol_baseline_steps is not None and sobol_freq_steps is not None:
        sobol_conv_diff = sobol_baseline_steps - sobol_freq_steps

    # Forward Sobol freq uncertainty/error to metrics (for UI fallback via metrics.*)
    if sobol_freq_uncert_at_conv is not None:
        metrics_serialized["sobol_freq_uncert_at_conv"] = sobol_freq_uncert_at_conv
    if sobol_freq_err_at_conv is not None:
        metrics_serialized["sobol_freq_err_at_conv"] = sobol_freq_err_at_conv

    # Copy final estimates for parameters
    for param_name in (
        "frequency",
        "linewidth",
        "homogeneous_linewidth",
        "split",
        "dip_depth",
        "c_total",
        "k_np",
        "saturation",
        "sigma_inhom",
        "zeeman_split",
    ):
        if param_name in estimate:
            metrics_serialized[f"final_est_{param_name}"] = _maybe_finite(estimate[param_name])

    # Lineshape-agnostic derived estimates — the preferred axes for cross-lineshape
    # comparison (effective HWHM and realized contrast mean the same thing whether the
    # model parameterizes width as linewidth, homogeneous_linewidth, or saturation+sigma_inhom).
    if "saturation" in estimate and "sigma_inhom" in estimate:
        from nvision.spectra.nv_center import (
            NV_SATURATION_C_MAX,
            saturation_voigt_effective_hwhm_and_unc,
            saturation_voigt_realized_contrast_and_unc,
        )

        omega, _ = saturation_voigt_effective_hwhm_and_unc(estimate["saturation"], estimate["sigma_inhom"])
        metrics_serialized["final_est_effective_hwhm"] = _maybe_finite(omega)
        # c_max is a fixed constant (not inferred), so the realized contrast is
        # always computable here rather than gated on an "if c_max estimated" check.
        c_derived, _ = saturation_voigt_realized_contrast_and_unc(estimate["saturation"], NV_SATURATION_C_MAX)
        metrics_serialized["final_est_realized_contrast"] = _maybe_finite(c_derived)
    elif "linewidth" in estimate or "homogeneous_linewidth" in estimate:
        # Both are already HWHM-scale (unlike the old kernel-native fwhm_total, a
        # full width, which this branch used to also have to halve).
        hwhm = estimate.get("linewidth", estimate.get("homogeneous_linewidth"))
        metrics_serialized["final_est_effective_hwhm"] = _maybe_finite(hwhm)
        if "c_total" in estimate:
            metrics_serialized["final_est_realized_contrast"] = _maybe_finite(estimate["c_total"])

    # Grid-study coordinates: signal params held fixed by the generator (null = randomized
    # per repeat). Gives summary plots numeric axes instead of opaque generator names, and
    # lets the UI build its width/contrast/sigma_inhom/... facet selectors from these
    # structured values instead of parsing them back out of the generator name string.
    for grid_field in ("saturation", "sigma_inhom", "linewidth", "c_total"):
        val = getattr(generator_obj, grid_field, None)
        metrics_serialized[f"grid_{grid_field}"] = _maybe_finite(val) if val is not None else None
    # Lineshape variant ("lorentzian", "voigt", "saturation_voigt", ...) -- a string, so
    # kept separate from the numeric grid_* loop above.
    metrics_serialized["grid_variant"] = getattr(generator_obj, "variant", None)

    # Numeric noise level for trend-vs-noise analysis (None for non-Gauss noises).
    from nvision.sim.combinations import parse_gauss_sigma

    metrics_serialized["noise_sigma"] = parse_gauss_sigma(noise_name)

    # Ground-truth signal parameters for this repeat. frequency/zeeman_split vary per
    # repeat (fixed params equal their grid_ columns) — recording them lets analysis
    # de-confound repeat variance by the randomized parameters.
    try:
        for name, value in current_scan.true_signal.parameter_values().items():
            metrics_serialized[f"true_{name}"] = _maybe_finite(value)
    except Exception:  # e.g. MATLAB proxy signals without full parameter values
        pass

    if duration_ms_value is None:
        duration_ms_value = (time.perf_counter() - repeat_start_times[attempt_idx_in_combo]) * 1000
    metrics_serialized["duration_ms"] = _maybe_finite(duration_ms_value)

    # Convergence step milestones — pre-existing gap: these were computed above and
    # forwarded to entry_base (plot-manifest rows) but never to main_result_row, so
    # locator_results.csv never had them and every summary plot that checks for
    # "freq_converged_step"/"all_converged_step" (plot_experiment_summary,
    # plot_savings_vs_span_per_noise, plot_model_comparisons, plot_grid_study) was
    # silently skipping that metric.
    metrics_serialized["freq_converged_step"] = freq_converged_step
    metrics_serialized["all_converged_step"] = all_converged_step
    metrics_serialized["theory_step_budget"] = theory_step_budget

    main_result_row: dict[str, Any] = {
        "generator": gen_name,
        "noise": noise_name,
        "strategy": strat_name,
        "repeats": n_repeats,
        "max_steps": max_steps,
        "seed": seed,
        "attempt": attempt_idx_in_combo + 1,
        "stop_reason": repeat_stop_reasons[attempt_idx_in_combo],
        "failure_reason": failure_reason,
        **metrics_serialized,
    }

    entry_base: dict[str, Any] = {
        "generator": gen_name,
        "noise": noise_name,
        "strategy": strat_name,
        "repeat": attempt_idx_in_combo + 1,
        "repeat_total": n_repeats,
        "max_steps": max_steps,
        "seed": seed,
        "stop_reason": repeat_stop_reasons[attempt_idx_in_combo],
        "abs_err_x": metrics_serialized.get("abs_err_x"),
        "uncert": metrics_serialized.get("uncert"),
        "measurements": metrics_serialized.get("measurements"),
        "duration_ms": metrics_serialized.get("duration_ms"),
        "last_run": repeat_timestamps[attempt_idx_in_combo] if attempt_idx_in_combo < len(repeat_timestamps) else None,
        "sweep_steps": sweep_steps,
        "locator_steps": locator_steps,
        "sobol_baseline_steps": sobol_baseline_steps,
        "sobol_freq_steps": sobol_freq_steps,
        "sobol_conv_diff": sobol_conv_diff,
        "sobol_freq_uncert_at_conv": _maybe_finite(sobol_freq_uncert_at_conv),
        "sobol_freq_err_at_conv": _maybe_finite(sobol_freq_err_at_conv),
        "steps_to_fb": metrics_serialized.get("steps_to_fb"),
        "err_fb_at_milestone": metrics_serialized.get("err_fb_at_milestone"),
        "uncert_fb_at_milestone": metrics_serialized.get("uncert_fb_at_milestone"),
        "freq_converged_step": freq_converged_step,
        "all_converged_step": all_converged_step,
        "theory_step_budget": theory_step_budget,
        "failure_reason": failure_reason,
        "metrics": metrics_serialized,
    }

    return entry_base, main_result_row, current_history_df


def _scan_attempt_metrics(truth_positions: Sequence[float], estimate: dict[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    truth = [float(pos) for pos in truth_positions]

    if len(truth) == 1:
        x_hat = _first_finite(estimate, ("x1_hat", "x_hat", "peak_x", "frequency"))
        if x_hat is not None:
            metrics["abs_err_x"] = abs(x_hat - truth[0])
    elif len(truth) == 2:
        x1_hat = _maybe_finite(estimate.get("x1_hat"))
        x2_hat = _maybe_finite(estimate.get("x2_hat"))
        if x1_hat is not None and x2_hat is not None:
            xs = sorted([x1_hat, x2_hat])
            truth_sorted = sorted(truth)
            err1 = abs(xs[0] - truth_sorted[0])
            err2 = abs(xs[1] - truth_sorted[1])
            metrics["abs_err_x1"] = err1
            metrics["abs_err_x2"] = err2
            metrics["pair_rmse"] = math.sqrt(0.5 * (err1 * err1 + err2 * err2))

    for key in ("uncert", "uncert_pos", "uncert_sep", "final_entropy", "final_max_prob"):
        value = _maybe_finite(estimate.get(key))
        if value is not None:
            metrics[key] = value

    _promote_uncert(estimate, metrics)

    if "final_entropy" not in metrics:
        entropy_value = _maybe_finite(estimate.get("entropy"))
        if entropy_value is not None:
            metrics["final_entropy"] = entropy_value

    return metrics
