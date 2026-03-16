from __future__ import annotations

import logging
import math
import time
from typing import Any

import polars as pl

from nvision.cli.utils import _maybe_finite, _scan_attempt_metrics
from nvision.sim.locs.base import ScanBatch

log = logging.getLogger(__name__)


def generate_attempt_metrics(
    n_repeats: int,
    attempt_idx_in_combo: int,
    gen_name: str,
    noise_name: str,
    strat_name: str,
    repeat_stop_reasons: list[str],
    repeat_start_times: list[float],
    current_scan: ScanBatch,
    final_history_df: pl.DataFrame,
    finalize_results: pl.DataFrame,
    strat_obj: Any,
) -> tuple[dict[str, Any], dict[str, Any], pl.DataFrame]:
    """Calculate and format metrics for a single repeat."""
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

    duration_ms = (time.perf_counter() - repeat_start_times[attempt_idx_in_combo]) * 1000

    finalize_row = finalize_results.filter(pl.col("repeat_id") == attempt_idx_in_combo)
    if finalize_row.is_empty() or current_history_df.is_empty():
        estimate = {"x_hat": math.nan, "uncert": math.inf}
        measurements = 0
    else:
        estimate_dict = finalize_row.drop("repeat_id").to_dicts()[0]
        estimate = {k: float(v) for k, v in estimate_dict.items()}
        measurements = current_history_df.height

    attempt_metrics = _scan_attempt_metrics(current_scan.truth_positions, estimate)
    metrics_serialized = {key: _maybe_finite(value) for key, value in attempt_metrics.items()}
    metrics_serialized["measurements"] = _maybe_finite(measurements)
    metrics_serialized["duration_ms"] = _maybe_finite(duration_ms)

    _try_append_bayesian_metrics(
        metrics_serialized, strat_obj, attempt_idx_in_combo, current_scan, current_history_df, noise_name
    )

    main_result_row = {
        "generator": gen_name,
        "noise": noise_name,
        "strategy": strat_name,
        "repeats": n_repeats,
        "attempt": attempt_idx_in_combo + 1,
        "stop_reason": repeat_stop_reasons[attempt_idx_in_combo],
        **metrics_serialized,
    }

    entry_base = {
        "generator": gen_name,
        "noise": noise_name,
        "strategy": strat_name,
        "repeat": attempt_idx_in_combo + 1,
        "repeat_total": n_repeats,
        "stop_reason": repeat_stop_reasons[attempt_idx_in_combo],
        "abs_err_x": metrics_serialized.get("abs_err_x"),
        "uncert": metrics_serialized.get("uncert"),
        "measurements": metrics_serialized.get("measurements"),
        "duration_ms": metrics_serialized.get("duration_ms"),
        "metrics": metrics_serialized,
    }

    return entry_base, main_result_row, current_history_df


def _try_append_bayesian_metrics(
    metrics_serialized: dict,
    strat_obj: Any,
    attempt_idx_in_combo: int,
    current_scan: ScanBatch,
    current_history_df: pl.DataFrame,
    noise_name: str,
):
    try:
        from nvision.sim.locs.nv_center.evaluation import BayesianMetrics
        from nvision.sim.locs.nv_center.sequential_bayesian_locator import NVCenterSequentialBayesianLocator

        locator_for_metrics = None
        if hasattr(strat_obj, "_get_locator"):
            locator_for_metrics = strat_obj._get_locator(attempt_idx_in_combo)
        elif isinstance(strat_obj, NVCenterSequentialBayesianLocator):
            locator_for_metrics = strat_obj

        if locator_for_metrics and hasattr(locator_for_metrics, "parameter_history"):
            meta = getattr(current_scan, "meta", {}) or {}

            linewidth = getattr(current_scan, "linewidth", None)
            if linewidth is None and "omega" in meta:
                val = meta["omega"]
                if val is not None:
                    linewidth = val * 2.0

            amplitude = getattr(current_scan, "amplitude", None)
            if amplitude is None:
                amplitude = meta.get("a")

            background = getattr(current_scan, "background", None)
            if background is None:
                background = meta.get("base")

            ground_truth = {
                "frequency": current_scan.truth_positions[0]
                if current_scan.truth_positions
                else meta.get("f0", meta.get("f_B")),
                "linewidth": linewidth,
                "amplitude": amplitude,
                "background": background,
            }
            ground_truth = {k: v for k, v in ground_truth.items() if v is not None}

            metrics_obj = BayesianMetrics.from_history(
                parameter_history=locator_for_metrics.parameter_history,
                ground_truth=ground_truth,
                measurement_history=current_history_df,
                noise_model=noise_name if "poisson" in noise_name.lower() else "gaussian",
                noise_params={"sigma": 0.05} if "gaussian" in noise_name.lower() else None,
            )

            bayes_summary = metrics_obj.summary()
            for k, v in bayes_summary.items():
                metrics_serialized[k] = _maybe_finite(v)
    except Exception as e:
        log.warning(f"Failed to calculate Bayesian metrics for repeat {attempt_idx_in_combo + 1}: {e}")
