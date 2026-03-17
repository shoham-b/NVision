from __future__ import annotations

import logging
import math
import time
from typing import Any

import polars as pl

from nvision.cli.utils import _maybe_finite, _scan_attempt_metrics
from nvision.core.experiment import CoreExperiment

log = logging.getLogger(__name__)


def _truth_positions(experiment: CoreExperiment) -> list[float]:
    """Extract ground truth peak positions from a CoreExperiment."""
    return [p.value for p in experiment.true_signal.parameters if "frequency" in p.name or "position" in p.name]


def generate_attempt_metrics(
    n_repeats: int,
    attempt_idx_in_combo: int,
    gen_name: str,
    noise_name: str,
    strat_name: str,
    repeat_stop_reasons: list[str],
    repeat_start_times: list[float],
    current_scan: CoreExperiment,
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
        estimate = {k: float(v) for k, v in estimate_dict.items() if isinstance(v, (int, float))}
        measurements = current_history_df.height

    truth_positions = _truth_positions(current_scan)
    attempt_metrics = _scan_attempt_metrics(truth_positions, estimate)
    metrics_serialized = {key: _maybe_finite(value) for key, value in attempt_metrics.items()}
    metrics_serialized["measurements"] = _maybe_finite(measurements)
    metrics_serialized["duration_ms"] = _maybe_finite(duration_ms)

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
