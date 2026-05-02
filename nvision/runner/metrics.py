"""Per-repeat metrics extraction."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Any

import polars as pl

from nvision.models.experiment import CoreExperiment
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
    current_scan: CoreExperiment,
    final_history_df: pl.DataFrame,
    finalize_results: pl.DataFrame,
    strat_obj: Any,
) -> tuple[dict[str, Any], dict[str, Any], pl.DataFrame]:
    """Calculate and format metrics for a single repeat.

    Returns
    -------
    tuple
        ``(entry_base, main_result_row, current_history_df)``
    """
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
    ):
        if _sweep_key in estimate:
            metrics_serialized[_sweep_key] = _maybe_finite(estimate[_sweep_key])

    if duration_ms_value is None:
        duration_ms_value = (time.perf_counter() - repeat_start_times[attempt_idx_in_combo]) * 1000
    metrics_serialized["duration_ms"] = _maybe_finite(duration_ms_value)

    main_result_row: dict[str, Any] = {
        "generator": gen_name,
        "noise": noise_name,
        "strategy": strat_name,
        "repeats": n_repeats,
        "attempt": attempt_idx_in_combo + 1,
        "stop_reason": repeat_stop_reasons[attempt_idx_in_combo],
        **metrics_serialized,
    }

    sweep_steps: int | None = None
    locator_steps: int | None = None
    if not finalize_row.is_empty():
        if "sweep_steps" in finalize_row.columns:
            val = finalize_row.get_column("sweep_steps")[0]
            if val is not None:
                sweep_steps = int(val)
        if "locator_steps" in finalize_row.columns:
            val = finalize_row.get_column("locator_steps")[0]
            if val is not None:
                locator_steps = int(val)

    entry_base: dict[str, Any] = {
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
        "sweep_steps": sweep_steps,
        "locator_steps": locator_steps,
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
