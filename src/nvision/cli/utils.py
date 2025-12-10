from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

log = logging.getLogger("nvision")


def _to_native(obj: Any) -> Any:
    """Recursively convert NumPy scalars to native Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_native(v) for v in obj]
    return obj


def _get_generator_category(generator_name: str) -> str:
    """Determine the category of a generator from its name."""
    if generator_name.startswith("OnePeak-"):
        return "OnePeak"
    elif generator_name.startswith("TwoPeak-"):
        return "TwoPeak"
    elif generator_name.startswith("NVCenter-"):
        return "NVCenter"
    return "Unknown"


def _maybe_finite(value: object) -> float | None:
    if isinstance(value, int | float):
        value_float = float(value)
        if math.isfinite(value_float):
            return value_float
    return None


def _scan_attempt_metrics(
    truth_positions: Sequence[float], estimate: dict[str, object]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    truth = [float(pos) for pos in truth_positions]

    if len(truth) == 1:
        x_hat = estimate.get("x1_hat", estimate.get("x_hat"))
        if isinstance(x_hat, int | float) and math.isfinite(float(x_hat)):
            metrics["abs_err_x"] = abs(float(x_hat) - truth[0])
    else:
        x1_hat = estimate.get("x1_hat")
        x2_hat = estimate.get("x2_hat")
        if (
            isinstance(x1_hat, int | float)
            and math.isfinite(float(x1_hat))
            and isinstance(x2_hat, int | float)
            and math.isfinite(float(x2_hat))
        ):
            xs = sorted([float(x1_hat), float(x2_hat)])
            truth_sorted = sorted(truth)
            err1 = abs(xs[0] - truth_sorted[0])
            err2 = abs(xs[1] - truth_sorted[1])
            metrics["abs_err_x1"] = err1
            metrics["abs_err_x2"] = err2
            metrics["pair_rmse"] = math.sqrt(0.5 * (err1 * err1 + err2 * err2))

    for key in ("uncert", "uncert_pos", "uncert_sep", "final_entropy", "final_max_prob"):
        value = estimate.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            metrics[key] = float(value)

    return metrics


def _load_duration_estimates(out_dir: Path) -> dict[tuple[str, str, str], float]:
    """Load duration estimates from previous run metadata if available."""
    csv_path = out_dir / "locator_results.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pl.read_csv(csv_path)
        if "duration_ms" not in df.columns:
            return {}

        # Group by config and take mean duration
        # We assume columns generator, noise, strategy exist
        if not all(c in df.columns for c in ["generator", "noise", "strategy"]):
            return {}

        stats = df.group_by(["generator", "noise", "strategy"]).agg(pl.col("duration_ms").mean())

        estimates = {}
        for row in stats.to_dicts():
            key = (row["generator"], row["noise"], row["strategy"])
            estimates[key] = row["duration_ms"]
        return estimates
    except Exception:
        return {}
