import math
from typing import List

import polars as pl


def find_two_peaks(
    history: pl.DataFrame,
    repeat_id: int,
    min_separation: float,
) -> dict:
    """
    Find up to two peaks in the given history for a repeat, separated by at least min_separation.
    Returns a dictionary with peak estimates and uncertainties.
    """
    repeat_history = history.filter(pl.col("repeat_id") == repeat_id)
    if repeat_history.is_empty():
        return {
            "repeat_id": repeat_id,
            "x1_hat": math.nan,
            "x2_hat": math.nan,
            "uncert": math.inf,
            "uncert_pos": math.inf,
            "uncert_sep": math.inf,
            "measurements": 0,
        }

    sorted_history = repeat_history.sort("signal_values", descending=True)
    picks: List[float] = []
    for x in sorted_history["x"]:
        if not picks or all(abs(x - p) > min_separation for p in picks):
            picks.append(x)
        if len(picks) == 2:
            break

    picks.sort()
    if len(picks) == 0:
        return {
            "repeat_id": repeat_id,
            "x1_hat": math.nan,
            "x2_hat": math.nan,
            "uncert": math.inf,
            "uncert_pos": math.inf,
            "uncert_sep": math.inf,
            "measurements": repeat_history.height,
        }
    if len(picks) == 1:
        return {
            "repeat_id": repeat_id,
            "x1_hat": float(picks[0]),
            "x2_hat": float(picks[0]),
            "uncert": 0.0,
            "uncert_pos": 0.0,
            "uncert_sep": 0.0,
            "measurements": repeat_history.height,
        }

    dist = abs(picks[1] - picks[0])
    return {
        "repeat_id": repeat_id,
        "x1_hat": float(picks[0]),
        "x2_hat": float(picks[1]),
        "uncert": float(0.5 * dist),
        "uncert_pos": float(0.5 * dist),
        "uncert_sep": float(0.5 * dist),
        "measurements": repeat_history.height,
    }
