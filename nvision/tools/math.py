from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def _to_native(obj: Any) -> Any:
    """Recursively convert NumPy scalars to native Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_native(v) for v in obj]
    return obj


def _maybe_finite(value: object) -> float | None:
    if isinstance(value, int | float):
        value_float = float(value)
        if math.isfinite(value_float):
            return value_float
    return None


def _first_finite(estimate: dict[str, object], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _maybe_finite(estimate.get(key))
        if value is not None:
            return value
    return None


def _promote_uncert(estimate: dict[str, object], metrics: dict[str, float]) -> None:
    if "uncert" in metrics:
        return

    preferred_uncert = _first_finite(
        estimate,
        ("uncert_frequency", "uncert_position", "uncert_x1", "uncert_peak_x"),
    )
    if preferred_uncert is not None:
        metrics["uncert"] = preferred_uncert
        return

    for key, raw in estimate.items():
        if key.startswith("uncert_"):
            value = _maybe_finite(raw)
            if value is not None:
                metrics["uncert"] = value
                return
