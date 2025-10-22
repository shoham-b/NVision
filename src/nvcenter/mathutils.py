"""Unified math utilities for NvCenter.

This module centralizes small math helpers used across the project to avoid
duplicating functionality in subpackages. Prefer using library-provided
operations (e.g., Polars, Python math) where available, and keep only
project-specific helpers here.
"""

from __future__ import annotations

import math
from typing import Iterable


def normalized_sum(values: Iterable[float]) -> float:
    """Return the sum of values after normalizing them into [0, 1] range.

    If the iterable is empty, returns 0.0.
    If all values are equal, returns 0.0 (since normalization would divide by zero otherwise).
    """
    vals = list(values)
    if not vals:
        return 0.0
    vmin = min(vals)
    vmax = max(vals)
    span = vmax - vmin
    if span == 0:
        return 0.0
    total = 0.0
    for v in vals:
        total += (v - vmin) / span
    return total


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp a value into the [lo, hi] range."""
    return max(lo, min(hi, v))


def safe_log(x: float, eps: float = 1e-12) -> float:
    """Numerically safe natural log guarding against non-positive inputs."""
    return math.log(max(x, eps))
