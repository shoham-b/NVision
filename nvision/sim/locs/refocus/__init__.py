"""Shape-aware refocus strategies for inferring signal windows from dip observations.

Public API
----------
- :func:`detect_dips`: Find double-monotonic dips in (x, y) observations.
- :func:`infer_dip_widths`: Per-dip width inference using Strategies A/B.
- :func:`infer_focus_window`: Aggregate window from per-dip widths (Strategy C).
"""

from __future__ import annotations

from nvision.sim.locs.refocus.strategies import detect_dips, infer_dip_widths
from nvision.sim.locs.refocus.window import infer_focus_window

__all__ = [
    "detect_dips",
    "infer_dip_widths",
    "infer_focus_window",
]
