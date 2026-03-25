from __future__ import annotations

import math
from datetime import timedelta

from rich.progress import ProgressColumn, Task


def _format_eta_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:0.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:0.1f}m"
    # `str(timedelta(...))` yields e.g. `1 day, 2:03:04` or `2:03:04` for < 1 day.
    return str(timedelta(seconds=int(seconds)))


class EstimatedTimeRemainingColumn(ProgressColumn):
    """ETA derived from cached duration estimates (task.total/task.completed are in ms)."""

    def render(self, task: Task) -> object:
        total_ms = task.total
        if total_ms is None or total_ms <= 0:
            return "ETA --"

        completed_ms = float(task.completed)
        remaining_expected_ms = max(float(total_ms) - completed_ms, 0.0)
        if remaining_expected_ms <= 0.0:
            return "ETA 0s"

        # Calibration: scale cached expectations by observed runtime-per-estimated-ms so far.
        elapsed = getattr(task, "elapsed", None)
        scale = 1.0
        if elapsed is not None and completed_ms > 0:
            elapsed_s: float | None = None
            if isinstance(elapsed, timedelta):
                elapsed_s = elapsed.total_seconds()
            else:
                try:
                    elapsed_s = float(elapsed)
                except (TypeError, ValueError):
                    elapsed_s = None

            if elapsed_s is not None and elapsed_s > 0:
                actual_elapsed_ms = elapsed_s * 1000.0
                scale = actual_elapsed_ms / completed_ms
                if not math.isfinite(scale) or scale <= 0:
                    scale = 1.0

        remaining_actual_ms = remaining_expected_ms * scale
        remaining_s = remaining_actual_ms / 1000.0
        return f"ETA {_format_eta_seconds(remaining_s)}"


class DotsColumn(ProgressColumn):
    def render(self, task: Task) -> object:
        return "." * int(task.completed)
