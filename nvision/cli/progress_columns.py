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

    @staticmethod
    def _elapsed_seconds(task: Task) -> float | None:
        elapsed = getattr(task, "elapsed", None)
        if elapsed is None:
            return None
        if isinstance(elapsed, timedelta):
            return elapsed.total_seconds()
        try:
            return float(elapsed)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _updated_scale(cls, task: Task, *, total_ms: float, completed_ms: float, previous_scale: float) -> float:
        new_scale = 1.0
        # Require some completed progress before trusting calibration (very early estimates can be noisy).
        min_completed_fraction = 0.02
        if completed_ms > total_ms * min_completed_fraction:
            elapsed_s = cls._elapsed_seconds(task)
            if elapsed_s is not None and elapsed_s > 0 and completed_ms > 0:
                actual_elapsed_ms = elapsed_s * 1000.0
                new_scale = actual_elapsed_ms / completed_ms
                if not math.isfinite(new_scale) or new_scale <= 0:
                    new_scale = 1.0

        # Smooth calibration to reduce jumpiness between completion events.
        alpha = 0.25
        return (
            alpha * new_scale + (1.0 - alpha) * previous_scale
            if math.isfinite(previous_scale) and previous_scale > 0
            else new_scale
        )

    def render(self, task: Task) -> object:
        total_ms = task.total
        if total_ms is None or total_ms <= 0:
            return "ETA --"

        completed_ms = float(task.completed)
        remaining_expected_ms = max(float(total_ms) - completed_ms, 0.0)
        if remaining_expected_ms <= 0.0:
            return "ETA 0s"

        # Rich re-renders frequently; `task.elapsed` increases even when completed doesn't.
        # That caused the ETA to "drift" continuously between completion updates.
        #
        # We therefore only recalibrate the scale factor when `task.completed` changes.
        last_completed_ms = getattr(task, "_nvision_eta_last_completed_ms", None)
        scale: float = float(getattr(task, "_nvision_eta_scale", 1.0))

        if last_completed_ms is None or completed_ms != float(last_completed_ms):
            scale = self._updated_scale(task, total_ms=float(total_ms), completed_ms=completed_ms, previous_scale=scale)
            task._nvision_eta_last_completed_ms = completed_ms
            task._nvision_eta_scale = scale

        remaining_actual_ms = remaining_expected_ms * scale
        remaining_s = remaining_actual_ms / 1000.0
        return f"ETA {_format_eta_seconds(remaining_s)}"


class DotsColumn(ProgressColumn):
    def render(self, task: Task) -> object:
        return "." * int(task.completed)
