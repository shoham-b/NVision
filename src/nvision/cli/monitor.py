from __future__ import annotations

import queue
import threading
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from nvision.cli.progress_columns import DotsColumn


class ProgressMonitor:
    """Encapsulates the rich.live UI and threaded reading of the progress queue."""

    def __init__(self, console: Console, progress_queue: queue.Queue):
        self.console = console
        self.progress_queue = progress_queue

        self.main_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )

        self.sub_progress = Progress(
            TextColumn("{task.description}"),
            DotsColumn(),
        )

        self.progress_group = Group(self.main_progress, self.sub_progress)
        self.live = Live(self.progress_group, console=self.console, refresh_per_second=10)

        self.main_task_id = self.main_progress.add_task("[cyan]Total Progress", total=0)
        self._tid_to_weight: dict[Any, float] = {}
        self._completed_weighted = 0.0
        self._monitor_thread: threading.Thread | None = None

    def set_total_weight(self, total: float):
        self.main_progress.update(self.main_task_id, total=total)

    def register_task(self, description: str, total: int, weight: float) -> Any:
        tid = self.sub_progress.add_task(description, total=total)
        self._tid_to_weight[tid] = weight
        return tid

    def _monitor_loop(self):
        while True:
            item = self.progress_queue.get()
            if item is None:
                break
            tid, advance = item
            self.sub_progress.update(tid, advance=advance)

            weight = self._tid_to_weight.get(tid, 1000.0)
            self._completed_weighted += advance * weight
            self.main_progress.update(self.main_task_id, completed=self._completed_weighted)

            for task in self.sub_progress.tasks:
                if task.id == tid and task.completed >= task.total:
                    self.sub_progress.remove_task(tid)
                    break

    def start(self):
        self.live.start()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        self.progress_queue.put(None)
        if self._monitor_thread:
            self._monitor_thread.join()
        self.live.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
