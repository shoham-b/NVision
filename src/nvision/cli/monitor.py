from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from nvision.cli.progress_columns import DotsColumn


class MonitorLogHandler(logging.Handler):
    """Queue formatted log lines for the Live log panel (no direct Console writes)."""

    def __init__(self, incoming: queue.Queue, *, formatter: logging.Formatter | None = None) -> None:
        super().__init__()
        self.incoming = incoming
        if formatter is not None:
            self.setFormatter(formatter)
        else:
            self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.incoming.put(escape(self.format(record)))
        except Exception:
            self.handleError(record)


class ProgressMonitor:
    """Rich Live UI: fixed log viewport, overall progress bar, locator rows — no scroll fight."""

    def __init__(
        self,
        console: Console,
        progress_queue: queue.Queue,
        *,
        log_incoming: queue.Queue | None,
        live_mode: bool = True,
        max_log_lines: int = 6,
        log_panel_height: int = 10,
        locator_panel_height: int = 8,
    ) -> None:
        self.console = console
        self.progress_queue = progress_queue
        self.log_incoming = log_incoming
        self.live_mode = live_mode
        self.max_log_lines = max_log_lines
        self._log_panel_height = log_panel_height
        self._locator_panel_height = locator_panel_height

        self._log_lines: deque[str] = deque(maxlen=max_log_lines)
        self._log_panel = _LogPanel(self)

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

        self._locator_panel = Panel(
            self.sub_progress,
            title="[bold]Locators[/]",
            border_style="dim",
            height=locator_panel_height,
        )

        self._live: Live | None = None
        self.main_task_id = self.main_progress.add_task("[cyan]Total progress", total=0)
        self._tid_to_weight: dict[Any, float] = {}
        self._completed_weighted = 0.0
        self._monitor_thread: threading.Thread | None = None

    def _layout(self) -> Group:
        if not self.live_mode:
            return Group(self.main_progress, self.sub_progress)
        return Group(self._log_panel, self.main_progress, self._locator_panel)

    def set_total_weight(self, total: float) -> None:
        self.main_progress.update(self.main_task_id, total=total)

    def register_task(self, description: str, total: int, weight: float) -> Any:
        tid = self.sub_progress.add_task(description, total=total)
        self._tid_to_weight[tid] = weight
        return tid

    def _sync_logs(self) -> None:
        if self.log_incoming is None:
            return
        while True:
            try:
                line = self.log_incoming.get_nowait()
            except queue.Empty:
                break
            self._log_lines.append(line)

    def _refresh_live(self) -> None:
        if self.live_mode and self._live is not None:
            self._live.update(self._layout())

    def _monitor_loop(self) -> None:
        while True:
            self._sync_logs()

            try:
                item = self.progress_queue.get(timeout=0.12)
            except queue.Empty:
                self._refresh_live()
                continue

            if item is None:
                self._sync_logs()
                self._refresh_live()
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

            self._sync_logs()
            self._refresh_live()

    def start(self) -> None:
        self._sync_logs()
        if self.live_mode:
            self._live = Live(
                self._layout(),
                console=self.console,
                refresh_per_second=10,
                vertical_overflow="visible",
                transient=False,
            )
            self._live.start()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self) -> None:
        self.progress_queue.put(None)
        if self._monitor_thread:
            self._monitor_thread.join()
            self._monitor_thread = None
        if self._live is not None:
            self._live.stop()
            self._live = None

    def __enter__(self) -> ProgressMonitor:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()


class _LogPanel:
    __slots__ = ("_monitor",)

    def __init__(self, monitor: ProgressMonitor) -> None:
        self._monitor = monitor

    def __rich__(self) -> RenderableType:
        lines = list(self._monitor._log_lines)
        if not lines:
            body: Text | str = Text("—", style="dim italic")
        else:
            body = Text("\n".join(lines))
        return Panel(
            body,
            title="[bold]Logs[/]",
            border_style="dim",
            height=self._monitor._log_panel_height,
        )
