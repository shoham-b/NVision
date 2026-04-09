from __future__ import annotations

import contextlib
import logging
import queue
import sys
import threading
from collections import deque
from typing import Any, Literal, TextIO

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.text import Text

from nvision.cli.progress_columns import DotsColumn, EstimatedTimeRemainingColumn

ScrollFocus = Literal["logs", "errors", "locators"]

_LOG_HISTORY_MAX = 8000
_ERROR_HISTORY_MAX = 2000

_FOCUS_ORDER: tuple[ScrollFocus, ...] = ("logs", "errors", "locators")

# Restored in `_keyboard_tty_exit` after Live monitor stops (POSIX arrow keys).
_tty_attrs_saved: Any = None


def _keyboard_tty_enter() -> None:
    """cbreak stdin for single-key reads while the Live UI runs (POSIX only)."""
    global _tty_attrs_saved
    if sys.platform == "win32" or not sys.stdin.isatty():
        return
    try:
        import termios
        import tty

        _tty_attrs_saved = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
    except (OSError, ImportError, AttributeError):
        _tty_attrs_saved = None


def _keyboard_tty_exit() -> None:
    global _tty_attrs_saved
    if _tty_attrs_saved is None:
        return
    try:
        import termios

        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _tty_attrs_saved)
    except (OSError, ImportError, AttributeError):
        pass
    finally:
        _tty_attrs_saved = None


def _poll_scroll_key_win32() -> str | None:
    try:
        import msvcrt
    except ImportError:
        return None
    try:
        if not msvcrt.kbhit():
            return None
        c = msvcrt.getch()
        if c == b"\t":
            return "tab"
        if c in (b"\xe0", b"\x00"):
            c2 = msvcrt.getch()
            mapping = {
                b"H": "up",
                b"P": "down",
                b"I": "pgup",
                b"Q": "pgdn",
                b"G": "home",
                b"O": "end",
            }
            return mapping.get(c2)
    except OSError:
        return None
    return None


def _posix_read_csi_tail(third: str, stdin: TextIO) -> str | None:
    """Parse CSI sequence starting after ESC [ (third is first char after '[')."""
    import select

    if third in "AB":
        return "up" if third == "A" else "down"
    if third in "HF":
        return "home" if third == "H" else "end"
    buf = third
    while buf and buf[-1] not in "~":
        if not select.select([stdin], [], [], 0.05)[0]:
            break
        buf += stdin.read(1)
    if "5~" in buf:
        return "pgup"
    if "6~" in buf:
        return "pgdn"
    if "1~" in buf:
        return "home"
    if "4~" in buf:
        return "end"
    return None


def _posix_key_after_esc(stdin: TextIO) -> str | None:
    """Byte(s) after ESC; caller already consumed ESC."""
    import select

    if not select.select([stdin], [], [], 0.05)[0]:
        return None
    second = stdin.read(1)
    if second == "H":
        return "home"
    if second == "F":
        return "end"
    if second == "O":
        if select.select([stdin], [], [], 0.05)[0]:
            third_o = stdin.read(1)
            if third_o == "H":
                return "home"
            if third_o == "F":
                return "end"
        return None
    if second != "[":
        return None
    if not select.select([stdin], [], [], 0.05)[0]:
        return None
    third = stdin.read(1)
    return _posix_read_csi_tail(third, stdin)


def _poll_scroll_key_posix() -> str | None:
    """Requires `_keyboard_tty_enter`; reads escape sequences from stdin."""
    try:
        import select
    except ImportError:
        return None
    if not sys.stdin.isatty():
        return None
    stdin = sys.stdin
    try:
        if not select.select([stdin], [], [], 0)[0]:
            return None
        first = stdin.read(1)
        if first == "\t":
            return "tab"
        if first != "\x1b":
            return None
        return _posix_key_after_esc(stdin)
    except (OSError, AttributeError, ValueError, EOFError):
        return None


def _poll_scroll_key() -> str | None:
    """Non-blocking read: arrow keys, PgUp/Dn, Home/End, Tab. None if nothing."""
    if sys.platform == "win32":
        return _poll_scroll_key_win32()
    return _poll_scroll_key_posix()


def _scroll_after_key(scroll: int, key: str, inner_h: int, n_items: int) -> int:
    """Update scroll-from-bottom offset for one key (0 = newest)."""
    max_s = max(0, n_items - inner_h)
    half = max(1, inner_h // 2)
    if key == "up":
        return min(scroll + 1, max_s)
    if key == "down":
        return max(scroll - 1, 0)
    if key == "pgup":
        return min(scroll + half, max_s)
    if key == "pgdn":
        return max(scroll - half, 0)
    if key == "home":
        return max_s
    if key == "end":
        return 0
    return scroll


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


class MonitorErrorHandler(logging.Handler):
    """Queue ERROR/CRITICAL lines for the Live errors panel (no direct Console writes)."""

    def __init__(self, incoming: queue.Queue, *, formatter: logging.Formatter | None = None) -> None:
        super().__init__(level=logging.ERROR)
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
    """Rich Live UI: scrollable log + errors + locator viewports, overall progress bar."""

    def __init__(
        self,
        console: Console,
        progress_queue: queue.Queue,
        *,
        log_incoming: queue.Queue | None,
        error_incoming: queue.Queue | None = None,
        live_mode: bool = True,
        max_log_lines: int = _LOG_HISTORY_MAX,
        max_error_lines: int = _ERROR_HISTORY_MAX,
        log_panel_height: int = 10,
        error_panel_height: int = 6,
        locator_panel_height: int = 8,
    ) -> None:
        self.console = console
        self.progress_queue = progress_queue
        self.log_incoming = log_incoming
        self.error_incoming = error_incoming
        self.live_mode = live_mode
        self.max_log_lines = max_log_lines
        self.max_error_lines = max_error_lines
        self._log_panel_height = log_panel_height
        self._error_panel_height = error_panel_height
        self._locator_panel_height = locator_panel_height

        self._log_lines: deque[str] = deque(maxlen=max_log_lines)
        self._error_lines: deque[str] = deque(maxlen=max_error_lines)
        self._log_scroll: int = 0
        self._err_scroll: int = 0
        self._loc_scroll: int = 0
        self._scroll_focus: ScrollFocus = "logs"
        self._log_panel = _LogPanel(self)
        self._error_panel = _ErrorPanel(self)

        self.main_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            EstimatedTimeRemainingColumn(),
        )

        self.sub_progress = Progress(
            TextColumn("{task.description}"),
            DotsColumn(),
        )

        self._locator_panel = _LocatorPanel(self)

        self._live: Live | None = None
        self.main_task_id = self.main_progress.add_task("[cyan]Total progress", total=0)
        self._tid_to_weight: dict[Any, float] = {}
        self._completed_weighted = 0.0
        self._monitor_thread: threading.Thread | None = None

    def _inner_height(self, panel_height: int) -> int:
        return max(1, panel_height - 2)

    def _layout(self) -> Group:
        if not self.live_mode:
            return Group(self.main_progress, self.sub_progress)
        if self.error_incoming is not None:
            return Group(
                self._log_panel,
                self._error_panel,
                self.main_progress,
                self._locator_panel,
            )
        return Group(self._log_panel, self.main_progress, self._locator_panel)

    def _focus_cycle(self) -> tuple[ScrollFocus, ...]:
        if self.error_incoming is not None:
            return _FOCUS_ORDER
        return ("logs", "locators")

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
            if self._log_scroll != 0:
                # Keep the same visible lines when new output arrives while reading history.
                log_h = self._inner_height(self._log_panel_height)
                self._log_scroll = min(
                    self._log_scroll + 1,
                    max(0, len(self._log_lines) - log_h),
                )

    def _sync_errors(self) -> None:
        if self.error_incoming is None:
            return
        while True:
            try:
                line = self.error_incoming.get_nowait()
            except queue.Empty:
                break
            self._error_lines.append(line)
            if self._err_scroll != 0:
                err_h = self._inner_height(self._error_panel_height)
                self._err_scroll = min(
                    self._err_scroll + 1,
                    max(0, len(self._error_lines) - err_h),
                )

    def _sync_incoming(self) -> None:
        self._sync_logs()
        self._sync_errors()

    def _apply_scroll_key(self, key: str) -> None:
        if key == "tab":
            order = self._focus_cycle()
            i = order.index(self._scroll_focus)
            self._scroll_focus = order[(i + 1) % len(order)]
            return
        log_h = self._inner_height(self._log_panel_height)
        err_h = self._inner_height(self._error_panel_height)
        loc_h = self._inner_height(self._locator_panel_height)
        if self._scroll_focus == "logs":
            self._log_scroll = _scroll_after_key(self._log_scroll, key, log_h, len(self._log_lines))
        elif self._scroll_focus == "errors":
            self._err_scroll = _scroll_after_key(self._err_scroll, key, err_h, len(self._error_lines))
        else:
            self._loc_scroll = _scroll_after_key(self._loc_scroll, key, loc_h, len(self.sub_progress.tasks))

    def _poll_keys(self) -> None:
        key = _poll_scroll_key()
        if key is not None:
            self._apply_scroll_key(key)

    def _refresh_live(self) -> None:
        if self.live_mode and self._live is not None:
            self._live.update(self._layout())

    def _monitor_loop(self) -> None:
        while True:
            self._sync_incoming()
            self._poll_keys()

            try:
                item = self.progress_queue.get(timeout=0.12)
            except queue.Empty:
                self._poll_keys()
                self._refresh_live()
                continue

            if item is None:
                self._sync_incoming()
                self._poll_keys()
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

            self._sync_incoming()
            self._poll_keys()
            self._refresh_live()

    def start(self) -> None:
        self._sync_incoming()
        if self.live_mode:
            self._live = Live(
                self._layout(),
                console=self.console,
                refresh_per_second=10,
                vertical_overflow="visible",
                transient=False,
            )
            self._live.start()
            # Ensure `task.elapsed` for the ETA column starts immediately.
            # This matters more with multiprocessing, where the first completed update
            # can happen significantly after the run begins.
            with contextlib.suppress(Exception):
                self.main_progress.start_task(self.main_task_id)
            _keyboard_tty_enter()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self) -> None:
        self.progress_queue.put(None)
        if self._monitor_thread:
            self._monitor_thread.join()
            self._monitor_thread = None
        _keyboard_tty_exit()
        if self._live is not None:
            self._live.stop()
            self._live = None

    def __enter__(self) -> ProgressMonitor:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()


def _panel_title(base: str, *, focused: bool) -> str:
    hint = "[dim]Tab | arrows | PgUp/Dn | Home/End[/]"
    mark = "[bold cyan]*[/] " if focused else ""
    return f"{mark}[bold]{base}[/] {hint}"


def _error_panel_title(*, focused: bool) -> str:
    hint = "[dim]Tab | arrows | PgUp/Dn | Home/End[/]"
    mark = "[bold red]*[/] " if focused else ""
    return f"{mark}[bold red]Errors[/] {hint}"


class _LogPanel:
    __slots__ = ("_monitor",)

    def __init__(self, monitor: ProgressMonitor) -> None:
        self._monitor = monitor

    def __rich__(self) -> RenderableType:
        m = self._monitor
        lines = list(m._log_lines)
        inner_h = m._inner_height(m._log_panel_height)
        focused = m._scroll_focus == "logs"
        start = 0
        window: list[str] = []

        if not lines:
            body: Text | str = Text("—", style="dim italic")
        else:
            total = len(lines)
            max_scroll = max(0, total - inner_h)
            m._log_scroll = min(m._log_scroll, max_scroll)
            start = max(0, total - inner_h - m._log_scroll)
            window = lines[start : start + inner_h]
            body = Text("\n".join(window))

        title = _panel_title("Logs", focused=focused)
        subtitle = None
        if lines and inner_h < len(lines) and window:
            subtitle = f"[dim]{start + 1}-{start + len(window)} of {len(lines)}[/]"

        return Panel(
            body,
            title=title,
            subtitle=subtitle,
            border_style="cyan" if focused else "dim",
            height=m._log_panel_height,
        )


class _ErrorPanel:
    __slots__ = ("_monitor",)

    def __init__(self, monitor: ProgressMonitor) -> None:
        self._monitor = monitor

    def __rich__(self) -> RenderableType:
        m = self._monitor
        lines = list(m._error_lines)
        inner_h = m._inner_height(m._error_panel_height)
        focused = m._scroll_focus == "errors"
        start = 0
        window: list[str] = []

        if not lines:
            body: Text | str = Text("— no errors —", style="dim italic")
        else:
            total = len(lines)
            max_scroll = max(0, total - inner_h)
            m._err_scroll = min(m._err_scroll, max_scroll)
            start = max(0, total - inner_h - m._err_scroll)
            window = lines[start : start + inner_h]
            body = Text("\n".join(window), style="red")

        title = _error_panel_title(focused=focused)
        subtitle = None
        if lines and inner_h < len(lines) and window:
            subtitle = f"[dim]{start + 1}-{start + len(window)} of {len(lines)}[/]"

        return Panel(
            body,
            title=title,
            subtitle=subtitle,
            border_style="red" if focused else "dim red",
            height=m._error_panel_height,
        )


def _locator_row_markup(task: object) -> str:
    """Combo path + dots; color by state (matches task_builder `[cyan]…[/cyan]` descriptions)."""
    desc = getattr(task, "description", None) or ""
    dots = "." * int(getattr(task, "completed", 0))
    inner = desc
    if desc.startswith("[cyan]") and desc.endswith("[/cyan]"):
        inner = desc[6:-8]
    total = getattr(task, "total", None)
    completed = float(getattr(task, "completed", 0))
    if total is None or total <= 0 or completed <= 0:
        style = "cyan"
    elif completed >= total:
        style = "green"
    else:
        style = "yellow"
    return f"[{style}]{inner}[/{style}]  {dots}"


class _LocatorPanel:
    __slots__ = ("_monitor",)

    def __init__(self, monitor: ProgressMonitor) -> None:
        self._monitor = monitor

    def __rich__(self) -> RenderableType:
        m = self._monitor
        inner_h = m._inner_height(m._locator_panel_height)
        focused = m._scroll_focus == "locators"
        start = 0
        window: list[str] = []

        rows: list[str] = []
        for task in m.sub_progress.tasks:
            rows.append(_locator_row_markup(task))

        if not rows:
            body: Text | str = Text("—", style="dim italic")
        else:
            total = len(rows)
            max_scroll = max(0, total - inner_h)
            m._loc_scroll = min(m._loc_scroll, max_scroll)
            start = max(0, total - inner_h - m._loc_scroll)
            window = rows[start : start + inner_h]
            # Task descriptions use Rich markup (see task_builder); parse it like Progress columns do.
            body = Text.from_markup("\n".join(window))

        title = _panel_title("Locators", focused=focused)
        subtitle = None
        if rows and inner_h < len(rows) and window:
            subtitle = f"[dim]{start + 1}-{start + len(window)} of {len(rows)}[/]"

        return Panel(
            body,
            title=title,
            subtitle=subtitle,
            border_style="cyan" if focused else "dim",
            height=m._locator_panel_height,
        )
