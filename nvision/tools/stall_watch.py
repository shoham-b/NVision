"""Stall watchdog: dump every thread's stack to a file when a process makes no progress for a while.

A run that "just sits there" leaves nothing in the normal log. With this armed, a process that
goes ``NVISION_STALL_DUMP_S`` seconds (default 600; 0 disables) without finishing its current
unit of work writes the stack of each of its threads to ``logs/stall-<role>-<pid>.txt``, repeating
every interval while it stays stuck. The stacks show what it is blocked on (a lock, a SQLite
write, a queue, or genuine computation), which is what diagnosing a stall needs.

Uses :mod:`faulthandler` (a C-level timer thread), so it works even while the Python interpreter
is stuck holding a lock or blocked in native code.
"""

from __future__ import annotations

import contextlib
import faulthandler
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import TextIO

from nvision.tools.paths import LOGS_ROOT

STALL_DUMP_S: float = float(os.getenv("NVISION_STALL_DUMP_S", "600"))

_file: TextIO | None = None
_file_role: str | None = None


def _dump_file(role: str) -> TextIO:
    # faulthandler keeps writing to the file object it was given, so it must stay open.
    global _file, _file_role
    if _file is None or _file_role != role:
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
        _file = (LOGS_ROOT / f"stall-{role}-{os.getpid()}.txt").open("a", buffering=1, encoding="utf-8")
        _file_role = role
    return _file


def arm(role: str, label: str, timeout_s: float | None = None) -> None:
    """(Re)start the countdown: dump all stacks if not disarmed/re-armed within ``timeout_s``."""
    timeout = STALL_DUMP_S if timeout_s is None else timeout_s
    if timeout <= 0:
        return
    out = _dump_file(role)
    out.write(f"\n=== {datetime.now(tz=UTC):%Y-%m-%d %H:%M:%S UTC} no progress on: {label} (dump in {timeout:.0f}s)\n")
    faulthandler.dump_traceback_later(timeout, repeat=True, file=out)


def disarm() -> None:
    faulthandler.cancel_dump_traceback_later()


@contextlib.contextmanager
def watch(role: str, label: str) -> Iterator[None]:
    """Arm for the duration of one unit of work (e.g. a task) and disarm afterwards."""
    arm(role, label)
    try:
        yield
    finally:
        disarm()
