"""Per-task log context for combination-scoped CLI file logs."""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token

_combo_initials: ContextVar[str] = ContextVar("nvision_combo_initials", default="")


def format_combination_log_initials(generator_name: str, noise_name: str, strategy_name: str) -> str:
    """Short gen/noise/strategy initials for log lines (hyphen segment initials, up to two per field)."""

    def abbrev(label: str) -> str:
        parts = [p for p in label.replace("_", "-").split("-") if p]
        if not parts:
            return "?"
        return "".join(p[0].upper() for p in parts[:2])

    return f"{abbrev(generator_name)}.{abbrev(noise_name)}.{abbrev(strategy_name)}"


def set_combination_log_initials(generator_name: str, noise_name: str, strategy_name: str) -> Token[str]:
    """Return a token for :func:`reset_combination_log_initials`."""
    tag = format_combination_log_initials(generator_name, noise_name, strategy_name)
    return _combo_initials.set(tag)


def reset_combination_log_initials(token: Token[str]) -> None:
    _combo_initials.reset(token)


class CombinationLogFilter(logging.Filter):
    """Stamp ``combo_prefix`` on records in the emitting thread (before QueueHandler pickling)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if "combo_prefix" not in record.__dict__:
            tag = _combo_initials.get()
            record.combo_prefix = f"[{tag}] " if tag else ""
        return True
