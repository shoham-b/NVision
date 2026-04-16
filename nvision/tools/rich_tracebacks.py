"""Rich traceback configuration utilities."""

from __future__ import annotations

import concurrent.futures
import multiprocessing

import typer
from rich.traceback import install


def install_rich_tracebacks() -> None:
    """Configure rich traceback to show relevant stack frames only."""
    suppress = [typer, multiprocessing, concurrent.futures]
    try:
        import numba

        suppress.append(numba)
    except ImportError:
        pass

    install(show_locals=False, suppress=suppress, width=100, word_wrap=True)
