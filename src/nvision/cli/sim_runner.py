from __future__ import annotations

from typing import Any

import polars as pl

from nvision.core.structures import LocatorTask


def run_simulation_batch(
    task: LocatorTask,
) -> tuple[pl.DataFrame, pl.DataFrame, list[Any], list[float], list[str]]:
    """Run a simulation batch using the core architecture."""
    from nvision.cli.native_runner import run_native_simulation_batch

    return run_native_simulation_batch(task)
