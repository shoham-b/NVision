"""NVision runner package.

Owns all logic for executing locator experiments end-to-end:

- ``executor``  — runs a LocatorTask (cache, repeats, metrics, plots)
                 and exposes ``run_loop`` for one-repeat iteration
- ``convert``   — RunResult → DataFrame conversion utilities
- ``metrics``   — per-repeat metrics extraction
- ``plots``     — per-repeat plot generation
- ``task_builder`` — LocatorTask list builder (``TaskListBuildConfig``, ``build_task_list``)
- ``cache``     — graph embed/restore; locator key helpers re-export :mod:`nvision.cache.locator_keys`
"""

from nvision.runner.convert import (
    denormalize_x,
    extract_peak_estimates,
    run_result_to_finalize_record,
    run_result_to_history_df,
)
from nvision.runner.executor import run_loop, run_task
from nvision.runner.metrics import generate_attempt_metrics
from nvision.runner.plots import generate_attempt_plots
from nvision.runner.task_builder import TaskListBuildConfig, build_task_list

__all__ = [
    "TaskListBuildConfig",
    "build_task_list",
    "denormalize_x",
    "extract_peak_estimates",
    "generate_attempt_metrics",
    "generate_attempt_plots",
    "run_loop",
    "run_result_to_finalize_record",
    "run_result_to_history_df",
    "run_task",
]
