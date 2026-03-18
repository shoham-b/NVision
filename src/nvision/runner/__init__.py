"""NVision runner package.

Owns all logic for executing locator experiments end-to-end:

- ``loop``        — single-repeat measurement loop (``run_loop``)
- ``batch``       — N-repeat batch orchestrator (``run_simulation_batch``)
- ``convert``     — RunResult → DataFrame conversion utilities
- ``combination`` — (generator × noise × strategy) orchestrator with caching
- ``metrics``     — per-repeat metrics extraction
- ``plots``       — per-repeat plot generation
- ``tasks``       — LocatorTask list builder (``TaskBuildConfig``, ``build_tasks``)
- ``cache``       — graph content embed/restore helpers
"""

from nvision.runner.batch import run_simulation_batch
from nvision.runner.combination import run_combination
from nvision.runner.convert import (
    denormalize_x,
    extract_peak_estimates,
    run_result_to_finalize_record,
    run_result_to_history_df,
)
from nvision.runner.loop import run_loop
from nvision.runner.metrics import generate_attempt_metrics
from nvision.runner.plots import generate_attempt_plots
from nvision.runner.tasks import TaskBuildConfig, build_tasks

__all__ = [
    "TaskBuildConfig",
    "build_tasks",
    "denormalize_x",
    "extract_peak_estimates",
    "generate_attempt_metrics",
    "generate_attempt_plots",
    "run_combination",
    "run_loop",
    "run_result_to_finalize_record",
    "run_result_to_history_df",
    "run_simulation_batch",
]
