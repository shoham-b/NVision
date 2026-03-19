"""Task-list builder — creates runnable LocatorTask objects."""

from __future__ import annotations

import queue
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from nvision.models.task import LocatorTask
from nvision.sim.combinations import CombinationGrid
from nvision.tools.paths import slugify

if TYPE_CHECKING:
    from nvision.cli.monitor import ProgressMonitor


@dataclass
class TaskListBuildConfig:
    """Configuration for building the runnable task list."""

    repeats: int
    seed: int
    out_dir: Path
    scans_dir: Path
    bayes_dir: Path
    cache_dir: Path
    log_queue: queue.Queue
    progress_queue: queue.Queue
    log_level_value: int
    loc_max_steps: int
    loc_timeout_s: int
    no_cache: bool
    ignore_cache_strategy: str | None
    require_cache: bool
    filter_category: str | None
    filter_strategy: str | None
    filter_generator: str | None


def build_task_list(
    config: TaskListBuildConfig,
    monitor: ProgressMonitor,
) -> tuple[list[LocatorTask], float]:
    """Build all matching tasks and register them with the progress monitor."""
    grid = CombinationGrid()
    duration_estimates = _load_duration_estimates(config.out_dir)

    task_list: list[LocatorTask] = []
    used_slugs: set[str] = set()
    total_weighted_repeats = 0.0

    for combo in grid.iter(config.filter_category, config.filter_strategy, config.filter_generator):
        slug_base = "_".join(slugify(p) for p in (combo.generator_name, combo.noise_name, combo.strategy_name))
        slug = slug_base
        suffix = 1
        while slug in used_slugs:
            suffix += 1
            slug = f"{slug_base}-{suffix}"
        used_slugs.add(slug)

        desc = f"[cyan]{combo.generator_name}/{combo.noise_name}/{combo.strategy_name}[/cyan]"
        est_duration = duration_estimates.get(
            (combo.generator_name, combo.noise_name, combo.strategy_name),
            1000.0,
        )
        task_id = monitor.register_task(desc, total=config.repeats, weight=est_duration)
        total_weighted_repeats += config.repeats * est_duration

        task_list.append(
            LocatorTask(
                combination=combo,
                repeats=config.repeats,
                seed=config.seed,
                slug=slug,
                out_dir=config.out_dir,
                scans_dir=config.scans_dir,
                bayes_dir=config.bayes_dir,
                loc_max_steps=config.loc_max_steps,
                loc_timeout_s=config.loc_timeout_s,
                use_cache=not config.no_cache,
                cache_dir=config.cache_dir,
                log_queue=config.log_queue,
                log_level=config.log_level_value,
                ignore_cache_strategy=config.ignore_cache_strategy,
                require_cache=config.require_cache,
                progress_queue=config.progress_queue,
                task_id=task_id,
            )
        )

    monitor.set_total_weight(total=total_weighted_repeats)
    return task_list, total_weighted_repeats


def _load_duration_estimates(out_dir: Path) -> dict[tuple[str, str, str], float]:
    """Load duration estimates from previous run metadata if available."""
    csv_path = out_dir / "locator_results.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pl.read_csv(csv_path)
        if "duration_ms" not in df.columns:
            return {}
        if not all(c in df.columns for c in ["generator", "noise", "strategy"]):
            return {}

        stats = df.group_by(["generator", "noise", "strategy"]).agg(pl.col("duration_ms").mean())
        return {(row["generator"], row["noise"], row["strategy"]): row["duration_ms"] for row in stats.to_dicts()}
    except Exception:
        return {}
