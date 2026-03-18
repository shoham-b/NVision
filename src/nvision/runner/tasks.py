"""Task list builder — constructs LocatorTask objects for every scenario combination."""

from __future__ import annotations

import queue
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from nvision.models.task import LocatorTask
from nvision.sim import cases as sim_cases
from nvision.tools.paths import slugify
from nvision.tools.utils import (
    _get_generator_category,
    _locator_strategies_for_generator,
    _noise_presets,
)

if TYPE_CHECKING:
    from nvision.cli.monitor import ProgressMonitor


@dataclass
class TaskBuildConfig:
    """Configuration for building the scenario task grid.

    Collects all CLI-level knobs in one place so ``build_tasks`` has a
    single-argument signature and callsites stay readable.
    """

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


def build_tasks(
    config: TaskBuildConfig,
    monitor: ProgressMonitor,
) -> tuple[list[LocatorTask], float]:
    """Build the full list of LocatorTask objects and register them with the monitor.

    Returns
    -------
    tuple[list[LocatorTask], float]
        ``(tasks, total_weighted_repeats)`` where ``total_weighted_repeats``
        is the sum of ``repeats × estimated_duration`` across all tasks.
    """
    generator_map = dict(sim_cases.generators_basic())
    noise_map = dict(_noise_presets())
    duration_estimates = _load_duration_estimates(config.out_dir)

    tasks: list[LocatorTask] = []
    seen_configs: set[tuple[str, str, str]] = set()
    used_slugs: set[str] = set()
    total_weighted_repeats = 0.0

    for gen_name, gen_obj in generator_map.items():
        if config.filter_category and _get_generator_category(gen_name) != config.filter_category:
            continue

        for strat_name, strat_obj in _locator_strategies_for_generator(gen_name):
            if config.filter_strategy and config.filter_strategy not in strat_name:
                continue

            for noise_name, noise_obj in noise_map.items():
                config_key = (gen_name, noise_name, strat_name)
                if config_key in seen_configs:
                    continue
                seen_configs.add(config_key)

                slug_base = "_".join(slugify(p) for p in (gen_name, noise_name, strat_name))
                slug = slug_base
                suffix = 1
                while slug in used_slugs:
                    suffix += 1
                    slug = f"{slug_base}-{suffix}"
                used_slugs.add(slug)

                desc = f"[cyan]{gen_name}/{noise_name}/{strat_name}[/cyan]"
                est_duration = duration_estimates.get((gen_name, noise_name, strat_name), 1000.0)
                task_id = monitor.register_task(desc, total=config.repeats, weight=est_duration)
                total_weighted_repeats += config.repeats * est_duration

                tasks.append(
                    LocatorTask(
                        generator_name=gen_name,
                        generator=gen_obj,
                        noise_name=noise_name,
                        noise=noise_obj,
                        strategy_name=strat_name,
                        strategy=strat_obj,
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
    return tasks, total_weighted_repeats


def _load_duration_estimates(out_dir: Path) -> dict[tuple[str, str, str], float]:
    """Load duration estimates from previous run metadata if available."""
    csv_path = out_dir / "locator_results.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pl.read_csv(csv_path)
        if "duration_ms" not in df.columns:
            return {}

        # Group by config and take mean duration
        # We assume columns generator, noise, strategy exist
        if not all(c in df.columns for c in ["generator", "noise", "strategy"]):
            return {}

        stats = df.group_by(["generator", "noise", "strategy"]).agg(pl.col("duration_ms").mean())

        estimates = {}
        for row in stats.to_dicts():
            key = (row["generator"], row["noise"], row["strategy"])
            estimates[key] = row["duration_ms"]
        return estimates
    except Exception:
        return {}
