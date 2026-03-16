from __future__ import annotations

import queue
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nvision.cli.utils import (
    _get_generator_category,
    _load_duration_estimates,
    _locator_strategies_for_generator,
    _noise_presets,
)
from nvision.core.paths import slugify
from nvision.core.structures import LocatorTask
from nvision.sim import cases as sim_cases

if TYPE_CHECKING:
    from nvision.cli.monitor import ProgressMonitor


def build_tasks(
    repeats: int,
    seed: int,
    out_dir: Path,
    scans_dir: Path,
    bayes_dir: Path,
    cache_dir: Path,
    log_queue: queue.Queue,
    progress_queue: queue.Queue,
    log_level_value: int,
    loc_max_steps: int,
    loc_timeout_s: int,
    no_cache: bool,
    ignore_cache_strategy: str | None,
    require_cache: bool,
    filter_category: str | None,
    filter_strategy: str | None,
    monitor: ProgressMonitor,
) -> tuple[list[LocatorTask], float]:
    """Builds the list of tasks and registers them with the monitor, returning total weight."""
    generator_map = dict(sim_cases.generators_basic())
    noise_map = dict(_noise_presets())
    duration_estimates = _load_duration_estimates(out_dir)

    tasks: list[LocatorTask] = []
    seen_configs: set[tuple[str, str, str]] = set()
    used_slugs: set[str] = set()
    total_weighted_repeats = 0.0

    for gen_name, gen_obj in generator_map.items():
        if filter_category and _get_generator_category(gen_name) != filter_category:
            continue

        for strat_name, strat_obj in _locator_strategies_for_generator(gen_name):
            if filter_strategy and filter_strategy not in strat_name:
                continue

            for noise_name, noise_obj in noise_map.items():
                config_key = (gen_name, noise_name, strat_name)
                if config_key in seen_configs:
                    continue
                seen_configs.add(config_key)

                slug_base = "_".join(slugify(part) for part in (gen_name, noise_name, strat_name))
                slug_candidate = slug_base
                suffix = 1
                while slug_candidate in used_slugs:
                    suffix += 1
                    slug_candidate = f"{slug_base}-{suffix}"
                used_slugs.add(slug_candidate)

                desc = f"[cyan]{gen_name}/{noise_name}/{strat_name}[/cyan]"
                est_duration = duration_estimates.get((gen_name, noise_name, strat_name), 1000.0)

                task_id = monitor.register_task(desc, total=repeats, weight=est_duration)
                total_weighted_repeats += repeats * est_duration

                tasks.append(
                    LocatorTask(
                        generator_name=gen_name,
                        generator=gen_obj,
                        noise_name=noise_name,
                        noise=noise_obj,
                        strategy_name=strat_name,
                        strategy=strat_obj,
                        repeats=repeats,
                        seed=seed,
                        slug=slug_candidate,
                        out_dir=out_dir,
                        scans_dir=scans_dir,
                        bayes_dir=bayes_dir,
                        loc_max_steps=loc_max_steps,
                        loc_timeout_s=loc_timeout_s,
                        use_cache=not no_cache,
                        cache_dir=cache_dir,
                        log_queue=log_queue,
                        log_level=log_level_value,
                        ignore_cache_strategy=ignore_cache_strategy,
                        require_cache=require_cache,
                        progress_queue=progress_queue,
                        task_id=task_id,
                    )
                )

    monitor.set_total_weight(total=total_weighted_repeats)
    return tasks, total_weighted_repeats
