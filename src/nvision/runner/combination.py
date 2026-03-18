"""Combination orchestrator — runs one (generator × noise × strategy) combination.

Handles cache lookup/save, delegates to the batch runner, and collects
per-repeat metrics and plots.
"""

from __future__ import annotations

import logging
from typing import Any

from nvision.cache import CacheBridge
from nvision.models.task import LocatorTask
from nvision.runner.batch import run_simulation_batch
from nvision.runner.cache import embed_graph_content, restore_graphs
from nvision.runner.metrics import generate_attempt_metrics
from nvision.runner.plots import generate_attempt_plots
from nvision.tools.utils import _get_generator_category
from nvision.viz import Viz

log = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 2


def run_combination(task: LocatorTask) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Execute one (generator, noise, strategy) combination for all repeats.

    Checks the cache before running. Saves results to cache after running.
    Returns a list of ``(entries, result_row)`` tuples — one per repeat.

    Parameters
    ----------
    task : LocatorTask
        Full task configuration.

    Returns
    -------
    list[tuple[list[dict], dict]]
        Per-repeat ``(plot_entries, metrics_row)`` pairs.
        Empty list when ``require_cache`` is set but no cache entry exists.
    """

    gen_name = task.generator_name
    noise_name = task.noise_name
    strat_name = task.strategy_name
    n_repeats = task.repeats

    log.info(
        "Starting combination: %s/%s/%s for %s repeats",
        gen_name,
        noise_name,
        strat_name,
        n_repeats,
    )

    category = _get_generator_category(gen_name)
    bridge = CacheBridge(task.cache_dir)
    cache = bridge.get_cache_for_category(category)
    viz = Viz(task.out_dir / "graphs")

    combo_cfg = {
        "kind": "locator_combination",
        "schema_version": CACHE_SCHEMA_VERSION,
        "generator": gen_name,
        "noise": noise_name,
        "strategy": strat_name,
        "repeats": n_repeats,
        "seed": task.seed,
        "max_steps": task.loc_max_steps,
        "timeout_s": task.loc_timeout_s,
    }

    skip_cache = task.ignore_cache_strategy is not None and strat_name == task.ignore_cache_strategy

    # ── Cache read ────────────────────────────────────────────────────────────
    if task.require_cache:
        cached = cache.get_cached_results(combo_cfg)
        if cached:
            log.debug(
                "Cache hit for %s/%s/%s (seed=%s); restoring graphs.", gen_name, noise_name, strat_name, task.seed
            )
            restore_graphs(cached, task.out_dir)
            bridge.close()
            return cached
        log.warning(
            "Cache missing for %s/%s/%s (seed=%s) and --require-cache is set. Skipping.",
            gen_name,
            noise_name,
            strat_name,
            task.seed,
        )
        bridge.close()
        return []

    if task.use_cache and not skip_cache:
        cached = cache.get_cached_results(combo_cfg)
        if cached:
            log.debug(
                "Cache hit for %s/%s/%s (seed=%s); restoring graphs.", gen_name, noise_name, strat_name, task.seed
            )
            restore_graphs(cached, task.out_dir)
            bridge.close()
            return cached

    # ── Simulation ────────────────────────────────────────────────────────────
    history_df, finalize_df, experiments, repeat_start_times, stop_reasons = run_simulation_batch(task)

    all_results: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    for attempt_idx in range(n_repeats):
        current_scan = experiments[attempt_idx]

        entry_base, main_result_row, current_history_df = generate_attempt_metrics(
            n_repeats=n_repeats,
            attempt_idx_in_combo=attempt_idx,
            gen_name=gen_name,
            noise_name=noise_name,
            strat_name=strat_name,
            repeat_stop_reasons=stop_reasons,
            repeat_start_times=repeat_start_times,
            current_scan=current_scan,
            final_history_df=history_df,
            finalize_results=finalize_df,
            strat_obj=task.strategy,
        )

        entries = generate_attempt_plots(
            viz=viz,
            entry_base=entry_base,
            attempt_idx_in_combo=attempt_idx,
            current_scan=current_scan,
            current_history_df=current_history_df,
            noise_obj=task.noise,
            strat_obj=task.strategy,
            slug_base=task.slug,
            out_dir=task.out_dir,
            scans_dir=task.scans_dir,
            bayes_dir=task.bayes_dir,
        )

        all_results.append((entries, main_result_row))

        # ── Per-repeat cache write ────────────────────────────────────────────
        if task.use_cache and not skip_cache:
            part_cfg = {
                "kind": "locator_run",
                "schema_version": CACHE_SCHEMA_VERSION,
                "generator": gen_name,
                "noise": noise_name,
                "strategy": strat_name,
                "repeat": attempt_idx,
                "seed": task.seed,
                "max_steps": task.loc_max_steps,
                "timeout_s": task.loc_timeout_s,
            }
            cache.save_cached_repeat(part_cfg, embed_graph_content(entries, task.out_dir), main_result_row)

        if task.progress_queue:
            task.progress_queue.put((task.task_id, 1))
        else:
            log.debug("Finished repeat %s for %s/%s/%s", attempt_idx + 1, gen_name, noise_name, strat_name)

    # ── Full-combination cache write ──────────────────────────────────────────
    if task.use_cache and not skip_cache and all_results:
        full_results = [(embed_graph_content(entries, task.out_dir), row) for entries, row in all_results]
        cache.save_cached_results(combo_cfg, full_results)

    bridge.close()
    return all_results
