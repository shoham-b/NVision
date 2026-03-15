from __future__ import annotations

import logging
from typing import Any

from nvision.cache import CacheBridge
from nvision.cli.cache_helpers import embed_graph_content, restore_graphs
from nvision.cli.metrics_exporter import generate_attempt_metrics
from nvision.cli.plot_exporter import generate_attempt_plots
from nvision.cli.sim_runner import run_simulation_batch
from nvision.cli.utils import _get_generator_category
from nvision.core.types import LocatorTask
from nvision.viz import Viz


def _run_combination(task: LocatorTask) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    log = logging.getLogger("nvision")

    n_repeats = task.repeats
    gen_name = task.generator_name
    noise_name = task.noise_name
    strat_name = task.strategy_name
    main_seed = task.seed

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
        "generator": gen_name,
        "noise": noise_name,
        "strategy": strat_name,
        "repeats": n_repeats,
        "seed": main_seed,
        "max_steps": task.loc_max_steps,
        "timeout_s": task.loc_timeout_s,
    }

    skip_cache_for_strategy = task.ignore_cache_strategy is not None and strat_name == task.ignore_cache_strategy

    if task.require_cache:
        cached_results = cache.get_cached_results(combo_cfg)
        if cached_results:
            log.debug(f"Cache hit for {gen_name}/{noise_name}/{strat_name} (seed={main_seed}); restoring graphs.")
            restore_graphs(cached_results, task.out_dir, log)
            bridge.close()
            return cached_results
        log.warning(
            f"Cache missing for {gen_name}/{noise_name}/{strat_name} (seed={main_seed}) and --require-cache is set. Skipping."
        )
        bridge.close()
        return []

    if task.use_cache and not skip_cache_for_strategy:
        cached_results = cache.get_cached_results(combo_cfg)
        if cached_results:
            log.debug(f"Cache hit for {gen_name}/{noise_name}/{strat_name} (seed={main_seed}); restoring graphs.")
            restore_graphs(cached_results, task.out_dir, log)
            bridge.close()
            return cached_results

    # 1. Run full simulation batch
    final_history_df, finalize_results, initial_scans, repeat_start_times, repeat_stop_reasons = run_simulation_batch(
        task
    )

    all_results_for_combination = []

    # 2. Extract metrics and plots per repeat
    for attempt_idx_in_combo in range(n_repeats):
        current_scan = initial_scans[attempt_idx_in_combo]

        entry_base, main_result_row, current_history_df = generate_attempt_metrics(
            n_repeats=n_repeats,
            attempt_idx_in_combo=attempt_idx_in_combo,
            gen_name=gen_name,
            noise_name=noise_name,
            strat_name=strat_name,
            repeat_stop_reasons=repeat_stop_reasons,
            repeat_start_times=repeat_start_times,
            current_scan=current_scan,
            final_history_df=final_history_df,
            finalize_results=finalize_results,
            strat_obj=task.strategy,
        )

        entries = generate_attempt_plots(
            viz=viz,
            entry_base=entry_base,
            attempt_idx_in_combo=attempt_idx_in_combo,
            current_scan=current_scan,
            current_history_df=current_history_df,
            noise_obj=task.noise,
            strat_obj=task.strategy,
            slug_base=task.slug,
            out_dir=task.out_dir,
            scans_dir=task.scans_dir,
            bayes_dir=task.bayes_dir,
        )

        all_results_for_combination.append((entries, main_result_row))

        if task.use_cache and not skip_cache_for_strategy:
            part_cfg = {
                "kind": "locator_run",
                "generator": gen_name,
                "noise": noise_name,
                "strategy": strat_name,
                "repeat": attempt_idx_in_combo,
                "seed": main_seed,
                "max_steps": task.loc_max_steps,
                "timeout_s": task.loc_timeout_s,
            }
            entries_to_cache = embed_graph_content(entries, task.out_dir, log)
            cache.save_cached_repeat(part_cfg, entries_to_cache, main_result_row)

        if task.progress_queue:
            task.progress_queue.put((task.task_id, 1))
        else:
            log.debug(f"Finished repeat {attempt_idx_in_combo + 1} for {gen_name}/{noise_name}")

    if task.use_cache and not skip_cache_for_strategy and all_results_for_combination:
        full_results_with_content = []
        for entries, res_row in all_results_for_combination:
            entries_with_content = embed_graph_content(entries, task.out_dir, log)
            full_results_with_content.append((entries_with_content, res_row))

        cache.save_cached_results(combo_cfg, full_results_with_content)

    bridge.close()
    return all_results_for_combination
