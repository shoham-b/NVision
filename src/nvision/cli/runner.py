from __future__ import annotations

import copy
import hashlib
import logging
import math
import random
import time
from pathlib import Path

import polars as pl

from nvision.cache import CacheBridge
from nvision.cli.utils import (
    _get_generator_category,
    _maybe_finite,
    _scan_attempt_metrics,
)
from nvision.core.paths import slugify
from nvision.core.types import LocatorTask
from nvision.sim import (
    NVCenterSequentialBayesianLocator,
)
from nvision.sim.locs.nv_center.evaluation import BayesianMetrics
from nvision.viz import Viz


def _category_cache_dir(base: Path, category: str) -> Path:
    slug = slugify(category or "unknown")
    return base / slug


def _restore_graphs(cached_results: list, out_dir: Path, log: logging.Logger) -> None:
    """Restore cached graph content to files."""
    count = 0
    try:
        for entries, _ in cached_results:
            for entry in entries:
                if "path" in entry and "content" in entry:
                    file_path = out_dir / entry["path"]
                    if not file_path.exists():
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(entry["content"], encoding="utf-8")
                        count += 1
        if count > 0:
            log.debug(f"Restored {count} graph files from cache.")
        else:
            log.debug("No graph content found in cache entries to restore.")
    except Exception as e:
        log.warning(f"Failed to restore cached graphs: {e}")


def _embed_graph_content(entries: list[dict], out_dir: Path, log: logging.Logger) -> list[dict]:
    """Embed graph file content into entries for caching."""
    entries_with_content = copy.deepcopy(entries)
    count = 0
    for entry in entries_with_content:
        if "path" in entry:
            try:
                file_path = out_dir / entry["path"]
                if file_path.exists():
                    entry["content"] = file_path.read_text(encoding="utf-8")
                    count += 1
            except Exception as e:
                log.warning(f"Failed to read graph content for caching: {e}")
    if count > 0:
        log.debug(f"Embedded content for {count} graph files.")
    return entries_with_content


def _run_combination(task: LocatorTask):  # noqa: C901
    gen_name = task.generator_name
    gen_obj = task.generator
    noise_name = task.noise_name
    noise_obj = task.noise
    strat_name = task.strategy_name
    strat_obj = task.strategy
    n_repeats = task.repeats
    main_seed = task.seed
    slug_base = task.slug
    out_dir = task.out_dir
    scans_dir = task.scans_dir
    bayes_dir = task.bayes_dir
    loc_max_steps = task.loc_max_steps
    loc_timeout_s = task.loc_timeout_s
    use_cache = task.use_cache
    cache_dir = task.cache_dir
    ignore_cache_strategy = task.ignore_cache_strategy
    require_cache = task.require_cache
    log = logging.getLogger("nvision")

    log.info(
        "Starting combination: %s/%s/%s for %s repeats",
        gen_name,
        noise_name,
        strat_name,
        n_repeats,
    )

    category = _get_generator_category(gen_name)
    # cache_category_dir = _category_cache_dir(cache_dir, category)  # No longer needed with CacheBridge
    bridge = CacheBridge(cache_dir)
    cache = bridge.get_cache_for_category(category)
    graphs_dir = out_dir / "graphs"
    viz = Viz(graphs_dir)

    combo_cfg = {
        "kind": "locator_combination",
        "generator": gen_name,
        "noise": noise_name,
        "strategy": strat_name,
        "repeats": n_repeats,
        "seed": main_seed,
        "max_steps": loc_max_steps,
        "timeout_s": loc_timeout_s,
    }

    # Check if cache should be ignored for this specific strategy
    skip_cache_for_strategy = ignore_cache_strategy is not None and strat_name == ignore_cache_strategy

    if require_cache:
        cached_results = cache.get_cached_results(combo_cfg)
        if cached_results:
            log.debug(
                "Cache hit for %s/%s/%s (seed=%s); restoring graphs and skipping simulation.",
                gen_name,
                noise_name,
                strat_name,
                main_seed,
            )
            _restore_graphs(cached_results, out_dir, log)
            return cached_results

        # If we are here, cache is missing or invalid, but require_cache is True
        log.warning(
            "Cache missing for %s/%s/%s (seed=%s) and --require-cache is set. Skipping.",
            gen_name,
            noise_name,
            strat_name,
            main_seed,
        )
        return []

    if use_cache and not skip_cache_for_strategy:
        cached_results = cache.get_cached_results(combo_cfg)
        if cached_results:
            log.debug(
                "Cache hit for %s/%s/%s (seed=%s); restoring graphs and skipping simulation.",
                gen_name,
                noise_name,
                strat_name,
                main_seed,
            )
            _restore_graphs(cached_results, out_dir, log)
            return cached_results

        # Original code had a "corrupted cache" warning if JSON parse failed.
        # get_cached_results returns None on failure.
        # We can just proceed to recompute.

        # Restore cached graphs if available
        if cached_results:
            log.debug(
                "Cache hit for %s/%s/%s (seed=%s); restoring graphs and skipping simulation.",
                gen_name,
                noise_name,
                strat_name,
                main_seed,
            )
            try:
                for entries, _ in cached_results:
                    for entry in entries:
                        if "path" in entry and "content" in entry:
                            file_path = out_dir / entry["path"]
                            if not file_path.exists():
                                file_path.parent.mkdir(parents=True, exist_ok=True)
                                file_path.write_text(entry["content"], encoding="utf-8")
            except Exception as e:
                log.warning(f"Failed to restore cached graphs: {e}")

            return cached_results

    all_results_for_combination = []

    # Initialize state for all repeats (batched)
    repeat_rngs = []
    initial_scans = []
    repeat_start_times = []
    repeat_stop_reasons = ["" for _ in range(n_repeats)]

    for attempt_idx_in_combo in range(n_repeats):
        combo_seed_str = f"{main_seed}-{gen_name}-{strat_name}-{noise_name}-{attempt_idx_in_combo}"
        attempt_seed = int(hashlib.sha256(combo_seed_str.encode("utf-8")).hexdigest(), 16) % (10**8)

        repeat_rngs.append(random.Random(attempt_seed))
        initial_scans.append(gen_obj.generate(repeat_rngs[-1]))
        repeat_start_times.append(time.perf_counter())

    # Batched history and repeat tracking
    history_rows = []
    repeats_df = pl.DataFrame(
        {
            "repeat_id": list(range(n_repeats)),
            "active": [True] * n_repeats,
        }
    )

    # Lockstep simulation loop using batched locator interface
    global_start_time = time.perf_counter()
    for step_num in range(loc_max_steps):
        active_repeats = repeats_df.filter(pl.col("active")).get_column("repeat_id").to_list()
        if not active_repeats:
            break

        if time.perf_counter() - global_start_time > loc_timeout_s:
            log.warning(f"Combination timeout ({loc_timeout_s}s) reached. Finalizing.")
            for rid in active_repeats:
                if not repeat_stop_reasons[rid]:
                    repeat_stop_reasons[rid] = "combination_timeout"
            break

        # Build current history DataFrame
        if history_rows:
            history_df = pl.DataFrame(history_rows)
        else:
            history_df = pl.DataFrame(
                {
                    "repeat_id": pl.Series("repeat_id", [], dtype=pl.Int64),
                    "step": pl.Series("step", [], dtype=pl.Int64),
                    "x": pl.Series("x", [], dtype=pl.Float64),
                    "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
                }
            )

        # Check which repeats should stop
        stop_decisions = strat_obj.should_stop(history_df, repeats_df, initial_scans[0])
        for row_dict in stop_decisions.to_dicts():
            rid = row_dict["repeat_id"]
            if row_dict["stop"] and rid in active_repeats:
                repeats_df = repeats_df.with_columns(
                    pl.when(pl.col("repeat_id") == rid).then(pl.lit(False)).otherwise(pl.col("active")).alias("active")
                )
                if not repeat_stop_reasons[rid]:
                    repeat_stop_reasons[rid] = "locator_stop"

        # Refresh active list after stop check
        active_repeats = repeats_df.filter(pl.col("active")).get_column("repeat_id").to_list()
        if not active_repeats:
            break

        # Propose next measurements for active repeats
        proposals = strat_obj.propose_next(history_df, repeats_df, initial_scans[0])

        # Execute measurements for each active repeat
        for row_dict in proposals.to_dicts():
            rid = row_dict["repeat_id"]
            if rid not in active_repeats:
                continue

            x_next = row_dict["x"]
            current_scan = initial_scans[rid]
            y_ideal = current_scan.signal(x_next)

            y_measured = (
                noise_obj.over_probe_noise.apply(y_ideal, repeat_rngs[rid], strat_obj)
                if noise_obj and noise_obj.over_probe_noise
                else y_ideal
            )

            row_data = {
                "repeat_id": rid,
                "step": step_num,
                "x": x_next,
                "signal_values": y_measured,
            }

            # Capture Bayesian metrics if available
            if hasattr(strat_obj, "_get_locator"):
                try:
                    locator_instance = strat_obj._get_locator(rid)
                    if hasattr(locator_instance, "current_estimates"):
                        est = locator_instance.current_estimates
                        if "entropy" in est:
                            row_data["entropy"] = est["entropy"]
                        if "max_prob" in est:
                            row_data["max_prob"] = est["max_prob"]
                        if "uncertainty" in est:
                            row_data["uncertainty"] = est["uncertainty"]
                except Exception:
                    pass  # Ignore if accessing locator fails

            history_rows.append(row_data)

    # Mark remaining active repeats as max_steps_reached
    for rid in range(n_repeats):
        if not repeat_stop_reasons[rid]:
            repeat_stop_reasons[rid] = "max_steps_reached"

    # Build final history DataFrame
    if history_rows:
        final_history_df = pl.DataFrame(history_rows)
    else:
        final_history_df = pl.DataFrame(
            {
                "repeat_id": pl.Series("repeat_id", [], dtype=pl.Int64),
                "step": pl.Series("step", [], dtype=pl.Int64),
                "x": pl.Series("x", [], dtype=pl.Float64),
                "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
            }
        )

    # Finalize all repeats at once
    finalize_results = strat_obj.finalize(final_history_df, repeats_df, initial_scans[0])

    # Collect results for each repeat
    for attempt_idx_in_combo in range(n_repeats):
        current_scan = initial_scans[attempt_idx_in_combo]

        # Extract history for this repeat
        if not final_history_df.is_empty():
            current_history_df = final_history_df.filter(pl.col("repeat_id") == attempt_idx_in_combo).drop("repeat_id")
        else:
            current_history_df = pl.DataFrame(
                {
                    "x": pl.Series("x", [], dtype=pl.Float64),
                    "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
                }
            )

        if current_history_df.is_empty():
            log.info(
                "No measurements recorded for repeat %s (reason=%s); generating baseline scan plot.",
                attempt_idx_in_combo + 1,
                repeat_stop_reasons[attempt_idx_in_combo],
            )

        duration_ms = (time.perf_counter() - repeat_start_times[attempt_idx_in_combo]) * 1000

        # Extract finalize result for this repeat
        finalize_row = finalize_results.filter(pl.col("repeat_id") == attempt_idx_in_combo)
        if finalize_row.is_empty() or current_history_df.is_empty():
            estimate = {"x_hat": math.nan, "uncert": math.inf}
            measurements = 0
        else:
            estimate_dict = finalize_row.drop("repeat_id").to_dicts()[0]
            estimate = {k: float(v) for k, v in estimate_dict.items()}
            measurements = current_history_df.height

        attempt_metrics = _scan_attempt_metrics(current_scan.truth_positions, estimate)
        attempt_slug = f"{slug_base}_r{attempt_idx_in_combo + 1}"
        out_path = scans_dir / f"{attempt_slug}.html"

        viz.plot_scan_measurements(
            current_scan,
            current_history_df,
            out_path,
            over_frequency_noise=noise_obj.over_frequency_noise if noise_obj else None,
        )

        metrics_serialized = {key: _maybe_finite(value) for key, value in attempt_metrics.items()}
        metrics_serialized["measurements"] = _maybe_finite(measurements)
        metrics_serialized["duration_ms"] = _maybe_finite(duration_ms)

        entries: list[dict[str, object]] = [
            {
                "type": "scan",
                "generator": gen_name,
                "noise": noise_name,
                "strategy": strat_name,
                "repeat": attempt_idx_in_combo + 1,
                "repeat_total": n_repeats,
                "stop_reason": repeat_stop_reasons[attempt_idx_in_combo],
                "abs_err_x": metrics_serialized.get("abs_err_x"),
                "uncert": metrics_serialized.get("uncert"),
                "measurements": metrics_serialized.get("measurements"),
                "duration_ms": metrics_serialized.get("duration_ms"),
                "metrics": metrics_serialized,
                "path": out_path.relative_to(out_dir).as_posix(),
            }
        ]

        # Bayesian plotting
        if hasattr(strat_obj, "_get_locator"):
            try:
                locator_instance = strat_obj._get_locator(attempt_idx_in_combo)
                if hasattr(locator_instance, "posterior_history") and hasattr(locator_instance, "freq_grid"):
                    # Compute model history
                    model_history = []
                    if hasattr(locator_instance, "parameter_history"):
                        for params in locator_instance.parameter_history:
                            model_history.append(locator_instance.odmr_model(locator_instance.freq_grid, params))

                    bayes_anim_path = bayes_dir / f"{attempt_slug}_posterior_anim.html"
                    viz.plot_posterior_animation(
                        locator_instance.posterior_history,
                        locator_instance.freq_grid,
                        bayes_anim_path,
                        model_history=model_history,
                    )
                    entries.append(
                        {
                            "type": "bayesian_interactive",
                            "generator": gen_name,
                            "noise": noise_name,
                            "strategy": strat_name,
                            "repeat": attempt_idx_in_combo + 1,
                            "path": bayes_anim_path.relative_to(out_dir).as_posix(),
                        }
                    )

                    if hasattr(locator_instance, "parameter_history"):
                        param_conv_path = bayes_dir / f"{attempt_slug}_param_convergence.html"
                        viz.plot_parameter_convergence(
                            locator_instance.parameter_history,
                            param_conv_path,
                        )
                        entries.append(
                            {
                                "type": "bayesian_parameter_convergence",
                                "generator": gen_name,
                                "noise": noise_name,
                                "strategy": strat_name,
                                "repeat": attempt_idx_in_combo + 1,
                                "path": param_conv_path.relative_to(out_dir).as_posix(),
                            }
                        )
            except Exception as e:
                log.warning(f"Failed to generate Bayesian animation: {e}")
        elif isinstance(strat_obj, NVCenterSequentialBayesianLocator):
            try:
                # Reconstruct state for this repeat
                strat_obj.reset_run_state()
                strat_obj._ingest_history(current_history_df)

                if hasattr(strat_obj, "posterior_history") and hasattr(strat_obj, "freq_grid"):
                    # Compute model history
                    model_history = []
                    if hasattr(strat_obj, "parameter_history"):
                        for params in strat_obj.parameter_history:
                            model_history.append(strat_obj.odmr_model(strat_obj.freq_grid, params))

                    bayes_anim_path = bayes_dir / f"{attempt_slug}_posterior_anim.html"
                    viz.plot_posterior_animation(
                        strat_obj.posterior_history,
                        strat_obj.freq_grid,
                        bayes_anim_path,
                        model_history=model_history,
                    )
                    entries.append(
                        {
                            "type": "bayesian_interactive",
                            "generator": gen_name,
                            "noise": noise_name,
                            "strategy": strat_name,
                            "repeat": attempt_idx_in_combo + 1,
                            "path": bayes_anim_path.relative_to(out_dir).as_posix(),
                        }
                    )

                    if hasattr(strat_obj, "parameter_history"):
                        param_conv_path = bayes_dir / f"{attempt_slug}_param_convergence.html"
                        viz.plot_parameter_convergence(
                            strat_obj.parameter_history,
                            param_conv_path,
                        )
                        entries.append(
                            {
                                "type": "bayesian_parameter_convergence",
                                "generator": gen_name,
                                "noise": noise_name,
                                "strategy": strat_name,
                                "repeat": attempt_idx_in_combo + 1,
                                "path": param_conv_path.relative_to(out_dir).as_posix(),
                            }
                        )
            except Exception as e:
                log.warning(f"Failed to generate Bayesian animation for single locator: {e}")

        # Calculate extended Bayesian metrics for this repeat
        try:
            # Reconstruct locator or retrieve from strategy if possible
            locator_for_metrics = None
            if hasattr(strat_obj, "_get_locator"):
                locator_for_metrics = strat_obj._get_locator(attempt_idx_in_combo)
            elif isinstance(strat_obj, NVCenterSequentialBayesianLocator):
                # For single locator, state might be reset or overwritten, need care
                # But here we are just after finalize, so state should be "final" for this repeat?
                # Actually strat_obj is shared for single locator if not carefully managed.
                # But our loop re-initializes it or we can re-ingest.
                # strat_obj.reset_run_state() # Don't reset if we want to use it
                # strat_obj._ingest_history(current_history_df) # Already done in finalize?
                locator_for_metrics = strat_obj

            if locator_for_metrics and hasattr(locator_for_metrics, "parameter_history"):
                # Initial ground truth from scan
                # Extract ground truth from scan or its metadata
                meta = getattr(current_scan, "meta", {}) or {}

                # NVCenterManufacturer uses 'omega' for HWHM, but CRB kernel expects FWHM 'linewidth'
                linewidth = getattr(current_scan, "linewidth", None)
                if linewidth is None and "omega" in meta:
                    # Look for omega (HWHM) -> convert to FWHM
                    val = meta["omega"]
                    if val is not None:
                        linewidth = val * 2.0

                amplitude = getattr(current_scan, "amplitude", None)
                if amplitude is None:
                    amplitude = meta.get("a")

                background = getattr(current_scan, "background", None)
                if background is None:
                    background = meta.get("base")

                ground_truth = {
                    "frequency": current_scan.truth_positions[0]
                    if current_scan.truth_positions
                    else meta.get("f0", meta.get("f_B")),
                    "linewidth": linewidth,
                    "amplitude": amplitude,
                    "background": background,
                }
                # Filter None values
                ground_truth = {k: v for k, v in ground_truth.items() if v is not None}

                metrics_obj = BayesianMetrics.from_history(
                    parameter_history=locator_for_metrics.parameter_history,
                    ground_truth=ground_truth,
                    measurement_history=current_history_df,
                    noise_model=noise_name if "poisson" in noise_name.lower() else "gaussian",
                    noise_params={"sigma": 0.05} if "gaussian" in noise_name.lower() else None,
                )

                bayes_summary = metrics_obj.summary()
                # Merge into metrics_serialized
                for k, v in bayes_summary.items():
                    metrics_serialized[k] = _maybe_finite(v)

        except Exception as e:
            log.warning(f"Failed to calculate Bayesian metrics for repeat {attempt_idx_in_combo + 1}: {e}")

        main_result_row = {
            "generator": gen_name,
            "noise": noise_name,
            "strategy": strat_name,
            "repeats": n_repeats,
            "attempt": attempt_idx_in_combo + 1,
            "stop_reason": repeat_stop_reasons[attempt_idx_in_combo],
            **metrics_serialized,
        }

        if use_cache and not skip_cache_for_strategy:
            part_cfg = {
                "kind": "locator_run",
                "generator": gen_name,
                "noise": noise_name,
                "strategy": strat_name,
                "repeat": attempt_idx_in_combo,
                "seed": main_seed,
                "max_steps": loc_max_steps,
                "timeout_s": loc_timeout_s,
            }
            # Embed graph content for caching
            entries_to_cache = _embed_graph_content(entries, out_dir, log)
            cache.save_cached_repeat(part_cfg, entries_to_cache, main_result_row)

        if task.progress_queue:
            task.progress_queue.put((task.task_id, 1))
        else:
            log.debug(f"Finished repeat {attempt_idx_in_combo + 1} for {gen_name}/{noise_name}")
        all_results_for_combination.append((entries, main_result_row))

    if use_cache and not skip_cache_for_strategy and all_results_for_combination:
        # Re-construct full results with content for saving full combination
        # The repeats were already saved with content, but save_cached_results takes a list
        # We need to make sure the list used here also has content if we want consistent saving behavior

        full_results_with_content = []
        for entries, res_row in all_results_for_combination:
            entries_with_content = _embed_graph_content(entries, out_dir, log)
            full_results_with_content.append((entries_with_content, res_row))

        cache.save_cached_results(combo_cfg, full_results_with_content)

    return all_results_for_combination
