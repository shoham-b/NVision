from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing
import random
import shutil
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
import typer
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from nvision.cache import CategoryCache, SimulationCache
from nvision.index_html import compile_html_index
from nvision.pathutils import ensure_out_dir, slugify
from nvision.sim import (
    AnalyticalBayesianLocator,
    CompositeNoise,
    NVCenterSequentialBayesianLocator,
    NVCenterSweepLocator,
    OnePeakGoldenLocator,
    OnePeakGridLocator,
    OnePeakSweepLocator,
    ProjectBayesianLocator,
    SimpleSequentialLocator,
    TwoPeakGoldenLocator,
    TwoPeakGridLocator,
    TwoPeakSweepLocator,
)
from nvision.sim import cases as sim_cases
from nvision.sim.locs.nv_center.test_bayesian_locator import TestBayesianLocator
from nvision.viz import Viz

log = logging.getLogger("nvision")

app = typer.Typer(help="NVision simulation runner")


class DotsColumn(ProgressColumn):
    def render(self, task: Task) -> Any:
        return "." * int(task.completed)


@dataclass(slots=True)
class LocatorTask:
    generator_name: str
    generator: Any
    noise_name: str
    noise: CompositeNoise | None
    strategy_name: str
    strategy: Any
    repeats: int
    seed: int
    slug: str
    out_dir: Path
    scans_dir: Path
    bayes_dir: Path
    loc_max_steps: int
    loc_timeout_s: int
    use_cache: bool
    cache_dir: Path
    log_queue: Any
    log_level: int
    ignore_cache_strategy: str | None
    require_cache: bool = False
    progress_queue: Any = None
    task_id: Any = None

    def __str__(self) -> str:
        return self.slug


def _noise_presets() -> list[tuple[str, CompositeNoise | None]]:
    """Return the predefined noise combinations for scenarios."""
    return sim_cases.noises_none() + sim_cases.noises_single_each() + sim_cases.noises_complex()


def _to_native(obj: Any) -> Any:
    """Recursively convert NumPy scalars to native Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_native(v) for v in obj]
    return obj


def _get_generator_category(generator_name: str) -> str:
    """Determine the category of a generator from its name."""
    if generator_name.startswith("OnePeak-"):
        return "OnePeak"
    elif generator_name.startswith("TwoPeak-"):
        return "TwoPeak"
    elif generator_name.startswith("NVCenter-"):
        return "NVCenter"
    return "Unknown"


def _locator_strategies_for_generator(generator_name: str) -> list[tuple[str, object]]:
    """Get the appropriate locator strategies for a given generator category."""
    category = _get_generator_category(generator_name)
    strategies: list[tuple[str, object]] = []
    if category == "OnePeak":
        strategies = [
            ("OnePeak-Grid", OnePeakGridLocator(n_points=21)),
            ("OnePeak-Golden", OnePeakGoldenLocator(max_evals=25)),
            ("OnePeak-Sweep", OnePeakSweepLocator(coarse_points=20, refine_points=10)),
        ]
    elif category == "TwoPeak":
        strategies = [
            ("TwoPeak-Grid", TwoPeakGridLocator(coarse_points=25)),
            ("TwoPeak-Golden", TwoPeakGoldenLocator(coarse_points=25, refine_points=5)),
            ("TwoPeak-Sweep", TwoPeakSweepLocator(coarse_points=25, refine_points=10)),
        ]
    elif category == "NVCenter":
        strategies = [
            ("NVCenter-Sweep", NVCenterSweepLocator(coarse_points=30, refine_points=10)),
            (
                "NVCenter-SequentialBayesian",
                NVCenterSequentialBayesianLocator(
                    max_evals=500, grid_resolution=400, distribution="voigt-zeeman"
                ),
            ),
            (
                "NVCenter-SimpleSequential",
                SimpleSequentialLocator(max_evals=60, grid_resolution=400),
            ),
            (
                "NVCenter-ProjectBayesian",
                ProjectBayesianLocator(
                    max_evals=500, grid_resolution=400, distribution="voigt-zeeman"
                ),
            ),
            (
                "NVCenter-AnalyticalBayesian",
                AnalyticalBayesianLocator(
                    max_evals=500,
                    grid_resolution=400,
                    distribution="voigt-zeeman",
                    n_warmup=20,
                ),
            ),
            (
                "NVCenter-TestBayesian",
                TestBayesianLocator(max_evals=20),
            ),
        ]
    return strategies


def _maybe_finite(value: object) -> float | None:
    if isinstance(value, int | float):
        value_float = float(value)
        if math.isfinite(value_float):
            return value_float
    return None


def _recalculate_subset_result(
    result_row: dict,
    entries: list[dict],
    scan: Any,
    max_steps: int,
) -> tuple[list[dict], dict] | None:
    param_hist = result_row.get("parameter_history")
    if not param_hist or len(param_hist) < max_steps:
        return None

    estimate_raw = param_hist[max_steps - 1]
    estimate = {k: float(v) for k, v in estimate_raw.items()}

    # Recalculate metrics
    truth = getattr(scan, "truth_positions", [])
    metrics = _scan_attempt_metrics(truth, estimate)

    new_res = result_row.copy()
    new_res.update({k: _maybe_finite(v) for k, v in metrics.items()})

    return entries, new_res


def _scan_attempt_metrics(
    truth_positions: Sequence[float], estimate: dict[str, object]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    truth = [float(pos) for pos in truth_positions]

    if len(truth) == 1:
        x_hat = estimate.get("x1_hat", estimate.get("x_hat"))
        if isinstance(x_hat, int | float) and math.isfinite(float(x_hat)):
            metrics["abs_err_x"] = abs(float(x_hat) - truth[0])
    else:
        x1_hat = estimate.get("x1_hat")
        x2_hat = estimate.get("x2_hat")
        if (
            isinstance(x1_hat, int | float)
            and math.isfinite(float(x1_hat))
            and isinstance(x2_hat, int | float)
            and math.isfinite(float(x2_hat))
        ):
            xs = sorted([float(x1_hat), float(x2_hat)])
            truth_sorted = sorted(truth)
            err1 = abs(xs[0] - truth_sorted[0])
            err2 = abs(xs[1] - truth_sorted[1])
            metrics["abs_err_x1"] = err1
            metrics["abs_err_x2"] = err2
            metrics["pair_rmse"] = math.sqrt(0.5 * (err1 * err1 + err2 * err2))

    for key in ("uncert", "uncert_pos", "uncert_sep", "final_entropy", "final_max_prob"):
        value = estimate.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            metrics[key] = float(value)

    return metrics


def _ensure_worker_queue_logging(queue: Any, level: int) -> None:
    """Attach a multiprocessing QueueHandler exactly once per worker process."""
    root_logger = logging.getLogger()
    if getattr(root_logger, "_nvision_queue_handler_attached", False):
        root_logger.setLevel(level)
        return

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    queue_handler = QueueHandler(queue)
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(level)
    root_logger._nvision_queue_handler_attached = True  # type: ignore[attr-defined]


def _run_combination(task: LocatorTask):  # noqa: C901
    if task.log_queue:
        _ensure_worker_queue_logging(task.log_queue, task.log_level)

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

    log.info(
        "Starting combination: %s/%s/%s for %s repeats",
        gen_name,
        noise_name,
        strat_name,
        n_repeats,
    )

    category = _get_generator_category(gen_name)
    category = _get_generator_category(gen_name)
    sim_cache = SimulationCache(cache_dir)
    cache = sim_cache.for_category(category)
    graphs_dir = out_dir / "graphs"
    viz = Viz(graphs_dir)

    combo_cfg_meta = {
        "kind": "locator_combination",
        "generator": gen_name,
        "noise": noise_name,
        "strategy": strat_name,
        # "repeats": n_repeats,  # Removed to allow elastic repeats
        "seed": main_seed,
        "max_steps": loc_max_steps,
        "timeout_s": loc_timeout_s,
    }
    # Create key configuration by removing runtime constraints that shouldn't invalidate cache
    combo_cfg_key = combo_cfg_meta.copy()
    combo_cfg_key.pop("max_steps", None)
    combo_cfg_key.pop("timeout_s", None)
    combo_key = cache.make_key(combo_cfg_key)

    # Check if cache should be ignored for this specific strategy
    skip_cache_for_strategy = (
        ignore_cache_strategy is not None and strat_name == ignore_cache_strategy
    )

    cached_results_full: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    # Initialize state for all repeats (moved up for elastic subsetting)
    repeat_rngs = []
    initial_scans = []
    repeat_start_times = []
    repeat_stop_reasons = ["" for _ in range(n_repeats)]
    repeats_needed = []

    for attempt_idx_in_combo in range(n_repeats):
        combo_seed_str = f"{main_seed}-{gen_name}-{strat_name}-{noise_name}-{attempt_idx_in_combo}"
        attempt_seed = int(hashlib.sha256(combo_seed_str.encode("utf-8")).hexdigest(), 16) % (10**8)
        repeat_rngs.append(random.Random(attempt_seed))
        initial_scans.append(gen_obj.generate(repeat_rngs[-1]))
        repeat_start_times.append(time.perf_counter())

    cached_steps = 0
    needs_subsetting = False

    if require_cache or (use_cache and not skip_cache_for_strategy):
        # Elastic Steps: Check if cached steps are sufficient
        cached_config = cache.get_config(combo_key)
        cached_steps = cached_config.get("max_steps", 0) if cached_config else 0

        if cached_steps >= loc_max_steps:
            cached_combo_df = cache.load_df(combo_key)
            if cached_steps > loc_max_steps:
                needs_subsetting = True
        else:
            cached_combo_df = None

        if (
            cached_combo_df is not None
            and "results" in cached_combo_df.columns
            and not cached_combo_df.is_empty()
        ):
            cached_payload_raw = cached_combo_df.get_column("results")[0]
            if isinstance(cached_payload_raw, str):
                try:
                    cached_payload = json.loads(cached_payload_raw)
                    for record in cached_payload:
                        if not isinstance(record, dict):
                            break
                        entries = record.get("entries")
                        result_row = record.get("main_result_row")
                        if not isinstance(entries, list) or not isinstance(result_row, dict):
                            break
                        cached_results_full.append((entries, result_row))
                except Exception:
                    log.warning(
                        "Cached combination payload for %s/%s/%s is corrupted.",
                        gen_name,
                        noise_name,
                        strat_name,
                    )

    if require_cache:
        if len(cached_results_full) >= n_repeats:
            if task.progress_queue:
                task.progress_queue.put((task.task_id, n_repeats))
            log.debug(
                "Cache hit for %s/%s/%s (seed=%s); skipping simulation.",
                gen_name,
                noise_name,
                strat_name,
                main_seed,
            )
            return cached_results_full[:n_repeats]

        # If we are here, cache is missing or insufficient, but require_cache is True
        log.warning(
            "Cache missing/insufficient (%d/%d) for %s/%s/%s (seed=%s) and --require-cache is set. Skipping.",
            len(cached_results_full),
            n_repeats,
            gen_name,
            noise_name,
            strat_name,
            main_seed,
        )
        if task.progress_queue:
            task.progress_queue.put((task.task_id, n_repeats))
        return []

    if use_cache and not skip_cache_for_strategy:
        if len(cached_results_full) >= n_repeats:
            if task.progress_queue:
                task.progress_queue.put((task.task_id, n_repeats))
            log.debug(
                "Cache hit for %s/%s/%s (seed=%s); skipping simulation.",
                gen_name,
                noise_name,
                strat_name,
                main_seed,
            )
            return cached_results_full[:n_repeats]
        elif cached_results_full:
            log.info(
                "Partial cache hit for %s/%s/%s: found %d/%d repeats.",
                gen_name,
                noise_name,
                strat_name,
                len(cached_results_full),
                n_repeats,
            )
            if task.progress_queue:
                task.progress_queue.put((task.task_id, len(cached_results_full)))

    all_results_for_combination = [None] * n_repeats

    # Fill from elastic cache first
    n_cached = len(cached_results_full)
    for i in range(min(n_repeats, n_cached)):
        all_results_for_combination[i] = cached_results_full[i]

    # Batched history and repeat tracking
    history_rows = []

    # Process loaded results (subsetting) and check partial cache
    for attempt_idx_in_combo in range(n_repeats):
        if attempt_idx_in_combo < n_cached:
            # Already loaded from combo cache
            if needs_subsetting:
                res = all_results_for_combination[attempt_idx_in_combo]
                if res:
                    sub = _recalculate_subset_result(
                        res[1], res[0], initial_scans[attempt_idx_in_combo], loc_max_steps
                    )
                    if sub:
                        all_results_for_combination[attempt_idx_in_combo] = sub
                        continue
                    # If subsetting fails (no history), fall through to re-run?
                    # For now we assume history is present if greater steps.
            continue

        # Check partial cache if enabled (for repeats NOT in combo cache)
        if use_cache and not skip_cache_for_strategy:
            # Logic similar to elastic combo
            part_cfg_meta = {
                "kind": "locator_run",
                "generator": gen_name,
                "noise": noise_name,
                "strategy": strat_name,
                "repeat": attempt_idx_in_combo,
                "seed": main_seed,
                "max_steps": loc_max_steps,
                "timeout_s": loc_timeout_s,
            }
            part_cfg_key = part_cfg_meta.copy()
            part_cfg_key.pop("max_steps", None)
            part_cfg_key.pop("timeout_s", None)

            repeat_part_key = cache.make_key(part_cfg_key)

            # Check steps for partial
            pc_config = cache.get_config(repeat_part_key)
            pc_steps = pc_config.get("max_steps", 0) if pc_config else 0

            if pc_steps >= loc_max_steps:
                part_df = cache.load_df(repeat_part_key)
                if part_df is not None and not part_df.is_empty():
                    try:
                        entries_raw = part_df["plot_manifest"][0]
                        result_row_raw = part_df["result_row"][0]
                        entries = json.loads(entries_raw)
                        result_row = json.loads(result_row_raw)

                        if isinstance(entries, list) and isinstance(result_row, dict):
                            # Subsetting for partial
                            if pc_steps > loc_max_steps:
                                sub = _recalculate_subset_result(
                                    result_row,
                                    entries,
                                    initial_scans[attempt_idx_in_combo],
                                    loc_max_steps,
                                )
                                if sub:
                                    entries, result_row = sub
                                else:
                                    repeats_needed.append(attempt_idx_in_combo)
                                    continue

                            all_results_for_combination[attempt_idx_in_combo] = (
                                entries,
                                result_row,
                            )
                            if task.progress_queue:
                                task.progress_queue.put((task.task_id, 1))
                            continue
                    except Exception:
                        pass

        repeats_needed.append(attempt_idx_in_combo)

    # Batched history and repeat tracking
    history_rows = []
    repeats_df = pl.DataFrame(
        {
            "repeat_id": list(range(n_repeats)),
            "active": [rid in repeats_needed for rid in range(n_repeats)],
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
                    pl.when(pl.col("repeat_id") == rid)
                    .then(pl.lit(False))
                    .otherwise(pl.col("active"))
                    .alias("active")
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
        if all_results_for_combination[attempt_idx_in_combo] is not None:
            continue

        current_scan = initial_scans[attempt_idx_in_combo]

        # Extract history for this repeat
        if not final_history_df.is_empty():
            current_history_df = final_history_df.filter(
                pl.col("repeat_id") == attempt_idx_in_combo
            ).drop("repeat_id")
        else:
            current_history_df = pl.DataFrame(
                {
                    "x": pl.Series("x", [], dtype=pl.Float64),
                    "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
                }
            )

        if current_history_df.is_empty():
            log.info(
                "No measurements recorded for repeat %s (reason=%s); "
                "generating baseline scan plot.",
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
        metrics_serialized["bayesian_convergence"] = 0.0

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
                if hasattr(locator_instance, "posterior_history") and hasattr(
                    locator_instance, "freq_grid"
                ):
                    # Compute model history
                    model_history = []
                    if hasattr(locator_instance, "parameter_history"):
                        for params in locator_instance.parameter_history:
                            model_history.append(
                                locator_instance.odmr_model(locator_instance.freq_grid, params)
                            )

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

        main_result_row = {
            "generator": gen_name,
            "noise": noise_name,
            "strategy": strat_name,
            "repeats": n_repeats,
            "attempt": attempt_idx_in_combo + 1,
            "stop_reason": repeat_stop_reasons[attempt_idx_in_combo],
            **metrics_serialized,
        }
        if hasattr(strat_obj, "parameter_history"):
            # Serialize history for elastic steps (subsetting)
            main_result_row["parameter_history"] = _to_native(strat_obj.parameter_history)

        if use_cache and not skip_cache_for_strategy:
            part_cfg_meta = {
                "kind": "locator_run",
                "generator": gen_name,
                "noise": noise_name,
                "strategy": strat_name,
                "repeat": attempt_idx_in_combo,
                "seed": main_seed,
                "max_steps": loc_max_steps,
                "timeout_s": loc_timeout_s,
            }
            part_cfg_key = part_cfg_meta.copy()
            part_cfg_key.pop("max_steps", None)
            part_cfg_key.pop("timeout_s", None)

            repeat_part_key = cache.make_key(part_cfg_key)
            cache_df = pl.DataFrame(
                {
                    "plot_manifest": [json.dumps(entries)],
                    "result_row": [json.dumps(main_result_row)],
                }
            )
            cache.save_df(cache_df, repeat_part_key, metadata={"config": part_cfg_meta})

        if task.progress_queue:
            task.progress_queue.put((task.task_id, 1))
        else:
            log.debug(f"Finished repeat {attempt_idx_in_combo + 1} for {gen_name}/{noise_name}")
        all_results_for_combination[attempt_idx_in_combo] = (entries, main_result_row)

    if (
        use_cache
        and not skip_cache_for_strategy
        and all(r is not None for r in all_results_for_combination)
    ):
        # Only cache combo if ALL results are present (which they should be, either from cache or run)
        # Note: if some failed (kept as None?), we should handle it.
        # But we initialize with None. If logic is correct, all indices are filled.
        # Filter Nones just in case?
        valid_results = [r for r in all_results_for_combination if r is not None]
        combo_payload = [
            {"entries": entries, "main_result_row": main_result_row}
            for entries, main_result_row in valid_results
        ]
        combo_df = pl.DataFrame({"results": [json.dumps(combo_payload)]})
        cache.save_df(combo_df, combo_key, metadata={"config": combo_cfg_meta})

    return [r for r in all_results_for_combination if r is not None]


def _load_duration_estimates(out_dir: Path) -> dict[tuple[str, str, str], float]:
    csv_path = out_dir / "locator_results.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pl.read_csv(csv_path)
        if "duration_ms" not in df.columns:
            return {}

        # Group by config and take mean duration
        stats = df.group_by(["generator", "noise", "strategy"]).agg(pl.col("duration_ms").mean())

        estimates = {}
        for row in stats.to_dicts():
            key = (row["generator"], row["noise"], row["strategy"])
            estimates[key] = row["duration_ms"]
        return estimates
    except Exception:
        return {}


@app.command(name="list")
def list_cache(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
):
    """List all cached simulations."""
    cache_root = out / "cache"
    if not cache_root.exists():
        console = Console()
        console.print("[yellow]No cache directory found.[/yellow]")
        return

    table = Table(title="Cached Simulations")
    table.add_column("Category", style="cyan")
    table.add_column("Generator", style="green")
    table.add_column("Noise", style="magenta")
    table.add_column("Strategy", style="blue")
    table.add_column("Seed", justify="right")
    table.add_column("Repeats", justify="right")

    console = Console()

    # Iterate over all subdirectories in cache root
    found_any = False
    for category_dir in cache_root.iterdir():
        if category_dir.is_dir():
            cat_cache = CategoryCache(category_dir)
            items = cat_cache.list_content()
            for config in items:
                # We only want to list "locator_combination" items
                kind = config.get("kind")
                if kind == "locator_combination":
                    found_any = True
                    table.add_row(
                        category_dir.name,
                        str(config.get("generator", "-")),
                        str(config.get("noise", "-")),
                        str(config.get("strategy", "-")),
                        str(config.get("seed", "-")),
                        str(config.get("repeats", "-")),
                    )

    if found_any:
        console.print(table)
    else:
        console.print("[yellow]No cached combinations found (or no metadata available).[/yellow]")


@app.command()
def run(  # noqa: C901
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    repeats: Annotated[int, typer.Option("--repeats", help="Number of repeats per scenario")] = 5,
    seed: Annotated[int, typer.Option("--seed", help="RNG seed (int)")] = 123,
    loc_max_steps: Annotated[
        int,
        typer.Option("--loc-max-steps", help="Max steps for locator measurement loop"),
    ] = 150,
    loc_timeout_s: Annotated[
        int,
        typer.Option("--loc-timeout", help="Timeout in seconds for a single locator run"),
    ] = 1500,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable caching for this run"),
    ] = False,
    ignore_cache_strategy: Annotated[
        str | None,
        typer.Option(
            "--ignore-cache-strategy",
            help="Ignore cache for a specific strategy (e.g., 'NVCenter-SimpleSequential')",
        ),
    ] = None,
    filter_category: Annotated[
        str | None,
        typer.Option(
            "--filter-category",
            help="Filter by generator category (e.g., 'NVCenter')",
        ),
    ] = None,
    strategy: Annotated[
        str | None,
        typer.Option(
            "--strategy",
            help="Filter by strategy name (e.g., 'NVCenter-SequentialBayesian')",
        ),
    ] = None,
    noise: Annotated[
        str | None,
        typer.Option(
            "--noise",
            help="Filter by noise preset name (e.g., 'NoNoise')",
        ),
    ] = None,
    parallel: Annotated[
        bool,
        typer.Option("--parallel/--no-parallel", help="Run simulations in parallel"),
    ] = True,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
            case_sensitive=False,
        ),
    ] = "INFO",
    no_progress: Annotated[
        bool,
        typer.Option("--no-progress", help="Disable progress bars"),
    ] = False,
    require_cache: Annotated[
        bool,
        typer.Option("--require-cache", help="Skip simulation if cache is missing"),
    ] = False,
) -> int:
    """Typer-driven command-line interface entry point."""
    console = Console()
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
    ]
    logging.basicConfig(
        level=log_level_value,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    logging.getLogger("nvision").setLevel(log_level_value)

    with multiprocessing.Manager() as manager:
        log_queue = manager.Queue(-1)
        listener = QueueListener(log_queue, *handlers)
        listener.start()

        out_dir: Path = out
        ensure_out_dir(out_dir)

        cache_dir = out_dir / "cache"
        if no_cache and cache_dir.exists():
            log.debug("Clearing cache.")
            shutil.rmtree(cache_dir)
        ensure_out_dir(cache_dir)

        graphs_dir = out_dir / "graphs"
        ensure_out_dir(graphs_dir)
        scans_dir = graphs_dir / "scans"
        ensure_out_dir(scans_dir)
        bayes_dir = graphs_dir / "bayes"
        ensure_out_dir(bayes_dir)

        log.debug("Starting simulations...")

        generator_map = dict(sim_cases.generators_basic())
        noise_map = dict(_noise_presets())

        tasks: list[LocatorTask] = []
        seen_configs: set[tuple[str, str, str]] = set()
        used_slugs: set[str] = set()

        progress_queue = manager.Queue()

        main_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )

        sub_progress = Progress(
            TextColumn("{task.description}"),
            DotsColumn(),
        )

        progress_group = Group(main_progress, sub_progress)

        with Live(progress_group, refresh_per_second=10):
            duration_estimates = _load_duration_estimates(out_dir)
            tid_to_weight = {}

            main_task_id = main_progress.add_task("[cyan]Total Progress", total=0)
            total_weighted_repeats = 0.0

            for gen_name, gen_obj in generator_map.items():
                if filter_category and _get_generator_category(gen_name) != filter_category:
                    continue

                for strat_name, strat_obj in _locator_strategies_for_generator(gen_name):
                    if strategy and strat_name != strategy:
                        continue
                    for noise_name, noise_obj in noise_map.items():
                        if noise and noise_name != noise:
                            continue
                        config_key = (gen_name, noise_name, strat_name)
                        if config_key in seen_configs:
                            continue
                        seen_configs.add(config_key)

                        slug_base = "_".join(
                            slugify(part) for part in (gen_name, noise_name, strat_name)
                        )
                        slug_candidate = slug_base
                        suffix = 1
                        while slug_candidate in used_slugs:
                            suffix += 1
                            slug_candidate = f"{slug_base}-{suffix}"
                        used_slugs.add(slug_candidate)

                        desc = f"[cyan]{gen_name}/{noise_name}/{strat_name}[/cyan]"
                        task_id = sub_progress.add_task(desc, total=repeats)

                        est_duration = duration_estimates.get(
                            (gen_name, noise_name, strat_name), 1000.0
                        )
                        tid_to_weight[task_id] = est_duration
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

            main_progress.update(main_task_id, total=total_weighted_repeats)

            plot_manifest: list[dict[str, object]] = []
            df_rows: list[dict] = []

            def monitor_progress():
                completed_weighted = 0.0
                while True:
                    item = progress_queue.get()
                    if item is None:
                        break
                    tid, advance = item
                    sub_progress.update(tid, advance=advance)

                    weight = tid_to_weight.get(tid, 1000.0)
                    completed_weighted += advance * weight
                    main_progress.update(main_task_id, completed=completed_weighted)

                    for task in sub_progress.tasks:
                        if task.id == tid and task.completed >= task.total:
                            sub_progress.remove_task(tid)
                            break

            monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
            monitor_thread.start()

            if parallel:
                with ProcessPoolExecutor() as executor:
                    futures = {
                        executor.submit(_run_combination, locator_task): locator_task
                        for locator_task in tasks
                    }
                    for future in as_completed(futures):
                        results_for_combination = future.result()
                        for entries, main_result_row in results_for_combination:
                            plot_manifest.extend(entries)
                            df_rows.append(main_result_row)
            else:
                for locator_task in tasks:
                    results_for_combination = _run_combination(locator_task)
                    for entries, main_result_row in results_for_combination:
                        plot_manifest.extend(entries)
                        df_rows.append(main_result_row)

            progress_queue.put(None)
            monitor_thread.join()

        listener.stop()

        df_loc = pl.DataFrame(df_rows)
        # Drop complex columns not supported in CSV
        if "parameter_history" in df_loc.columns:
            df_loc = df_loc.drop("parameter_history")

        out_path = out_dir / "locator_results.csv"
        df_loc.write_csv(out_path.as_posix())
        log.info(f"Wrote locator results to: {out_path}")

        viz = Viz(graphs_dir)
        try:
            summary_plots_meta = viz.plot_locator_summary(df_loc)
            for meta in summary_plots_meta:
                meta["path"] = Path(meta["path"]).relative_to(out_dir).as_posix()
            plot_manifest.extend(summary_plots_meta)
            log.info(f"Saved {len(summary_plots_meta)} summary plots")
        except Exception as exc:
            log.warning(f"Plotting failed: {exc}")

        if not plot_manifest:
            log.warning("No plots were generated. Adding a dummy entry to manifest.")
            plot_manifest.append(
                {
                    "type": "scan",
                    "generator": "Dummy-Generator",
                    "noise": "None",
                    "strategy": "Dummy-Strategy",
                    "repeat": 1,
                    "repeat_total": 1,
                    "stop_reason": "no_data",
                    "abs_err_x": None,
                    "uncert": None,
                    "measurements": 0,
                    "duration_ms": 0,
                    "metrics": {},
                    "path": "",  # Empty path, so iframe will be empty
                }
            )

        manifest_path = out_dir / "plots_manifest.json"
        manifest_path.write_text(json.dumps(plot_manifest, indent=2), encoding="utf-8")

        try:
            idx = compile_html_index(out_dir)
            log.info(f"Generated HTML index at: {idx.absolute().as_uri()}")
        except Exception as exc:
            log.warning(f"Failed to build HTML index: {exc}")

        log.info(f"Wrote locator results to: {out_dir}")
    return 0


@app.command()
def gui(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    port: Annotated[int, typer.Option("--port", help="Port to run the server on")] = 8080,
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Do not open the browser automatically"),
    ] = False,
) -> None:
    """Launch the NiceGUI results viewer."""
    from nvision.gui import run_gui

    ensure_out_dir(out)
    run_gui(out, port=port, show=not no_browser)


def _should_delete_file(
    file_path: Path,
    strategy: str | None,
    generator: str | None,
    noise: str | None,
) -> bool:
    """Determine if a file should be deleted based on filters."""
    with file_path.open() as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            return False

    return all(
        [
            not (strategy and config.get("strategy") != strategy),
            not (generator and config.get("generator") != generator),
            not (noise and config.get("noise") != noise),
        ]
    )


def _get_cache_dirs(cache_base_dir: Path, category: str | None) -> list[Path]:
    """Get list of cache directories to process."""
    if category:
        return [cache_base_dir / category]
    categories = [p for p in cache_base_dir.iterdir() if p.is_dir()]
    return categories if categories else [cache_base_dir / "unknown"]


def _find_matching_files(
    cache_base_dir: Path,
    category: str | None,
    strategy: str | None,
    generator: str | None,
    noise: str | None,
) -> tuple[list[Path], list[Path]]:
    """Find cache files matching the given filters."""
    matches = []
    configs = []
    for cat_dir in _get_cache_dirs(cache_base_dir, category):
        if not cat_dir.exists():
            continue
        for entry in cat_dir.glob("*.parquet"):
            cfg_path = entry.with_suffix(".json")
            if not cfg_path.exists():
                continue
            try:
                with cfg_path.open() as f:
                    config = json.load(f)
                if _matches_filter(config, category, strategy, generator, noise):
                    matches.append(entry)
                    configs.append(cfg_path)
            except json.JSONDecodeError:
                continue
    return matches, configs


def _delete_files(files: list[Path], dry_run: bool) -> None:
    """Delete files with dry run support."""
    for path in files:
        if dry_run:
            typer.echo(f"[dry-run] Would delete: {path}")
        else:
            path.unlink(missing_ok=True)
            typer.echo(f"Deleted: {path}")


@app.command()
def cache_clean(
    out: Annotated[Path, typer.Option("--out", help="Output directory", dir_okay=True)] = Path(
        "artifacts"
    ),
    category: Annotated[
        str | None,
        typer.Option("--category", help="Generator category (OnePeak, TwoPeak, NVCenter)"),
    ] = None,
    strategy: Annotated[
        str | None, typer.Option("--strategy", help="Locator strategy filter")
    ] = None,
    generator: Annotated[
        str | None, typer.Option("--generator", help="Generator name filter")
    ] = None,
    noise: Annotated[str | None, typer.Option("--noise", help="Noise preset filter")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show matches without deleting")
    ] = False,
) -> None:
    """Delete cached simulation artifacts matching optional filters."""

    cache_base_dir = out / "cache"
    if not cache_base_dir.exists():
        typer.echo(f"No cache directory found at {cache_base_dir}")
        raise typer.Exit(code=0)

    # Find matching files
    matches, configs = _find_matching_files(cache_base_dir, category, strategy, generator, noise)

    if not matches:
        typer.echo("No cache parts matched the provided filters.")
        raise typer.Exit(code=0)

    # Show what will be deleted
    typer.echo("Matched cache files:")
    for path in matches + configs:
        typer.echo(f"  {path}")

    if dry_run:
        typer.echo("\nRun without --dry-run to delete these files.")
        raise typer.Exit(code=0)

    # Confirm before deletion
    if not typer.confirm(f"\nDelete {len(matches)} cache files and {len(configs)} configs?"):
        raise typer.Abort()

    # Perform deletion
    _delete_files(matches + configs, dry_run)


def _matches_filter(
    config: dict[str, Any],
    category: str | None,
    strategy: str | None,
    generator: str | None,
    noise: str | None,
) -> bool:
    """Check if a config matches all the given filters."""
    return all(
        [
            strategy is None or config.get("strategy") == strategy,
            generator is None or config.get("generator") == generator,
            noise is None or config.get("noise") == noise,
        ]
    )


@app.command()
def evaluate_bayesian(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts/eval"),
    repeats: Annotated[int, typer.Option("--repeats", help="Number of repeats per scenario")] = 5,
    max_steps: Annotated[int, typer.Option("--max-steps", help="Max steps per run")] = 50,
) -> None:
    """
    Evaluate the Bayesian locator against a set of standard scenarios.
    """
    from nvision.evaluation.bayesian_eval import EvalScenario, print_results, run_evaluation

    scenarios = [
        EvalScenario(
            "Standard NV",
            {"frequency": 2.87e9, "linewidth": 12e6, "amplitude": 0.05, "background": 1.0},
        ),
        EvalScenario(
            "Weak Signal",
            {"frequency": 2.87e9, "linewidth": 12e6, "amplitude": 0.01, "background": 1.0},
        ),
        EvalScenario(
            "Off-Resonance",
            {"frequency": 3.00e9, "linewidth": 12e6, "amplitude": 0.05, "background": 1.0},
        ),
    ]

    log.info(
        f"Starting Bayesian evaluation with {len(scenarios)} scenarios, {repeats} repeats each."
    )
    results = run_evaluation(scenarios, repeats=repeats, max_steps=max_steps, output_dir=out)
    print_results(results)
    log.info(f"Evaluation complete. Results saved to {out}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    NVision simulation runner.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


def cli(*args, **kwargs):
    """Backward-compatible entry point invoking the Typer app."""
    return app(*args, **kwargs)


__all__ = ["cli"]
