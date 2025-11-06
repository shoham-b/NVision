from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import multiprocessing
import random
import shutil
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn

from nvision.cache import DataFrameCache
from nvision.index_html import compile_html_index
from nvision.pathutils import ensure_out_dir, slugify
from nvision.sim import (
    CompositeNoise,
    NVCenterSequentialBayesianLocator,
    NVCenterSweepLocator,
    OnePeakGoldenLocator,
    OnePeakGridLocator,
    OnePeakSweepLocator,
    TwoPeakGoldenLocator,
    TwoPeakGridLocator,
    TwoPeakSweepLocator,
)
from nvision.sim import cases as sim_cases
from nvision.viz import Viz

log = logging.getLogger("nvision")


def _noise_presets() -> list[tuple[str, CompositeNoise | None]]:
    """Return the predefined noise combinations for scenarios."""
    return sim_cases.noises_none() + sim_cases.noises_single_each() + sim_cases.noises_complex()


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
                NVCenterSequentialBayesianLocator(max_evals=60, grid_resolution=400),
            ),
        ]
    return strategies


def _maybe_finite(value: object) -> float | None:
    if isinstance(value, int | float):
        value_float = float(value)
        if math.isfinite(value_float):
            return value_float
    return None


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

    for key in ("uncert", "uncert_pos", "uncert_sep"):
        value = estimate.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            metrics[key] = float(value)

    return metrics


def _run_combination(args):  # noqa: C901
    (
        gen_name,
        gen_obj,
        noise_name,
        noise_obj,
        strat_name,
        strat_obj,
        n_repeats,
        main_seed,
        slug_base,
        out_dir,
        scans_dir,
        bayes_dir,
        loc_max_steps,
        loc_timeout_s,
        use_cache,
        cache_dir,
        log_queue,
        log_level,
    ) = args

    if log_queue:
        queue_handler = QueueHandler(log_queue)
        worker_log = logging.getLogger()
        worker_log.addHandler(queue_handler)
        worker_log.setLevel(log_level)

    log.info(f"Starting combination: {gen_name}/{noise_name}/{strat_name} for {n_repeats} repeats")

    cache = DataFrameCache(cache_dir)
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
    combo_key = cache.make_key(combo_cfg)

    if use_cache:
        cached_combo_df = cache.load_df(combo_key)
        if (
            cached_combo_df is not None
            and "results" in cached_combo_df.columns
            and not cached_combo_df.is_empty()
        ):
            cached_payload_raw = cached_combo_df.get_column("results")[0]
            if isinstance(cached_payload_raw, str):
                try:
                    cached_payload = json.loads(cached_payload_raw)
                except json.JSONDecodeError:
                    log.warning(
                        "Cached combination payload for %s/%s/%s is corrupted; recomputing.",
                        gen_name,
                        noise_name,
                        strat_name,
                    )
                else:
                    cached_results: list[tuple[list[dict[str, object]], dict[str, object]]] = []
                    for record in cached_payload:
                        if not isinstance(record, dict):
                            break
                        entries = record.get("entries")
                        result_row = record.get("main_result_row")
                        if not isinstance(entries, list) or not isinstance(result_row, dict):
                            break
                        cached_results.append((entries, result_row))
                    else:
                        if cached_results:
                            log.info(
                                "Cache hit for %s/%s/%s (seed=%s); skipping simulation.",
                                gen_name,
                                noise_name,
                                strat_name,
                                main_seed,
                            )
                            return cached_results
                    log.warning(
                        "Cached combination payload for %s/%s/%s malformed; recomputing.",
                        gen_name,
                        noise_name,
                        strat_name,
                    )

    all_results_for_combination = []

    # Initialize state for all repeats
    repeat_rngs = []
    initial_scans = []
    repeat_locators = []
    repeat_histories = []
    repeat_start_times = []
    finished_repeats = [False] * n_repeats

    for attempt_idx_in_combo in range(n_repeats):
        combo_seed_str = f"{main_seed}-{gen_name}-{strat_name}-{noise_name}-{attempt_idx_in_combo}"
        attempt_seed = int(hashlib.sha256(combo_seed_str.encode("utf-8")).hexdigest(), 16) % (10**8)

        repeat_rngs.append(random.Random(attempt_seed))
        initial_scans.append(gen_obj.generate(repeat_rngs[-1]))

        repeat_locators.append(copy.deepcopy(strat_obj))
        repeat_histories.append([])
        repeat_start_times.append(time.perf_counter())

    # Lockstep simulation loop
    global_start_time = time.perf_counter()
    for _step in range(loc_max_steps):
        if all(finished_repeats):
            break

        if time.perf_counter() - global_start_time > loc_timeout_s:
            log.warning(f"Combination timeout ({loc_timeout_s}s) reached. Finalizing.")
            break

        for attempt_idx_in_combo in range(n_repeats):
            if finished_repeats[attempt_idx_in_combo]:
                continue

            current_locator = repeat_locators[attempt_idx_in_combo]
            current_scan = initial_scans[attempt_idx_in_combo]
            current_history_list = repeat_histories[attempt_idx_in_combo]

            current_history_df = pl.DataFrame(current_history_list)

            if current_locator.should_stop(current_history_df, current_scan):
                finished_repeats[attempt_idx_in_combo] = True
                continue

            if time.perf_counter() - repeat_start_times[attempt_idx_in_combo] > loc_timeout_s:
                log.warning(f"Repeat {attempt_idx_in_combo+1} ({gen_name}/{noise_name}) timed out.")
                finished_repeats[attempt_idx_in_combo] = True
                continue

            x_next = current_locator.propose_next(current_history_df, current_scan)
            y_ideal = current_scan.signal(x_next)

            y_measured = (
                noise_obj.over_probe_noise.apply(
                    y_ideal, repeat_rngs[attempt_idx_in_combo], current_locator
                )
                if noise_obj and noise_obj.over_probe_noise
                else y_ideal
            )

            current_history_list.append({"x": x_next, "signal_values": y_measured})
            repeat_histories[attempt_idx_in_combo] = current_history_list

    # Finalize and collect results for each repeat
    for attempt_idx_in_combo in range(n_repeats):
        current_locator = repeat_locators[attempt_idx_in_combo]
        current_scan = initial_scans[attempt_idx_in_combo]
        current_history_list = repeat_histories[attempt_idx_in_combo]
        current_history_df = pl.DataFrame(current_history_list)

        duration_ms = (time.perf_counter() - repeat_start_times[attempt_idx_in_combo]) * 1000

        if current_history_df.is_empty():
            estimate = {"x_hat": math.nan, "uncert": math.inf}
            measurements = 0
        else:
            estimate = current_locator.finalize(current_history_df, current_scan)
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
                "abs_err_x": metrics_serialized.get("abs_err_x"),
                "uncert": metrics_serialized.get("uncert"),
                "measurements": metrics_serialized.get("measurements"),
                "duration_ms": metrics_serialized.get("duration_ms"),
                "metrics": metrics_serialized,
                "path": out_path.relative_to(out_dir).as_posix(),
            }
        ]

        is_bayesian = isinstance(current_locator, NVCenterSequentialBayesianLocator)

        if is_bayesian and measurements > 0:
            bo_output_path = bayes_dir / f"{attempt_slug}_bo.png"
            bo_result = current_locator.plot_bo(bo_output_path)
            if bo_result is not None:
                entries.append(
                    {
                        "type": "bayesian",
                        "generator": gen_name,
                        "noise": noise_name,
                        "strategy": strat_name,
                        "repeat": attempt_idx_in_combo + 1,
                        "repeat_total": n_repeats,
                        "path": bo_result.relative_to(out_dir).as_posix(),
                        "format": "png",
                    }
                )

            posterior_hist_path = bayes_dir / f"{attempt_slug}_posterior_history.png"
            posterior_result = current_locator.plot_posterior_history(posterior_hist_path)
            if posterior_result is not None:
                entries.append(
                    {
                        "type": "bayesian_stats",
                        "kind": "posterior_history",
                        "generator": gen_name,
                        "noise": noise_name,
                        "strategy": strat_name,
                        "repeat": attempt_idx_in_combo + 1,
                        "repeat_total": n_repeats,
                        "path": posterior_result.relative_to(out_dir).as_posix(),
                        "format": "png",
                    }
                )

            convergence_stats_path = bayes_dir / f"{attempt_slug}_convergence_stats.png"
            convergence_result = current_locator.plot_convergence_stats(convergence_stats_path)
            if convergence_result is not None:
                entries.append(
                    {
                        "type": "bayesian_stats",
                        "kind": "convergence_stats",
                        "generator": gen_name,
                        "noise": noise_name,
                        "strategy": strat_name,
                        "repeat": attempt_idx_in_combo + 1,
                        "repeat_total": n_repeats,
                        "path": convergence_result.relative_to(out_dir).as_posix(),
                        "format": "png",
                    }
                )

        main_result_row = {
            "generator": gen_name,
            "noise": noise_name,
            "strategy": strat_name,
            "repeats": n_repeats,
            "attempt": attempt_idx_in_combo + 1,
            **metrics_serialized,
        }

        if use_cache:
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
            repeat_part_key = cache.make_key(part_cfg)
            cache_df = pl.DataFrame(
                {
                    "plot_manifest": [json.dumps(entries)],
                    "result_row": [json.dumps(main_result_row)],
                }
            )
            cache.save_df(cache_df, repeat_part_key)

        log.info(f"Finished repeat {attempt_idx_in_combo + 1} for {gen_name}/{noise_name}")
        all_results_for_combination.append((entries, main_result_row))

    if use_cache and all_results_for_combination:
        combo_payload = [
            {"entries": entries, "main_result_row": main_result_row}
            for entries, main_result_row in all_results_for_combination
        ]
        combo_df = pl.DataFrame({"results": [json.dumps(combo_payload)]})
        cache.save_df(combo_df, combo_key)

    return all_results_for_combination


def cli(  # noqa: C901
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
    ] = 300,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable caching for this run"),
    ] = False,
    parallel: Annotated[
        bool,
        typer.Option("--parallel", help="Run simulations in parallel"),
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
            log.info("Clearing cache.")
            shutil.rmtree(cache_dir)
        ensure_out_dir(cache_dir)

        graphs_dir = out_dir / "graphs"
        ensure_out_dir(graphs_dir)
        scans_dir = graphs_dir / "scans"
        ensure_out_dir(scans_dir)
        bayes_dir = graphs_dir / "bayes"
        ensure_out_dir(bayes_dir)

        log.info("Starting simulations...")

        generator_map = dict(sim_cases.generators_basic())
        noise_map = dict(_noise_presets())

        all_tasks = []
        for gen_name, gen_obj in generator_map.items():
            for strat_name, strat_obj in _locator_strategies_for_generator(gen_name):
                for noise_name, noise_obj in noise_map.items():
                    slug_base = "_".join(
                        slugify(part) for part in (gen_name, noise_name, strat_name)
                    )
                    all_tasks.append(
                        (
                            gen_name,
                            gen_obj,
                            noise_name,
                            noise_obj,
                            strat_name,
                            strat_obj,
                            repeats,
                            seed,
                            slug_base,
                            out_dir,
                            scans_dir,
                            bayes_dir,
                            loc_max_steps,
                            loc_timeout_s,
                            not no_cache,
                            cache_dir,
                            log_queue,
                            log_level_value,
                        )
                    )

        plot_manifest: list[dict[str, object]] = []
        df_rows: list[dict] = []

        progress_kwargs = {
            "console": console,
            "refresh_per_second": 10,
        }
        if no_progress:
            progress_kwargs["disable"] = True

        with Progress(
            SpinnerColumn(),
            "•",
            "[progress.percentage]{task.percentage:>3.0f}%",
            BarColumn(bar_width=None),
            "•",
            "[progress.completed]{task.completed}/{task.total}",
            **progress_kwargs,
        ) as progress:
            task = progress.add_task("[cyan]Running simulations...", total=len(all_tasks) * repeats)

            if parallel:
                with ProcessPoolExecutor() as executor:
                    futures = {
                        executor.submit(_run_combination, task_args): task_args
                        for task_args in all_tasks
                    }
                    for future in as_completed(futures):
                        results_for_combination = future.result()
                        for entries, main_result_row in results_for_combination:
                            plot_manifest.extend(entries)
                            df_rows.append(main_result_row)
                        desc = f"Done: {main_result_row['generator']}/{main_result_row['noise']}"
                        progress.update(task, advance=repeats, description=desc)
            else:
                for task_args in all_tasks:
                    results_for_combination = _run_combination(task_args)
                    for entries, main_result_row in results_for_combination:
                        plot_manifest.extend(entries)
                        df_rows.append(main_result_row)
                    desc = f"Done: {main_result_row['generator']}/{main_result_row['noise']}"
                    progress.update(task, advance=repeats, description=desc)

        listener.stop()

        df_loc = pl.DataFrame(df_rows)
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

        manifest_path = out_dir / "plots_manifest.json"
        manifest_path.write_text(json.dumps(plot_manifest, indent=2), encoding="utf-8")

        try:
            idx = compile_html_index(out_dir)
            log.info(f"Generated HTML index at: {idx.absolute().as_uri()}")
        except Exception as exc:
            log.warning(f"Failed to build HTML index: {exc}")

        log.info(f"Wrote locator results to: {out_dir}")
    return 0


__all__ = ["cli"]
