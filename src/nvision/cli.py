from __future__ import annotations

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
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Annotated, Any

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
    SimpleSequentialLocator,
    TwoPeakGoldenLocator,
    TwoPeakGridLocator,
    TwoPeakSweepLocator,
)
from nvision.sim import cases as sim_cases
from nvision.viz import Viz

log = logging.getLogger("nvision")

app = typer.Typer(help="NVision simulation runner")


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

    def __str__(self) -> str:
        return self.slug


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
            (
                "NVCenter-SimpleSequential",
                SimpleSequentialLocator(max_evals=60, grid_resolution=400),
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


def _category_cache_dir(base: Path, category: str) -> Path:
    slug = slugify(category or "unknown")
    return base / slug


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
    # bayes_dir is currently unused
    loc_max_steps = task.loc_max_steps
    loc_timeout_s = task.loc_timeout_s
    use_cache = task.use_cache
    cache_dir = task.cache_dir
    ignore_cache_strategy = task.ignore_cache_strategy

    log.info(
        "Starting combination: %s/%s/%s for %s repeats",
        gen_name,
        noise_name,
        strat_name,
        n_repeats,
    )

    category = _get_generator_category(gen_name)
    cache_category_dir = _category_cache_dir(cache_dir, category)
    cache = DataFrameCache(cache_category_dir)
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

    # Check if cache should be ignored for this specific strategy
    skip_cache_for_strategy = (
        ignore_cache_strategy is not None and strat_name == ignore_cache_strategy
    )

    if use_cache and not skip_cache_for_strategy:
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

            history_rows.append(
                {
                    "repeat_id": rid,
                    "step": step_num,
                    "x": x_next,
                    "signal_values": y_measured,
                }
            )

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

        # is_bayesian is currently unused
        # is_bayesian = isinstance(strat_obj, NVCenterSequentialBayesianLocator)

        # Bayesian plotting is currently not supported in batched mode
        # TODO: Implement per-repeat Bayesian plot extraction from batched locator

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

    if use_cache and not skip_cache_for_strategy and all_results_for_combination:
        combo_payload = [
            {"entries": entries, "main_result_row": main_result_row}
            for entries, main_result_row in all_results_for_combination
        ]
        combo_df = pl.DataFrame({"results": [json.dumps(combo_payload)]})
        cache.save_df(combo_df, combo_key)

    return all_results_for_combination


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

        tasks: list[LocatorTask] = []
        seen_configs: set[tuple[str, str, str]] = set()
        used_slugs: set[str] = set()

        for gen_name, gen_obj in generator_map.items():
            for strat_name, strat_obj in _locator_strategies_for_generator(gen_name):
                for noise_name, noise_obj in noise_map.items():
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
            progress_task_id = progress.add_task(
                "[cyan]Running simulations...", total=len(tasks) * repeats
            )

            if parallel:
                with ProcessPoolExecutor() as executor:
                    futures = {
                        executor.submit(_run_combination, locator_task): locator_task
                        for locator_task in tasks
                    }
                    for future in as_completed(futures):
                        locator_task = futures[future]
                        results_for_combination = future.result()
                        for entries, main_result_row in results_for_combination:
                            plot_manifest.extend(entries)
                            df_rows.append(main_result_row)
                        desc = f"Done: {locator_task.generator_name}/{locator_task.noise_name}"
                        progress.update(progress_task_id, advance=repeats, description=desc)
            else:
                for locator_task in tasks:
                    results_for_combination = _run_combination(locator_task)
                    for entries, main_result_row in results_for_combination:
                        plot_manifest.extend(entries)
                        df_rows.append(main_result_row)
                    desc = f"Done: {locator_task.generator_name}/{locator_task.noise_name}"
                    progress.update(progress_task_id, advance=repeats, description=desc)

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


def cli(*args, **kwargs):
    """Backward-compatible entry point invoking the Typer app."""
    return app(*args, **kwargs)


__all__ = ["cli"]
