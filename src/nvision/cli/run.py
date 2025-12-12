from __future__ import annotations

import json
import logging
import multiprocessing
import shutil
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging.handlers import QueueListener
from pathlib import Path
from typing import Annotated

import concurrent.futures
import polars as pl
import typer
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from nvision.cli.main import app
from nvision.cli.runner import _run_combination
from nvision.cli.types import DotsColumn
from nvision.cli.utils import (
    _get_generator_category,
    _load_duration_estimates,
    _locator_strategies_for_generator,
    _noise_presets,
)
from nvision.core.paths import ensure_out_dir, slugify
from nvision.core.types import LocatorTask
from nvision.gui.report import compile_html_index
from nvision.sim import cases as sim_cases
from nvision.viz import Viz

log = logging.getLogger("nvision")
console = Console()


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
    suppress_list = [typer]
    try:
        import numba

        suppress_list.append(numba)
    except ImportError:
        pass

    # We need to import multiprocessing here since we removed it from top level to avoid Shadowing/Redefinition if it was already imported?
    # Actually, the lint said "Redefinition of unused `multiprocessing` from line 5".
    # Line 5 was `import multiprocessing`. I removed the second import on line 14 above.
    import multiprocessing

    suppress_list.extend([multiprocessing, concurrent.futures])

    handlers = [
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            log_time_format="%Y-%m-%d %H:%M:%S",
            tracebacks_show_locals=False,
            tracebacks_suppress=suppress_list,
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

        with Live(progress_group, console=console, refresh_per_second=10):
            duration_estimates = _load_duration_estimates(out_dir)
            tid_to_weight = {}

            main_task_id = main_progress.add_task("[cyan]Total Progress", total=0)
            total_weighted_repeats = 0.0

            for gen_name, gen_obj in generator_map.items():
                if filter_category and _get_generator_category(gen_name) != filter_category:
                    continue

                for strat_name, strat_obj in _locator_strategies_for_generator(gen_name):
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
                        task_id = sub_progress.add_task(desc, total=repeats)

                        est_duration = duration_estimates.get((gen_name, noise_name, strat_name), 1000.0)
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

            errors: list[Exception] = []

            if parallel:
                with ProcessPoolExecutor() as executor:
                    futures = {executor.submit(_run_combination, locator_task): locator_task for locator_task in tasks}
                    for future in as_completed(futures):
                        try:
                            results_for_combination = future.result()
                            for entries, main_result_row in results_for_combination:
                                plot_manifest.extend(entries)
                                df_rows.append(main_result_row)
                        except Exception:
                            # Use rich traceback
                            log.exception("Task failed with error")
                            # We can't easily get the original exception object if we want to suppress frames cleanly via log.exception
                            # But we need to track that an error occurred.
                            # Since we are inside an except block, sys.exc_info() is set.
                            errors.append(RuntimeError("Check logs for details"))
                            if len(errors) > 5:
                                log.error("Too many errors (>5), terminating...")
                                executor.shutdown(wait=False, cancel_futures=True)
                                break
            else:
                for locator_task in tasks:
                    try:
                        results_for_combination = _run_combination(locator_task)
                        for entries, main_result_row in results_for_combination:
                            plot_manifest.extend(entries)
                            df_rows.append(main_result_row)
                    except Exception:
                        log.exception("Task failed with error")
                        errors.append(RuntimeError("Check logs for details"))
                        if len(errors) > 5:
                            log.error("Too many errors (>5), terminating...")
                            break

            if errors:
                console.print("\n[bold red]Errors occurred during execution:[/bold red]")
                for i, err in enumerate(errors, 1):
                    console.print(f"{i}. {err}")
                raise typer.Exit(code=1)

            progress_queue.put(None)
            monitor_thread.join()

        listener.stop()

        df_loc = pl.DataFrame(df_rows)
        out_path = out_dir / "locator_results.csv"
        df_loc.write_csv(out_path.as_posix())
        log.info(f"Wrote locator results to: {out_path}")

        viz = Viz(graphs_dir)
        try:
            summary_plots_meta = viz.plot_locator_summary(df_loc) or []
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
