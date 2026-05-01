from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import queue
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import polars as pl
import typer
from rich.console import Console
from rich.logging import RichHandler

from nvision.cache import CacheBridge
from nvision.cli import defaults as cli_defaults
from nvision.cli.monitor import MonitorErrorHandler, MonitorLogHandler, ProgressMonitor
from nvision.gui.report import prepare_static_ui_data
from nvision.runner import TaskListBuildConfig, build_task_list, run_task
from nvision.sim import run_groups as sim_run_groups
from nvision.sim.grid_enums import GeneratorName, NoiseName
from nvision.tools.artifacts import (
    ensure_plot_manifest_non_empty,
    merge_locator_results_with_existing,
    merge_run_plot_manifest_with_existing_on_disk,
    prepare_artifact_tree,
    relativize_summary_plot_paths,
    write_locator_results_csv,
    write_plots_manifest,
    write_run_status,
)
from nvision.tools.log_context import CombinationLogFilter
from nvision.tools.paths import ARTIFACTS_ROOT, LOGS_ROOT, ensure_out_dir
from nvision.tools.utils import NVISION_RNG_SEED
from nvision.viz import Viz

log = logging.getLogger("nvision")
console = Console()


def _run_tasks_process_pool(  # noqa: C901
    tasks: list[object],
    *,
    runners: int,
    cache_bridge: CacheBridge | None,
    progress_queue: queue.Queue,
    log: logging.Logger,
    run_log_path: Path,
    monitor: ProgressMonitor | None = None,
    out_dir: Path | None = None,
    started_at: str | None = None,
) -> tuple[list[dict[str, object]], list[dict], list[Exception]]:
    """Run tasks in a process pool and aggregate results in the parent process.

    Workers should have `log_queue=None` and `progress_queue=None` so they don't attempt
    to share Rich/UI queues.
    """
    plot_manifest: list[dict[str, object]] = []
    df_rows: list[dict] = []
    errors: list[Exception] = []
    future_to_task: dict[concurrent.futures.Future, object] = {}
    completed_count = 0
    total_count = len(tasks)

    def _update_status(status: str) -> None:
        if out_dir is not None:
            with contextlib.suppress(Exception):
                write_run_status(
                    out_dir,
                    status,
                    total_tasks=total_count,
                    completed_tasks=completed_count,
                    started_at=started_at,
                )

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=runners)
    shutdown_called = False
    try:
        future_to_task = {executor.submit(run_task, task, cache_bridge=cache_bridge): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            # Check if user requested exit via 'q' key
            if monitor is not None and monitor.exit_requested:
                raise KeyboardInterrupt("User requested exit")
            locator_task = future_to_task[future]
            try:
                results_for_task = future.result()
                # Mark this task complete in the Rich UI once per task.
                if getattr(locator_task, "task_id", None) is not None:
                    progress_queue.put((locator_task.task_id, locator_task.repeats))
                for entries, main_result_row in results_for_task:
                    plot_manifest.extend(entries)
                    df_rows.append(main_result_row)
            except concurrent.futures.CancelledError:
                # Expected during shutdown - don't log as error
                pass
            except KeyboardInterrupt:
                raise  # Let outer handler deal with Ctrl+C
            except Exception as exc:
                # Still advance progress to keep the UI consistent.
                if getattr(locator_task, "task_id", None) is not None:
                    progress_queue.put((locator_task.task_id, locator_task.repeats))
                # Log clean error to console (via monitor), full traceback to file only
                log.error("Task failed with error (combination=%s): %s", locator_task.slug, type(exc).__name__)
                log.debug("Task failed with error (combination=%s)", locator_task.slug, exc_info=True)
                errors.append(RuntimeError(f"Check logs for details: {run_log_path.resolve().as_uri()}"))
                if len(errors) > 5:
                    log.error("Too many errors (>5), terminating...")
                    for pending_future in future_to_task:
                        pending_future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    shutdown_called = True
                    break
            finally:
                completed_count += 1
                _update_status("running")
    except KeyboardInterrupt:
        log.warning("Cancelling pending tasks due to interruption...")
        # Cancel all pending futures first
        for pending_future in future_to_task:
            pending_future.cancel()
        # Shutdown without waiting - terminate processes immediately
        if not shutdown_called:
            executor.shutdown(wait=False, cancel_futures=True)
            shutdown_called = True
        # On Windows, forcibly kill child processes if needed
        try:
            import psutil

            parent = psutil.Process()
            for child in parent.children(recursive=True):
                with contextlib.suppress(psutil.NoSuchProcess):
                    child.terminate()
            # Give processes a moment to terminate gracefully
            import time

            time.sleep(0.5)
            # Kill any remaining
            for child in parent.children(recursive=True):
                with contextlib.suppress(psutil.NoSuchProcess):
                    child.kill()
        except ImportError:
            pass  # psutil not available
        raise  # Re-raise to let the caller handle it
    finally:
        # If we didn't terminate early, wait for workers to finish (with timeout to avoid stall).
        try:
            if not shutdown_called:
                executor.shutdown(wait=True)
        except KeyboardInterrupt:
            # User hit Ctrl+C again during shutdown - force terminate without waiting
            with contextlib.suppress(Exception):
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass  # Best effort cleanup

    return plot_manifest, df_rows, errors


def _prune_run_logs(logs_dir: Path, *, max_runs: int = 2) -> None:
    """Drop legacy single-file logs; keep at most ``max_runs - 1`` prior session files (this run adds one)."""
    for legacy in logs_dir.glob("nvision-run.log*"):
        try:
            legacy.unlink(missing_ok=True)
        except OSError:
            # On Windows another process may temporarily hold the file.
            continue
    session_logs = sorted(
        logs_dir.glob("nvision-run-*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    while len(session_logs) >= max_runs:
        old = session_logs.pop(0)
        try:
            old.unlink(missing_ok=True)
        except OSError:
            # Best-effort cleanup; keep running if a stale log is locked.
            continue


def _rich_handler(console: Console, suppress: list[object]) -> RichHandler:
    return RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        log_time_format="%Y-%m-%d %H:%M:%S",
        tracebacks_show_locals=False,
        tracebacks_suppress=tuple(suppress),
    )


def run(  # noqa: C901
    out: Annotated[Path | None, typer.Option("--out", help="Output directory")] = None,
    repeats: int = cli_defaults.DEFAULT_REPEATS,
    loc_max_steps: Annotated[
        int,
        typer.Option("--loc-max-steps", help="Max steps for Bayesian locator measurement loop"),
    ] = cli_defaults.DEFAULT_LOC_MAX_STEPS,
    sweep_max_steps: Annotated[
        int | None,
        typer.Option(
            "--sweep-max-steps",
            help="Max steps for sweep locator. Omit to auto-compute from signal model.",
        ),
    ] = None,  # Auto-computed from signal model dip properties
    loc_timeout_s: Annotated[
        int,
        typer.Option("--loc-timeout", help="Timeout in seconds for a single locator run"),
    ] = cli_defaults.DEFAULT_LOC_TIMEOUT_S,
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
        typer.Option("--filter-category", help="Filter by generator category (NVCenter, TwoPeak)."),
    ] = None,
    filter_strategy: Annotated[
        str | None,
        typer.Option(
            "--filter-strategy",
            help="Filter by locator strategy (or substring match).",
        ),
    ] = None,
    filter_generator: Annotated[
        GeneratorName | None,
        typer.Option(
            "--filter-generator",
            help="Restrict to one registered generator name (see GeneratorName).",
        ),
    ] = None,
    filter_noise: Annotated[
        NoiseName | None,
        typer.Option(
            "--filter-noise",
            help="Restrict to one registered noise name (see NoiseName).",
        ),
    ] = None,
    filter_signal: Annotated[
        str | None,
        typer.Option(
            "--filter-signal",
            help="Filter by signal type/variant substring in generator name (e.g., 'voigt', 'lorentzian').",
        ),
    ] = None,
    run_group: Annotated[
        str | None,
        typer.Option("--run-group", help="Run group name to use for preset combinations (overrides filter options)."),
    ] = None,
    all_experiments: Annotated[
        bool,
        typer.Option("--all", help="Run all experiments (disables default filtering)"),
    ] = cli_defaults.DEFAULT_RUN_ALL,
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
    runners: Annotated[
        int,
        typer.Option(
            "--runners",
            min=1,
            help="Number of runner processes (use 1 for sequential execution).",
        ),
    ] = 8,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open results in browser after run"),
    ] = False,
    logs_root: Annotated[
        Path | None,
        typer.Option("--logs-root", help="Custom logs directory (default: logs/ under out)"),
    ] = None,
) -> int:
    """Typer-driven command-line interface entry point."""
    console = Console()

    defaulted_category = False
    defaulted_strategy = False
    defaulted_generator = False
    defaulted_noise = False
    combination_names: list[tuple[str, str, str]] | None = None

    if run_group is not None:
        group = sim_run_groups.get_run_group(run_group)
        # Cartesian product of the group's explicit name lists
        combination_names = [
            (g, n, s) for g in group.generator_names for n in group.noise_names for s in group.strategy_names
        ]
    elif not all_experiments:
        # Backward-compatible default: NVCenter category (old default_run_case behaviour)
        if filter_category is None:
            filter_category = "NVCenter"
            defaulted_category = True
        # Old default_run_case did not set a default strategy or generator,
        # so we leave them open (all strategies / all generators in the category).

    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
    suppress_list: list[object] = [typer]
    try:
        import numba

        suppress_list.append(numba)
    except ImportError:
        pass

    progress_queue: queue.Queue = queue.Queue()
    log_display_queue: queue.Queue = queue.Queue()
    error_display_queue: queue.Queue = queue.Queue()
    monitor = ProgressMonitor(
        console,
        progress_queue,
        log_incoming=log_display_queue if not no_progress else None,
        error_incoming=error_display_queue if not no_progress else None,
        live_mode=not no_progress,
    )

    log_queue: queue.Queue = queue.Queue(-1)
    if no_progress:
        stream_handlers: list[logging.Handler] = [_rich_handler(console, suppress_list)]
    else:
        stream_handlers = [
            MonitorLogHandler(
                log_display_queue,
                formatter=logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"),
            ),
            MonitorErrorHandler(
                error_display_queue,
                formatter=logging.Formatter(
                    "%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                ),
            ),
        ]

    if out is None:
        out = ARTIFACTS_ROOT
    ensure_out_dir(out)

    # Use custom logs root if provided, otherwise default to LOGS_ROOT
    effective_logs_root = logs_root if logs_root is not None else LOGS_ROOT
    ensure_out_dir(effective_logs_root)

    _prune_run_logs(effective_logs_root, max_runs=2)
    run_log_path = (
        effective_logs_root
        / f"nvision-run-{datetime.now(tz=ZoneInfo('Asia/Jerusalem')).strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Always capture full tracebacks in file
    _combo_filter = CombinationLogFilter()
    file_handler.addFilter(_combo_filter)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(combo_prefix)s%(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    stream_handlers.append(file_handler)

    listener = QueueListener(log_queue, *stream_handlers)
    listener.start()
    try:
        _queue_handler = QueueHandler(log_queue)
        _queue_handler.addFilter(_combo_filter)
        logging.basicConfig(
            level=logging.DEBUG,  # Allow all messages through; handlers filter by level
            format="%(message)s",
            datefmt="[%X]",
            handlers=[_queue_handler],
            force=True,
        )
        logging.getLogger("nvision").setLevel(logging.DEBUG)  # File handler always gets debug
        log.info("Session log: %s (up to two run logs kept under logs/)", run_log_path.resolve())

        filter_category_str = filter_category if filter_category is not None else None
        filter_strategy_str = filter_strategy if filter_strategy is not None else None
        filter_generator_str = (
            getattr(filter_generator, "value", filter_generator) if filter_generator is not None else None
        )
        filter_noise_str = getattr(filter_noise, "value", filter_noise) if filter_noise is not None else None
        if not all_experiments and filter_noise_str is None:
            filter_noise_str = "Poisson,Gauss,NoNoise"
            defaulted_noise = True
        filter_signal_str = filter_signal if filter_signal is not None else None

        if defaulted_category:
            log.info(
                "Defaulting to category %r. Use --all to run everything.",
                filter_category,
            )
        if defaulted_strategy:
            log.info(
                "Defaulting to strategy %r for %s. Use --all or --filter-strategy to change.",
                filter_strategy,
                filter_category,
            )
        if defaulted_generator and filter_generator:
            log.info(
                "Defaulting to generator %r (matches Bayesian UI scope). Use --all or --filter-generator to change.",
                filter_generator,
            )
        if defaulted_noise:
            log.info(
                "Defaulting to noises %r. Use --all or --filter-noise to change.",
                filter_noise_str,
            )

        out_dir: Path = out
        tree = prepare_artifact_tree(out_dir, clear_cache=False)

        log.debug("Starting simulations...")

        worker_log_queue = None if runners > 1 else log_queue
        worker_progress_queue = None if runners > 1 else progress_queue

        tasks, _ = build_task_list(
            TaskListBuildConfig(
                repeats=repeats,
                seed=NVISION_RNG_SEED,
                out_dir=out_dir,
                scans_dir=tree.scans_dir,
                bayes_dir=tree.bayes_dir,
                cache_dir=tree.cache_dir,
                log_queue=worker_log_queue,
                progress_queue=worker_progress_queue,
                log_level_value=log_level_value,
                loc_max_steps=loc_max_steps,
                sweep_max_steps=sweep_max_steps,
                loc_timeout_s=loc_timeout_s,
                no_cache=no_cache,
                ignore_cache_strategy=ignore_cache_strategy,
                require_cache=require_cache,
                filter_category=filter_category_str,
                filter_strategy=filter_strategy_str,
                filter_generator=filter_generator_str,
                filter_noise=filter_noise_str,
                filter_signal=filter_signal_str,
                combination_names=combination_names,
            ),
            monitor=monitor,
        )

        plot_manifest: list[dict[str, object]] = []
        df_rows: list[dict] = []
        errors: list[Exception] = []
        interrupted = False
        started_at = datetime.now(tz=ZoneInfo("Asia/Jerusalem")).isoformat()
        total_tasks = len(tasks)
        completed_tasks = 0

        def _update_run_status(status: str) -> None:
            with contextlib.suppress(Exception):
                write_run_status(
                    out_dir,
                    status,
                    total_tasks=total_tasks,
                    completed_tasks=completed_tasks,
                    started_at=started_at,
                )

        # Write initial scheduled status so the UI knows a run is queued
        _update_run_status("scheduled")

        # One bridge for the whole run avoids opening/closing SQLite per task (200+ tasks on `cases run all`).
        # For process-based parallelism we intentionally do not construct it in the parent.
        cache_bridge: CacheBridge | None = None
        if not no_cache and runners == 1:
            cache_bridge = CacheBridge(tree.cache_dir)
        try:
            with monitor:
                # Tasks are now actually executing
                _update_run_status("running")
                if runners > 1:
                    log.info("Parallel execution enabled with %s runner(s).", runners)
                    plot_manifest, df_rows, errors = _run_tasks_process_pool(
                        tasks,
                        runners=runners,
                        cache_bridge=cache_bridge,
                        progress_queue=progress_queue,
                        log=log,
                        run_log_path=run_log_path,
                        monitor=monitor,
                        out_dir=out_dir,
                        started_at=started_at,
                    )
                else:
                    for locator_task in tasks:
                        # Check if user requested exit via 'q' key
                        if monitor.exit_requested:
                            raise KeyboardInterrupt("User requested exit")
                        try:
                            results_for_task = run_task(locator_task, cache_bridge=cache_bridge)
                            for entries, main_result_row in results_for_task:
                                plot_manifest.extend(entries)
                                df_rows.append(main_result_row)
                        except Exception as exc:
                            # Log clean error to console (via monitor), full traceback to file only
                            log.error(
                                "Task failed with error (combination=%s): %s",
                                locator_task.slug,
                                type(exc).__name__,
                            )
                            log.debug("Task failed with error (combination=%s)", locator_task.slug, exc_info=True)
                            errors.append(RuntimeError(f"Check logs for details: {run_log_path.resolve().as_uri()}"))
                            if len(errors) > 5:
                                log.error("Too many errors (>5), terminating...")
                                break
                        finally:
                            completed_tasks += 1
                            _update_run_status("running")
        except KeyboardInterrupt:
            interrupted = True
            # Stop monitor immediately to clean up UI
            monitor.stop()
            console.print("\n[yellow]Interrupted by user. Saving partial results and generating UI...[/yellow]")
            log.warning("Run interrupted by user (Ctrl-C). Saving partial results...")
        finally:
            if cache_bridge is not None:
                cache_bridge.close()

        if errors and not interrupted:
            _update_run_status("error")
            console.print("\n[bold red]Errors occurred during execution:[/bold red]")
            for i, err in enumerate(errors, 1):
                console.print(f"{i}. {err}")
            raise typer.Exit(code=1)
    finally:
        # Drain the log queue before stopping the listener to avoid losing messages
        try:
            while not log_queue.empty():
                try:
                    log_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception:
            pass
        listener.stop()

    # Skip UI generation if there are hard errors (not just interruption)
    if errors and not interrupted:
        return 1

    logging.basicConfig(
        level=log_level_value,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[_rich_handler(console, suppress_list), file_handler],
        force=True,
    )
    logging.getLogger("nvision").setLevel(log_level_value)

    # Generate UI even with partial results after interruption
    if not df_rows:
        _update_run_status("partial" if interrupted else "complete")
        if interrupted:
            console.print("[yellow]No results collected before interruption.[/yellow]")
        else:
            console.print("[yellow]No results to display.[/yellow]")
        return 0

    if interrupted:
        _update_run_status("partial")
        console.print(f"[cyan]Processing {len(df_rows)} partial result(s)...[/cyan]")
    else:
        _update_run_status("complete")

    df_loc = pl.DataFrame(df_rows)
    df_loc = merge_locator_results_with_existing(df_loc, out_dir, log)
    out_path = write_locator_results_csv(df_loc, out_dir)
    log.info(f"Wrote locator results to: {out_path}")

    viz = Viz(tree.graphs_dir)
    summary_plots_meta: list[dict[str, object]] = []
    try:
        summary_plots_meta = viz.plot_locator_summary(df_loc) or []
        relativize_summary_plot_paths(summary_plots_meta, out_dir)
        log.info(f"Saved {len(summary_plots_meta)} summary plots")
    except Exception as exc:
        log.warning(f"Plotting failed: {exc}")

    merge_run_plot_manifest_with_existing_on_disk(plot_manifest, out_dir, log)
    plot_manifest.extend(summary_plots_meta)
    ensure_plot_manifest_non_empty(plot_manifest, log)
    write_plots_manifest(plot_manifest, out_dir)

    try:
        ui_entrypoint = prepare_static_ui_data(out_dir)
        log.info(f"Prepared static UI data. Open: {ui_entrypoint.absolute().as_uri()}")
        if interrupted:
            console.print(f"[green]Partial UI generated at: {ui_entrypoint.absolute().as_uri()}[/green]")
    except Exception as exc:
        log.warning(f"Failed to build HTML index: {exc}")

    log.info(f"Wrote locator results to: {out_dir}")
    log.info(f"View results: uv run python -m nvision serve --dir {out_dir}")
    return 0
