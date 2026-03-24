from __future__ import annotations

import json
import logging
import queue
import shutil
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.logging import RichHandler

from nvision.cache import CacheBridge
from nvision.cli.main import app
from nvision.cli.monitor import MonitorLogHandler, ProgressMonitor
from nvision.gui.report import prepare_static_ui_data
from nvision.runner import TaskListBuildConfig, build_task_list, run_task
from nvision.sim import cases as sim_cases
from nvision.tools.paths import (
    ARTIFACTS_ROOT,  # Assuming PROJECT_ROOT is defined in core.paths
    ensure_out_dir,
)
from nvision.tools.utils import NVISION_RNG_SEED
from nvision.viz import Viz

log = logging.getLogger("nvision")
console = Console()


def _rich_handler(console: Console, suppress: list[object]) -> RichHandler:
    return RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        log_time_format="%Y-%m-%d %H:%M:%S",
        tracebacks_show_locals=False,
        tracebacks_suppress=tuple(suppress),
    )


@app.command()
def run(  # noqa: C901
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = ARTIFACTS_ROOT,
    repeats: Annotated[int, typer.Option("--repeats", help="Number of repeats per scenario")] = 1,
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
    filter_strategy: Annotated[
        str | None,
        typer.Option(
            "--filter-strategy",
            help="Filter by locator strategy (e.g., 'Bayesian')",
        ),
    ] = None,
    filter_generator: Annotated[
        str | None,
        typer.Option(
            "--filter-generator",
            help="Restrict to one generator name (e.g. 'NVCenter-one_peak' for Bayesian UI parity)",
        ),
    ] = None,
    all_experiments: Annotated[
        bool,
        typer.Option("--all", help="Run all experiments (disables default filtering)"),
    ] = False,
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

    defaulted_category = False
    defaulted_strategy = False
    defaulted_generator = False
    if not all_experiments:
        default_case = sim_cases.default_run_case()
        if filter_category is None:
            filter_category = default_case.filter_category
            defaulted_category = True
        if filter_strategy is None and filter_category == default_case.filter_category:
            filter_strategy = default_case.filter_strategy
            defaulted_strategy = True
        if (
            filter_generator is None
            and default_case.filter_generator is not None
            and filter_category == default_case.filter_category
            and filter_strategy == default_case.filter_strategy
        ):
            filter_generator = default_case.filter_generator
            defaulted_generator = True

    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
    suppress_list: list[object] = [typer]
    try:
        import numba

        suppress_list.append(numba)
    except ImportError:
        pass

    progress_queue: queue.Queue = queue.Queue()
    log_display_queue: queue.Queue = queue.Queue()
    monitor = ProgressMonitor(
        console,
        progress_queue,
        log_incoming=log_display_queue if not no_progress else None,
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
            )
        ]
    listener = QueueListener(log_queue, *stream_handlers)
    listener.start()
    logging.basicConfig(
        level=log_level_value,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[QueueHandler(log_queue)],
        force=True,
    )
    logging.getLogger("nvision").setLevel(log_level_value)

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

    tasks, _ = build_task_list(
        TaskListBuildConfig(
            repeats=repeats,
            seed=NVISION_RNG_SEED,
            out_dir=out_dir,
            scans_dir=scans_dir,
            bayes_dir=bayes_dir,
            cache_dir=cache_dir,
            log_queue=log_queue,
            progress_queue=progress_queue,
            log_level_value=log_level_value,
            loc_max_steps=loc_max_steps,
            loc_timeout_s=loc_timeout_s,
            no_cache=no_cache,
            ignore_cache_strategy=ignore_cache_strategy,
            require_cache=require_cache,
            filter_category=filter_category,
            filter_strategy=filter_strategy,
            filter_generator=filter_generator,
        ),
        monitor=monitor,
    )

    plot_manifest: list[dict[str, object]] = []
    df_rows: list[dict] = []
    errors: list[Exception] = []

    # One bridge for the whole run avoids opening/closing SQLite per task (200+ tasks on `cases run all`).
    cache_bridge: CacheBridge | None = None if no_cache else CacheBridge(cache_dir)
    try:
        with monitor:
            for locator_task in tasks:
                try:
                    results_for_task = run_task(locator_task, cache_bridge=cache_bridge)
                    for entries, main_result_row in results_for_task:
                        plot_manifest.extend(entries)
                        df_rows.append(main_result_row)
                except Exception:
                    log.exception("Task failed with error")
                    errors.append(RuntimeError("Check logs for details"))
                    if len(errors) > 5:
                        log.error("Too many errors (>5), terminating...")
                        break
    finally:
        if cache_bridge is not None:
            cache_bridge.close()

    if errors:
        console.print("\n[bold red]Errors occurred during execution:[/bold red]")
        for i, err in enumerate(errors, 1):
            console.print(f"{i}. {err}")
        raise typer.Exit(code=1)

    listener.stop()
    logging.basicConfig(
        level=log_level_value,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[_rich_handler(console, suppress_list)],
        force=True,
    )
    logging.getLogger("nvision").setLevel(log_level_value)

    df_loc = pl.DataFrame(df_rows)
    out_path = out_dir / "locator_results.csv"

    # Merge with existing locator_results.csv if it exists
    if out_path.exists() and len(df_loc) > 0:
        try:
            old_df = pl.read_csv(out_path)
            # Combine and keep only the newest runs for matching scenarios
            # Assuming columns contain 'generator', 'noise', 'strategy', 'repeat'
            if set(["generator", "noise", "strategy", "repeat"]).issubset(old_df.columns):
                df_loc = pl.concat([old_df, df_loc]).unique(
                    subset=["generator", "noise", "strategy", "repeat"],
                    keep="last",
                    maintain_order=True,
                )
        except Exception as e:
            log.warning(f"Could not merge with existing CSV (perhaps schema changed!): {e}")

    df_loc.write_csv(out_path.as_posix())
    log.info(f"Wrote locator results to: {out_path}")

    viz = Viz(graphs_dir)
    summary_plots_meta = []
    try:
        summary_plots_meta = viz.plot_locator_summary(df_loc) or []
        for meta in summary_plots_meta:
            meta["path"] = Path(meta["path"]).relative_to(out_dir).as_posix()
        log.info(f"Saved {len(summary_plots_meta)} summary plots")
    except Exception as exc:
        log.warning(f"Plotting failed: {exc}")

    # Merge plot_manifest with existing manifest to prevent losing unrelated cached entries
    manifest_path = out_dir / "plots_manifest.json"
    if manifest_path.exists():
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            # Identify which combinations we just ran to replace them, but keep the rest
            new_combos = set()
            for entry in plot_manifest:
                if entry.get("type") == "scan":
                    new_combos.add((entry.get("generator"), entry.get("noise"), entry.get("strategy")))

            filtered_old = []
            for entry in old_manifest:
                # Remove old summary plots because we exactly regenerated them for the new merged df_loc
                if entry.get("type") == "summary":
                    continue
                # Keep scans that were not part of this newly run execution
                if entry.get("type") == "scan":
                    combo = (entry.get("generator"), entry.get("noise"), entry.get("strategy"))
                    if combo in new_combos:
                        continue
                filtered_old.append(entry)

            plot_manifest = filtered_old + plot_manifest
        except Exception as e:
            log.warning(f"Could not merge with existing plots_manifest.json: {e}")

    plot_manifest.extend(summary_plots_meta)

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
        ui_entrypoint = prepare_static_ui_data(out_dir)
        log.info(f"Prepared static UI data. Open: {ui_entrypoint.absolute().as_uri()}")
    except Exception as exc:
        log.warning(f"Failed to build HTML index: {exc}")

    log.info(f"Wrote locator results to: {out_dir}")
    return 0
