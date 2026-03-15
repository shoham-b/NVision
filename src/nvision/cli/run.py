from __future__ import annotations

import json
import logging
import queue
import shutil
from logging.handlers import QueueListener
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.logging import RichHandler

from nvision.cli.main import app
from nvision.cli.monitor import ProgressMonitor
from nvision.cli.runner import _run_combination
from nvision.cli.tasks import build_tasks
from nvision.core.paths import ensure_out_dir
from nvision.gui.report import compile_html_index
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
    filter_strategy: Annotated[
        str | None,
        typer.Option(
            "--filter-strategy",
            help="Filter by locator strategy (e.g., 'Bayesian')",
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

    # Default logic: if --all is not specified, default to NVCenter/Bayesian
    if not all_experiments:
        if filter_category is None:
            filter_category = "NVCenter"
            log.info("Defaulting to category 'NVCenter'. Use --all to run everything.")

        # Only default strategy to Bayesian if we are in the NVCenter category (explicitly or by default)
        if filter_strategy is None and filter_category == "NVCenter":
            filter_strategy = "Bayesian"
            log.info("Defaulting to strategy 'Bayesian' for NVCenter. Use --all or --filter-strategy to change.")

    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
    suppress_list = [typer]
    try:
        import numba

        suppress_list.append(numba)
    except ImportError:
        pass

    suppress_list.extend([])

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

    log_queue = queue.Queue(-1)
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

    progress_queue = queue.Queue()
    monitor = ProgressMonitor(console, progress_queue)

    tasks, _ = build_tasks(
        repeats=repeats,
        seed=seed,
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
        monitor=monitor,
    )

    plot_manifest: list[dict[str, object]] = []
    df_rows: list[dict] = []
    errors: list[Exception] = []

    with monitor:
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
