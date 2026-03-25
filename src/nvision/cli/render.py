from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from nvision.cache import CacheBridge
from nvision.cli.main import app
from nvision.gui.report import prepare_static_ui_data
from nvision.runner.cache import restore_graphs
from nvision.sim.combinations import CombinationGrid
from nvision.tools.artifacts import (
    ensure_plot_manifest_non_empty,
    prepare_artifact_tree,
    relativize_summary_plot_paths,
    write_locator_results_csv,
    write_plots_manifest,
)
from nvision.tools.paths import ARTIFACTS_ROOT
from nvision.tools.utils import NVISION_RNG_SEED
from nvision.viz import Viz
from nvision.viz.measurements import backfill_scan_plot_data_if_missing

log = logging.getLogger("nvision")
console = Console()


def _postprocess_manifest_entries(plot_manifest: list[dict[str, object]], out_dir: Path) -> None:
    """Backfill ``plot_data`` on scan rows from existing HTML when cache predates it."""
    for entry in plot_manifest:
        backfill_scan_plot_data_if_missing(entry, out_dir)


def _collect_cache_results(
    bridge,
    grid: CombinationGrid,
    filter_category,
    filter_strategy,
    filter_generator,
    repeats,
    seed,
    loc_max_steps,
    loc_timeout_s,
    out_dir,
    progress,
    task_id,
):
    df_rows = []
    plot_manifest = []
    combo_hits = 0
    combo_misses = 0

    for combo in grid.iter(filter_category, filter_strategy, filter_generator):
        progress.update(
            task_id,
            description=f"Checking {combo.generator_name}/{combo.noise_name}/{combo.strategy_name}",
        )

        category = CombinationGrid.generator_category(combo.generator_name)
        cache = bridge.get_cache_for_category(category)
        cached_results = cache.get_cached_combination(
            generator=combo.generator_name,
            noise=combo.noise_name,
            strategy=combo.strategy_name,
            repeats=repeats,
            seed=seed,
            max_steps=loc_max_steps,
            timeout_s=loc_timeout_s,
        )

        if cached_results:
            combo_hits += 1
            log.debug(
                "Cache hit for %s/%s/%s",
                combo.generator_name,
                combo.noise_name,
                combo.strategy_name,
            )
            restore_graphs(cached_results, out_dir)
            for entries, main_result_row in cached_results:
                plot_manifest.extend(entries)
                df_rows.append(main_result_row)
        else:
            combo_misses += 1
            log.debug(
                "Cache miss for %s/%s/%s",
                combo.generator_name,
                combo.noise_name,
                combo.strategy_name,
            )

    if combo_misses:
        log.info(
            "Cache: %s combination(s) loaded; %s not in cache. "
            "Render walks the full grid — misses are normal until every combo has been simulated "
            "(same --out, --repeats, --loc-max-steps, --loc-timeout as `nvision run`; no --no-cache). "
            "Use DEBUG log level to list each missing combo.",
            combo_hits,
            combo_misses,
        )
    elif combo_hits:
        log.info("Cache: all %s requested combination(s) loaded.", combo_hits)

    return df_rows, plot_manifest


@app.command()
def render(
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory (must match the run that wrote cache)"),
    ] = ARTIFACTS_ROOT,
    repeats: Annotated[int, typer.Option("--repeats", help="Number of repeats per scenario")] = 5,
    loc_max_steps: Annotated[
        int,
        typer.Option("--loc-max-steps", help="Max steps for locator measurement loop"),
    ] = 150,
    loc_timeout_s: Annotated[
        int,
        typer.Option("--loc-timeout", help="Timeout in seconds for a single locator run"),
    ] = 1500,
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
            help="Restrict to one generator name (optional).",
        ),
    ] = None,
    all_experiments: Annotated[
        bool,
        typer.Option("--all", help="Include all experiments (disables default filtering)"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
            case_sensitive=False,
        ),
    ] = "INFO",
) -> int:
    """Render reports and graphs from cache without running simulations."""

    # Default logic: if --all is not specified, default to NVCenter/Bayesian
    if not all_experiments:
        if filter_category is None:
            filter_category = "NVCenter"
            log.info("Defaulting to category 'NVCenter'. Use --all to render everything.")

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

    suppress_list.extend([multiprocessing, concurrent.futures])

    logging.basicConfig(
        level=log_level_value,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_time=True,
                log_time_format="%Y-%m-%d %H:%M:%S",
                tracebacks_show_locals=False,
                tracebacks_suppress=suppress_list,
            )
        ],
    )
    logging.getLogger("nvision").setLevel(log_level_value)

    out_dir: Path = out
    tree = prepare_artifact_tree(out_dir)

    log.info("Starting report generation from cache...")

    grid = CombinationGrid()
    bridge = CacheBridge(tree.cache_dir)
    plot_manifest: list[dict[str, object]] = []
    df_rows: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_id = progress.add_task("Checking cache...", total=None)

    df_rows, plot_manifest = _collect_cache_results(
        bridge,
        grid,
        filter_category,
        filter_strategy,
        filter_generator,
        repeats,
        NVISION_RNG_SEED,
        loc_max_steps,
        loc_timeout_s,
        out_dir,
        progress,
        task_id,
    )

    if not df_rows:
        log.warning("No results found in cache. The report will be empty.")

    df_loc = pl.DataFrame(df_rows)
    out_path = write_locator_results_csv(df_loc, out_dir)
    log.info(f"Wrote locator results to: {out_path}")

    viz = Viz(tree.graphs_dir)
    try:
        summary_plots_meta = viz.plot_locator_summary(df_loc) or []
        relativize_summary_plot_paths(summary_plots_meta, out_dir)
        plot_manifest.extend(summary_plots_meta)
        log.info(f"Saved {len(summary_plots_meta)} summary plots")
    except Exception as exc:
        log.warning(f"Plotting failed: {exc}")

    _postprocess_manifest_entries(plot_manifest, out_dir)

    ensure_plot_manifest_non_empty(plot_manifest, log)
    write_plots_manifest(plot_manifest, out_dir)

    try:
        ui_entrypoint = prepare_static_ui_data(out_dir)
        log.info(f"Prepared static UI data. Open: {ui_entrypoint.absolute().as_uri()}")
    except Exception as exc:
        log.warning(f"Failed to build HTML index: {exc}")

    log.info(f"Render complete. Results in: {out_dir}")
    return 0
