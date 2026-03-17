from __future__ import annotations

import concurrent.futures
import json
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
from nvision.cli.cache_helpers import restore_graphs
from nvision.cli.main import app
from nvision.cli.utils import (
    _get_generator_category,
    _locator_strategies_for_generator,
    _noise_presets,
)
from nvision.core.paths import ensure_out_dir
from nvision.gui.report import prepare_static_ui_data
from nvision.sim import cases as sim_cases
from nvision.viz import Viz

log = logging.getLogger("nvision")
console = Console()


def _collect_cache_results(
    bridge,
    generator_map,
    noise_map,
    filter_category,
    filter_strategy,
    repeats,
    seed,
    loc_max_steps,
    loc_timeout_s,
    out_dir,
    progress,
    task_id,
    seen_configs,
):
    df_rows = []
    plot_manifest = []

    for gen_name, _ in generator_map.items():
        if filter_category and _get_generator_category(gen_name) != filter_category:
            continue

        for strat_name, _ in _locator_strategies_for_generator(gen_name):
            if filter_strategy and filter_strategy not in strat_name:
                continue
            for noise_name, _ in noise_map.items():
                config_key = (gen_name, noise_name, strat_name)
                if config_key in seen_configs:
                    continue
                seen_configs.add(config_key)

                combo_cfg = {
                    "kind": "locator_combination",
                    "generator": gen_name,
                    "noise": noise_name,
                    "strategy": strat_name,
                    "repeats": repeats,
                    "seed": seed,
                    "max_steps": loc_max_steps,
                    "timeout_s": loc_timeout_s,
                }

                progress.update(task_id, description=f"Checking {gen_name}/{noise_name}/{strat_name}")

                category = _get_generator_category(gen_name)
                cache = bridge.get_cache_for_category(category)

                cached_results = cache.get_cached_results(combo_cfg)

                if cached_results:
                    log.debug(f"Cache hit for {gen_name}/{noise_name}/{strat_name}")
                    # Restore graphs
                    restore_graphs(cached_results, out_dir, log)

                    # Collect results
                    for entries, main_result_row in cached_results:
                        plot_manifest.extend(entries)
                        df_rows.append(main_result_row)
                else:
                    log.warning(f"Cache missing for {gen_name}/{noise_name}/{strat_name}")

    return df_rows, plot_manifest


@app.command()
def render(
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
    ensure_out_dir(out_dir)

    cache_dir = out_dir / "cache"
    ensure_out_dir(cache_dir)

    graphs_dir = out_dir / "graphs"
    ensure_out_dir(graphs_dir)
    scans_dir = graphs_dir / "scans"
    ensure_out_dir(scans_dir)
    bayes_dir = graphs_dir / "bayes"
    ensure_out_dir(bayes_dir)

    log.info("Starting report generation from cache...")

    generator_map = dict(sim_cases.generators_basic())
    noise_map = dict(_noise_presets())

    seen_configs: set[tuple[str, str, str]] = set()

    bridge = CacheBridge(cache_dir)
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
        generator_map,
        noise_map,
        filter_category,
        filter_strategy,
        repeats,
        seed,
        loc_max_steps,
        loc_timeout_s,
        out_dir,
        progress,
        task_id,
        seen_configs,
    )

    if not df_rows:
        log.warning("No results found in cache. The report will be empty.")

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
        ui_entrypoint = prepare_static_ui_data(out_dir)
        log.info(f"Prepared static UI data. Open: {ui_entrypoint.absolute().as_uri()}")
    except Exception as exc:
        log.warning(f"Failed to build HTML index: {exc}")

    log.info(f"Render complete. Results in: {out_dir}")
    return 0
