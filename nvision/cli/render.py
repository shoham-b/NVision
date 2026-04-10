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
from nvision.cli.app_instance import app
from nvision.gui.report import prepare_static_ui_data
from nvision.runner.cache import restore_graphs
from nvision.sim import cases as sim_cases
from nvision.sim.combinations import CombinationGrid
from nvision.tools.artifacts import (
    ensure_plot_manifest_non_empty,
    merge_locator_results_with_existing,
    merge_run_plot_manifest_with_existing_on_disk,
    plots_manifest_path,
    prepare_artifact_tree,
    relativize_summary_plot_paths,
    write_locator_results_csv,
    write_plots_manifest,
)
from nvision.runner.cache import strip_heavy_fields
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
                # Strip content field (used for cache storage only) to keep manifest small
                cleaned_entries = [{k: v for k, v in entry.items() if k != "content"} for entry in entries]
                plot_manifest.extend(cleaned_entries)
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


def _discover_cached_combination_configs(
    bridge: CacheBridge,
    *,
    filter_category: str | None,
    filter_strategy: str | None,
    filter_generator: str | None,
    repeats: int,
    loc_max_steps: int,
    loc_timeout_s: int,
    strict_params: bool,
) -> list[tuple[str, dict[str, object]]]:
    """Return cached combination configs filtered by CLI options."""
    stores = [("NVCenter", bridge.nv_center), ("Complementary", bridge.complementary)]
    discovered: list[tuple[str, dict[str, object]]] = []

    for store_category, store in stores:
        for key in store.backend:
            cfg = _read_locator_combination_config(store.backend.get(key))
            if cfg is None:
                continue

            category_for_repo = _config_category(cfg, store_category)
            if not _config_matches_filters(
                cfg,
                filter_category=filter_category,
                filter_strategy=filter_strategy,
                filter_generator=filter_generator,
            ):
                continue
            if strict_params and not _config_matches_run_params(
                cfg,
                repeats=repeats,
                loc_max_steps=loc_max_steps,
                loc_timeout_s=loc_timeout_s,
            ):
                continue

            discovered.append((category_for_repo, cfg))

    return discovered


def _read_locator_combination_config(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    cfg = payload.get("config")
    if not isinstance(cfg, dict) or cfg.get("kind") != "locator_combination":
        return None
    generator = cfg.get("generator")
    noise = cfg.get("noise")
    strategy = cfg.get("strategy")
    if not isinstance(generator, str) or not isinstance(noise, str) or not isinstance(strategy, str):
        return None
    return cfg


def _config_category(cfg: dict[str, object], store_category: str) -> str:
    generator = str(cfg.get("generator"))
    inferred = CombinationGrid.generator_category(generator)
    return inferred if inferred != "Unknown" else store_category


def _config_matches_filters(
    cfg: dict[str, object],
    *,
    filter_category: str | None,
    filter_strategy: str | None,
    filter_generator: str | None,
) -> bool:
    generator = str(cfg.get("generator"))
    strategy = str(cfg.get("strategy"))
    inferred_category = CombinationGrid.generator_category(generator)

    if filter_category and inferred_category != filter_category:
        return False
    if filter_generator and generator != filter_generator:
        return False
    return not (filter_strategy and filter_strategy not in strategy)


def _config_matches_run_params(
    cfg: dict[str, object],
    *,
    repeats: int,
    loc_max_steps: int,
    loc_timeout_s: int,
) -> bool:
    return bool(
        cfg.get("repeats") == repeats
        and cfg.get("max_steps") == loc_max_steps
        and cfg.get("timeout_s") == loc_timeout_s
    )


def _collect_cache_results_from_configs(
    bridge: CacheBridge,
    discovered_configs: list[tuple[str, dict[str, object]]],
    out_dir: Path,
    progress,
    task_id,
) -> tuple[list[dict], list[dict[str, object]], int]:
    """Hydrate results from discovered cache configs."""
    df_rows: list[dict] = []
    plot_manifest: list[dict[str, object]] = []
    hits = 0

    for category, cfg in discovered_configs:
        generator = str(cfg.get("generator", "-"))
        noise = str(cfg.get("noise", "-"))
        strategy = str(cfg.get("strategy", "-"))
        progress.update(
            task_id,
            description=f"Loading {generator}/{noise}/{strategy}",
        )

        cache = bridge.get_cache_for_category(category)
        cached_results = cache.get_cached_combination_by_config(cfg)
        if not cached_results:
            continue
        hits += 1
        restore_graphs(cached_results, out_dir)
        for entries, main_result_row in cached_results:
            # Strip heavy fields (content, plot_data) to keep manifest small
            cleaned_entries = []
            for entry in entries:
                cleaned = strip_heavy_fields(entry)
                cleaned_entries.append(cleaned)
                if entry.get("type") == "scan":
                    if not entry.get("generator") or not entry.get("strategy"):
                        log.warning(
                            "Scan entry missing generator or strategy field: %s",
                            {k: v for k, v in entry.items() if k in ("type", "generator", "strategy", "path", "repeat")},
                        )
            plot_manifest.extend(cleaned_entries)
            df_rows.append(main_result_row)

    # Debug logging for entry types
    if plot_manifest:
        type_counts: dict[str, int] = {}
        for entry in plot_manifest:
            etype = entry.get("type", "unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1
        log.info("Loaded from cache: %s entries of types: %s", len(plot_manifest), type_counts)

    return df_rows, plot_manifest, hits


def _rows_from_existing_manifest(out_dir: Path) -> list[dict[str, object]]:
    """Best-effort recovery of locator result rows from existing scan manifest entries."""
    path = plots_manifest_path(out_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []

    rows: list[dict[str, object]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "scan":
            continue
        generator = entry.get("generator")
        noise = entry.get("noise")
        strategy = entry.get("strategy")
        if not isinstance(generator, str) or not isinstance(noise, str) or not isinstance(strategy, str):
            continue
        row: dict[str, object] = {
            "generator": generator,
            "noise": noise,
            "strategy": strategy,
            "repeat": entry.get("repeat", 1),
        }
        for metric in ("abs_err_x", "uncert", "measurements", "duration_ms", "pair_rmse", "abs_err_x1", "abs_err_x2"):
            value = entry.get(metric)
            if value is not None:
                row[metric] = value
        rows.append(row)
    return rows


def _apply_default_filters(
    filter_category: str | None,
    filter_strategy: str | None,
    all_experiments: bool,
) -> tuple[str | None, str | None]:
    if all_experiments:
        return filter_category, filter_strategy

    default_case = sim_cases.default_run_case()
    default_category = default_case.filter_category.value if default_case.filter_category is not None else None
    default_strategy = default_case.filter_strategy.value if default_case.filter_strategy is not None else None

    if filter_category is None:
        filter_category = default_category
        if filter_category is not None:
            log.info("Defaulting to category %r. Use --all to render everything.", filter_category)

    if filter_strategy is None and filter_category == default_category and default_strategy is not None:
        filter_strategy = default_strategy
        log.info(
            "Defaulting to strategy %r for %s. Use --all or --filter-strategy to change.",
            filter_strategy,
            filter_category,
        )

    return filter_category, filter_strategy


def _configure_render_logging(log_level: str) -> None:
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


def _load_cache_rows_for_render(
    bridge: CacheBridge,
    out_dir: Path,
    grid: CombinationGrid,
    *,
    filter_category: str | None,
    filter_strategy: str | None,
    filter_generator: str | None,
    repeats: int,
    loc_max_steps: int,
    loc_timeout_s: int,
) -> tuple[list[dict], list[dict[str, object]]]:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_id = progress.add_task("Checking cache...", total=None)
        discovered = _discover_cached_combination_configs(
            bridge,
            filter_category=filter_category,
            filter_strategy=filter_strategy,
            filter_generator=filter_generator,
            repeats=repeats,
            loc_max_steps=loc_max_steps,
            loc_timeout_s=loc_timeout_s,
            strict_params=True,
        )

        if discovered:
            log.info("Discovered %s cached combination(s) matching requested parameters.", len(discovered))
            df_rows, plot_manifest, combo_hits = _collect_cache_results_from_configs(
                bridge,
                discovered,
                out_dir,
                progress,
                task_id,
            )
            log.info("Loaded %s combination(s) from cache metadata.", combo_hits)
            return df_rows, plot_manifest

        relaxed = _discover_cached_combination_configs(
            bridge,
            filter_category=filter_category,
            filter_strategy=filter_strategy,
            filter_generator=filter_generator,
            repeats=repeats,
            loc_max_steps=loc_max_steps,
            loc_timeout_s=loc_timeout_s,
            strict_params=False,
        )
        if relaxed:
            log.warning(
                "No exact cache match for repeats=%s max_steps=%s timeout_s=%s. "
                "Using %s discovered cached combination(s) with different run parameters.",
                repeats,
                loc_max_steps,
                loc_timeout_s,
                len(relaxed),
            )
            df_rows, plot_manifest, combo_hits = _collect_cache_results_from_configs(
                bridge,
                relaxed,
                out_dir,
                progress,
                task_id,
            )
            log.info("Loaded %s combination(s) from relaxed cache metadata lookup.", combo_hits)
            return df_rows, plot_manifest

        # Fallback for older cache payloads that may miss metadata.
        return _collect_cache_results(
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
    ] = sim_cases.DEFAULT_LOC_MAX_STEPS,
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
    filter_category, filter_strategy = _apply_default_filters(
        filter_category=filter_category,
        filter_strategy=filter_strategy,
        all_experiments=all_experiments,
    )
    _configure_render_logging(log_level)

    out_dir: Path = out
    tree = prepare_artifact_tree(out_dir)

    log.info("Starting report generation from cache...")

    grid = CombinationGrid()
    bridge = CacheBridge(tree.cache_dir)
    try:
        df_rows, plot_manifest = _load_cache_rows_for_render(
            bridge,
            out_dir,
            grid,
            filter_category=filter_category,
            filter_strategy=filter_strategy,
            filter_generator=filter_generator,
            repeats=repeats,
            loc_max_steps=loc_max_steps,
            loc_timeout_s=loc_timeout_s,
        )
    finally:
        bridge.close()

    df_loc = pl.DataFrame(df_rows)
    df_loc = merge_locator_results_with_existing(df_loc, out_dir, log)
    if df_loc.is_empty():
        recovered_rows = _rows_from_existing_manifest(out_dir)
        if recovered_rows:
            df_loc = pl.DataFrame(recovered_rows)
            log.info(
                "Recovered %s locator result row(s) from existing plots_manifest.json.",
                len(recovered_rows),
            )
    if df_loc.is_empty():
        log.warning("No results found in cache or existing artifacts. The report will be empty.")

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
    # Keep summary rows fresh for the rendered dataset; keep all non-summary entries (scan, bayesian_interactive, etc.)
    plot_manifest = [
        entry
        for entry in plot_manifest
        if entry.get("type") != "summary"
        and not (entry.get("type") == "scan" and entry.get("generator") == "Dummy-Generator" and not entry.get("path"))
    ]
    plot_manifest.extend(summary_plots_meta)
    _postprocess_manifest_entries(plot_manifest, out_dir)

    ensure_plot_manifest_non_empty(plot_manifest, log)
    manifest_path = write_plots_manifest(plot_manifest, out_dir)
    log.info("Wrote manifest with %s entries to: %s", len(plot_manifest), manifest_path)

    try:
        ui_entrypoint = prepare_static_ui_data(out_dir)
        log.info(f"Prepared static UI data. Open: {ui_entrypoint.absolute().as_uri()}")
    except Exception as exc:
        log.warning(f"Failed to build HTML index: {exc}")

    log.info(f"Render complete. Results in: {out_dir}")
    return 0
