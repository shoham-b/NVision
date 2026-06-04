from __future__ import annotations

import concurrent.futures
import fnmatch
import json
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from nvision.cache import CacheBridge, stable_config_hash
from nvision.cli import defaults as cli_defaults
from nvision.cli.app_instance import app
from nvision.gui.report import prepare_static_ui_data
from nvision.runner.cache import restore_graphs, strip_heavy_fields
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
    if not isinstance(cfg, dict):
        return None
    kind = cfg.get("kind")
    if kind not in ("locator_combination", "locator_combination_pointer"):
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

    # 1. Basic category/generator/strategy filters
    if filter_category and inferred_category != filter_category:
        return False
    if filter_generator and not fnmatch.fnmatch(generator, filter_generator):
        return False
    if filter_strategy:
        patterns = [p.strip() for p in filter_strategy.split(",")]
        if not any(fnmatch.fnmatch(strategy, p) for p in patterns):
            return False

    # 2. Grid validation: ensure the strategy is valid for this generator
    # (e.g. narrow signals should not have sweep strategies)
    grid = CombinationGrid()
    valid_strategies = [name for name, _ in grid.strategies_for(generator)]
    return strategy in valid_strategies


def _config_matches_run_params(
    cfg: dict[str, object],
    *,
    repeats: int,
    loc_max_steps: int,
    loc_timeout_s: int,
) -> bool:
    # Use repeats as a minimum threshold for discovery
    return bool(
        cfg.get("repeats", 0) >= repeats
        and cfg.get("max_steps") == loc_max_steps
        and cfg.get("timeout_s") == loc_timeout_s
    )


def _collect_cache_results_from_configs(  # noqa: C901
    bridge: CacheBridge,
    discovered_configs: list[tuple[str, dict[str, object]]],
    out_dir: Path,
    progress,
    task_id,
) -> tuple[list[dict], list[dict[str, object]], int]:
    """Hydrate results from discovered cache configs."""
    from rich.table import Table

    df_rows: list[dict] = []
    plot_manifest: list[dict[str, object]] = []
    hits = 0

    table = Table(title="Loading from Cache", box=None, header_style="bold cyan")
    table.add_column("Generator")
    table.add_column("Noise")
    table.add_column("Strategy")
    table.add_column("Steps", justify="right")
    table.add_column("Seed", justify="right")
    table.add_column("Repeats", justify="right")

    # Sort for consistent display
    sorted_configs = sorted(
        discovered_configs,
        key=lambda x: (
            str(x[1].get("generator", "")),
            str(x[1].get("noise", "")),
            str(x[1].get("strategy", "")),
            int(x[1].get("max_steps", 0)),
        ),
    )

    for category, cfg in sorted_configs:
        generator = str(cfg.get("generator", "-"))
        noise = str(cfg.get("noise", "-"))
        strategy = str(cfg.get("strategy", "-"))
        max_steps = int(cfg.get("max_steps", 0))
        seed = int(cfg.get("seed", 0))

        progress.update(
            task_id,
            description=f"Loading {generator}/{noise}/{strategy}",
        )

        cache = bridge.get_cache_for_category(category)

        # Determine achieved repeats from pointer if available
        achieved_repeats = int(cfg.get("repeats", 0))
        if cfg.get("kind") == "locator_combination_pointer":
            # Pointer was stored without "repeats" in its hash key — strip it before hashing
            ptr_cfg = {k: v for k, v in cfg.items() if k != "repeats"}
            key = stable_config_hash(ptr_cfg)
            ptr_df = cache._store.load_df(key)
            if ptr_df is not None and not ptr_df.is_empty():
                achieved_repeats = int(ptr_df.get_column("achieved_repeats")[0])

            # Load all achieved repeats
            cached_results = cache._repeats.load_repeats(key, achieved_repeats)
        else:
            cached_results = cache.get_cached_combination_by_config(cfg)

        if not cached_results:
            continue

        hits += 1
        table.add_row(generator, noise, strategy, str(max_steps), str(seed), str(achieved_repeats))

        restore_graphs(cached_results, out_dir)
        for entries, main_result_row in cached_results:
            # Strip heavy fields (content, plot_data) to keep manifest small
            cleaned_entries = []
            for entry in entries:
                cleaned = strip_heavy_fields(entry)
                cleaned_entries.append(cleaned)
                if entry.get("type") == "scan" and (not entry.get("generator") or not entry.get("strategy")):
                    log.warning(
                        "Scan entry missing generator or strategy field: %s",
                        {k: v for k, v in entry.items() if k in ("type", "generator", "strategy", "path", "repeat")},
                    )
            plot_manifest.extend(cleaned_entries)
            df_rows.append(main_result_row)

    if hits:
        console.print(table)

    # Debug logging for entry types
    if plot_manifest:
        type_counts: dict[str, int] = {}
        for entry in plot_manifest:
            etype = entry.get("type", "unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1
        log.info("Loaded from cache: %s entries of types: %s", len(plot_manifest), type_counts)

    return df_rows, plot_manifest, hits


def _rows_from_existing_manifest(out_dir: Path) -> list[dict[str, object]]:  # noqa: C901
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
            "repeats": entry.get("repeat_total"),
            "max_steps": entry.get("max_steps"),
            "seed": entry.get("seed"),
            "attempt": entry.get("repeat"),
            "stop_reason": entry.get("stop_reason"),
            "sweep_steps": entry.get("sweep_steps"),
            "locator_steps": entry.get("locator_steps"),
        }

        # Recover metrics from nested dict
        metrics = entry.get("metrics")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if v is not None:
                    row[k] = v

        # Also fallback/override individual fields
        for metric in (
            "abs_err_x",
            "uncert",
            "measurements",
            "duration_ms",
            "pair_rmse",
            "abs_err_x1",
            "abs_err_x2",
            "sweep_steps",
            "locator_steps",
        ):
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
    if all_experiments or filter_category is not None:
        return filter_category, filter_strategy

    # In render mode, we default to showing everything found in cache
    # unless a specific filter was provided.
    return None, filter_strategy


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
        task_id = progress.add_task("Discovering cache...", total=None)
        discovered = _discover_cached_combination_configs(
            bridge,
            filter_category=filter_category,
            filter_strategy=filter_strategy,
            filter_generator=filter_generator,
            repeats=repeats,
            loc_max_steps=loc_max_steps,
            loc_timeout_s=loc_timeout_s,
            strict_params=False,  # Always allow discovery of larger repeat counts
        )

        if not discovered:
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

        # Deduplicate: if we have multiple cached versions of the same scenario,
        # pick the one with the MOST repeats.
        best_configs: dict[tuple[str, str, str, int, int], tuple[str, dict[str, object]]] = {}
        for category, cfg in discovered:
            gen = str(cfg.get("generator"))
            noise = str(cfg.get("noise"))
            strat = str(cfg.get("strategy"))
            max_steps = int(cfg.get("max_steps", 0))
            seed = int(cfg.get("seed", 0))
            key = (gen, noise, strat, max_steps, seed)

            # For pointers, we need to check the actual achieved repeats from the store
            curr_repeats = int(cfg.get("repeats", 0))
            if cfg.get("kind") == "locator_combination_pointer":
                ptr_key = stable_config_hash(cfg)
                cache = bridge.get_cache_for_category(category)
                ptr_df = cache._store.load_df(ptr_key)
                if ptr_df is not None and not ptr_df.is_empty():
                    curr_repeats = int(ptr_df.get_column("achieved_repeats")[0])

            if key not in best_configs or curr_repeats > int(best_configs[key][1].get("repeats", 0)):
                # Note: we update the config dict to store the resolved 'achieved_repeats' for easier sorting later
                cfg_copy = dict(cfg)
                cfg_copy["repeats"] = curr_repeats
                best_configs[key] = (category, cfg_copy)

        final_configs = list(best_configs.values())
        log.info("Discovered %s unique cached scenario(s) (best versions).", len(final_configs))

        df_rows, plot_manifest, combo_hits = _collect_cache_results_from_configs(
            bridge,
            final_configs,
            out_dir,
            progress,
            task_id,
        )
        log.info("Loaded %s combination(s) from cache.", combo_hits)
        return df_rows, plot_manifest


@app.command()
def render(  # noqa: C901
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory (must match the run that wrote cache)"),
    ] = ARTIFACTS_ROOT,
    repeats: int = typer.Option(
        cli_defaults.DEFAULT_REPEATS,
        "--repeats",
        help="Number of repeats per scenario",
    ),
    loc_max_steps: int = typer.Option(
        cli_defaults.DEFAULT_LOC_MAX_STEPS,
        "--loc-max-steps",
        help="Max steps for Bayesian locator measurement loop",
    ),
    loc_timeout_s: int = typer.Option(
        cli_defaults.DEFAULT_LOC_TIMEOUT_S,
        "--loc-timeout",
        help="Timeout in seconds for a single locator run",
    ),
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
        typer.Option("--all", help="Include all experiments (default for render)"),
    ] = True,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
            case_sensitive=False,
        ),
    ] = "INFO",
    gcp: Annotated[
        bool,
        typer.Option("--gcp", help="Upload rendered results to GCP"),
    ] = cli_defaults.DEFAULT_GCP,
    gcp_bucket: Annotated[
        str | None,
        typer.Option("--gcp-bucket", help="GCP bucket to upload rendered results to"),
    ] = cli_defaults.DEFAULT_GCP_BUCKET,
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

    df_loc = pl.from_dicts(df_rows, infer_schema_length=None) if df_rows else pl.DataFrame()

    # Always attempt recovery of manifest rows to merge with cached results
    recovered_rows = _rows_from_existing_manifest(out_dir)
    if recovered_rows:
        df_recovered = pl.from_dicts(recovered_rows, infer_schema_length=None)
        # Ensure correct column typing
        for col in ("max_steps", "seed", "attempt", "repeats"):
            if col in df_recovered.columns:
                df_recovered = df_recovered.with_columns(pl.col(col).cast(pl.Int64, strict=False))

        if df_loc.is_empty():
            df_loc = df_recovered
            log.info("Recovered %s locator result row(s) from existing plots_manifest.json.", len(recovered_rows))
        else:
            # Diagonally concatenate and unique them to combine both sources
            key = ("generator", "noise", "strategy", "max_steps", "seed", "attempt")
            for col in key:
                if col not in df_loc.columns:
                    df_loc = df_loc.with_columns(pl.lit(None).alias(col))
                if col not in df_recovered.columns:
                    df_recovered = df_recovered.with_columns(pl.lit(None).alias(col))
            for col in ("max_steps", "seed", "attempt"):
                df_loc = df_loc.with_columns(pl.col(col).cast(pl.Int64, strict=False))

            combined = pl.concat([df_recovered, df_loc], how="diagonal").unique(
                subset=list(key),
                keep="last",
                maintain_order=True,
            )
            log.info(
                "Combined %s cached rows with %s recovered rows from plots_manifest.json.",
                len(df_loc),
                len(df_recovered),
            )
            df_loc = combined

    df_loc = merge_locator_results_with_existing(df_loc, out_dir, log)
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

    metric_plots_meta: list[dict[str, object]] = []
    try:
        metric_plots_meta = viz.plot_all_metrics(df_loc) or []
        relativize_summary_plot_paths(metric_plots_meta, out_dir)
        log.info(f"Saved {len(metric_plots_meta)} metric plots")
    except Exception as exc:
        log.warning(f"Metric plotting failed: {exc}")

    # Add metric plots to manifest
    plot_manifest.extend(metric_plots_meta)

    # Duplicate metric plotting blocks removed

    merge_run_plot_manifest_with_existing_on_disk(plot_manifest, out_dir, log)
    # Keep summary rows fresh for the rendered dataset; keep all non-summary entries (scan, bayesian_interactive, etc.)
    plot_manifest = [
        entry
        for entry in plot_manifest
        if entry.get("type") != "summary"
        and not (entry.get("type") == "scan" and entry.get("generator") == "Dummy-Generator" and not entry.get("path"))
    ]
    plot_manifest.extend(summary_plots_meta)

    ensure_plot_manifest_non_empty(plot_manifest, log)
    manifest_path = write_plots_manifest(plot_manifest, out_dir)
    log.info("Wrote manifest with %s entries to: %s", len(plot_manifest), manifest_path)

    try:
        ui_entrypoint = prepare_static_ui_data(out_dir)
        log.info(f"Prepared static UI data. Open: {ui_entrypoint.absolute().as_uri()}")
    except Exception as exc:
        log.warning(f"Failed to build HTML index: {exc}")

    log.info(f"Render complete. Results in: {out_dir}")

    effective_bucket = gcp_bucket or os.getenv("NVISION_GCP_BUCKET")
    if gcp and not effective_bucket:
        console.print("[bold red]Error:[/bold red] --gcp requires --gcp-bucket or NVISION_GCP_BUCKET env var")
        raise typer.Exit(1)

    if gcp and effective_bucket:
        from nvision.tools.gcp import get_public_url, upload_artifacts

        try:
            upload_artifacts(out_dir, effective_bucket)
            public_url = get_public_url(effective_bucket, out_dir.name)
            console.print(f"\n[bold green]Uploaded to GCP:[/bold green] {public_url}")
        except Exception as exc:
            log.warning(f"Failed to upload to GCP: {exc}")

    return 0
