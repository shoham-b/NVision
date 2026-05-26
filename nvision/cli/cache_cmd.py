from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from nvision.cache import CacheBridge
from nvision.cache.data_store import CategoryDataStore
from nvision.cli.app_instance import app
from nvision.sim.combinations import CombinationGrid
from nvision.sim.grid_enums import GeneratorName, NoiseName, StrategyFilter
from nvision.tools.utils import NVISION_RNG_SEED

console = Console()

# Create a Typer app for the cache command group
cache_app = typer.Typer(help="Manage simulation cache.", pretty_exceptions_show_locals=False)
app.add_typer(cache_app, name="cache")


def _get_caches(root: Path) -> list[tuple[str, CategoryDataStore]]:
    bridge = CacheBridge(root)
    return [("NVCenter", bridge.nv_center), ("Complementary", bridge.complementary)]


@cache_app.command(name="list")
def list_cache(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
) -> None:
    """List cached simulations grouped for readability."""
    cache_root = out / "cache"

    found_any = False
    grouped: dict[tuple[str, str, str, str, str, str, str], set[str]] = defaultdict(set)
    row_counts: dict[tuple[str, str, str, str, str, str, str], int] = defaultdict(int)
    updated_dates: dict[tuple[str, str, str, str, str, str, str], str] = {}
    for cat_name, cat_cache in _get_caches(cache_root):
        backend = cat_cache.backend
        for key in backend:
            payload = backend.get(key)
            if isinstance(payload, dict) and "config" in payload:
                config = payload["config"]
                kind = config.get("kind")
                if kind in ("locator_combination", "locator_combination_pointer"):
                    found_any = True
                    # Extract achieved repeats for pointer kind
                    if kind == "locator_combination_pointer":
                        repeats_val = "-"
                        if "data" in payload and isinstance(payload["data"], list) and len(payload["data"]) > 0:
                            repeats_val = str(payload["data"][0].get("achieved_repeats", "-"))
                    else:
                        repeats_val = str(config.get("repeats", "-"))

                    group_key = (
                        cat_name,
                        str(config.get("generator", "-")),
                        str(config.get("strategy", "-")),
                        repeats_val,
                        str(config.get("max_steps", "-")),
                        str(config.get("timeout_s", "-")),
                        str(config.get("schema_version", "-")),
                    )
                    grouped[group_key].add(str(config.get("noise", "-")))
                    row_counts[group_key] += 1

                    # Track the latest updated date
                    updated_at_val = payload.get("updated_at", "-")
                    if group_key not in updated_dates or (
                        updated_at_val != "-"
                        and (updated_dates[group_key] == "-" or updated_at_val > updated_dates[group_key])
                    ):
                        updated_dates[group_key] = updated_at_val

    if found_any:
        grouped_by_category: dict[str, list[tuple[str, str, str, str, str, str]]] = defaultdict(list)
        for cat_name, generator, strategy, repeats, max_steps, timeout_s, schema in sorted(grouped):
            grouped_by_category[cat_name].append((generator, strategy, repeats, max_steps, timeout_s, schema))

        for cat_name in sorted(grouped_by_category):
            table = Table(title=f"{cat_name} Cache (Grouped)")
            table.add_column("Generator", style="green")
            table.add_column("Strategy", style="blue")
            table.add_column("Repeats", justify="right")
            table.add_column("Max", justify="right")
            table.add_column("Timeout", justify="right")
            table.add_column("Schema", justify="right")
            table.add_column("Noises", justify="right")
            table.add_column("Rows", justify="right")
            table.add_column("Updated", justify="right")

            for generator, strategy, repeats, max_steps, timeout_s, schema in grouped_by_category[cat_name]:
                group_key = (cat_name, generator, strategy, repeats, max_steps, timeout_s, schema)
                table.add_row(
                    generator,
                    strategy,
                    repeats,
                    max_steps,
                    timeout_s,
                    schema,
                    str(len(grouped[group_key])),
                    str(row_counts[group_key]),
                    updated_dates.get(group_key, "-"),
                )
            console.print(table)
    else:
        console.print("[yellow]No cached combinations found (or no metadata available).[/yellow]")


def _matches_filter(
    config: dict[str, Any],
    category: str | None,
    strategy: StrategyFilter | None,
    generator: GeneratorName | None,
    noise: NoiseName | None,
    max_steps: int | None = None,
    repeats: int | None = None,
) -> bool:
    """Check if a config matches all the given filters."""
    if strategy and config.get("strategy") != strategy:
        return False
    if generator and config.get("generator") != generator:
        return False
    if max_steps is not None and config.get("max_steps") != max_steps:
        return False
    if repeats is not None and config.get("repeats") != repeats:
        return False
    return not (noise and not str(config.get("noise", "")).startswith(noise))


@cache_app.command(name="clean")
def cache_clean(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    category: Annotated[
        str | None,
        typer.Option("--category", help="Category filter (e.g. 'NVCenter')"),
    ] = None,
    strategy: Annotated[
        StrategyFilter | None,
        typer.Option("--strategy", help="Strategy filter (see StrategyFilter)."),
    ] = None,
    generator: Annotated[
        GeneratorName | None,
        typer.Option("--generator", help="Generator filter (see GeneratorName)."),
    ] = None,
    noise: Annotated[
        NoiseName | None,
        typer.Option("--noise", help="Noise preset filter (see NoiseName)."),
    ] = None,
    max_steps: Annotated[
        int | None,
        typer.Option("--max-steps", help="Max steps filter"),
    ] = None,
    repeats: Annotated[
        int | None,
        typer.Option("--repeats", help="Repeats filter"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show matches without deleting")] = False,
    force: Annotated[bool, typer.Option("--force", help="Skip confirmation")] = False,
) -> None:
    """Delete cached simulation artifacts matching optional filters."""
    cache_root = out / "cache"

    keys_to_delete: list[tuple[str, Any, str]] = []  # (CategoryName, CacheInstance, Key)

    for cat_name, cat_cache in _get_caches(cache_root):
        if category and category.lower() not in cat_name.lower():
            continue

        backend = cat_cache.backend
        for key in backend:
            payload = backend.get(key)
            if isinstance(payload, dict) and "config" in payload:
                cfg = payload["config"]
                kind = cfg.get("kind")
                if kind in ("locator_combination", "locator_combination_pointer"):
                    cfg_copy = dict(cfg)
                    if kind == "locator_combination_pointer" and "data" in payload and len(payload["data"]) > 0:
                        cfg_copy["repeats"] = payload["data"][0].get("achieved_repeats")
                    if _matches_filter(cfg_copy, None, strategy, generator, noise, max_steps, repeats):
                        keys_to_delete.append((cat_name, cat_cache, key))

    if not keys_to_delete:
        console.print("[yellow]No matching cache entries found.[/yellow]")
        return

    console.print(f"Found {len(keys_to_delete)} entries to delete.")

    if not dry_run and not force and not Confirm.ask("Are you sure you want to delete these?"):
        return

    if dry_run:
        console.print("[dim]Dry run: no files deleted.[/dim]")
    else:
        deleted_count = 0
        from nvision.cache.locator_repository import LocatorResultsRepository

        for _, cat_cache, key in keys_to_delete:
            payload = cat_cache.backend.get(key)
            if isinstance(payload, dict) and "config" in payload:
                cfg = payload["config"]
                # Resolve repeats if pointer config
                resolved_repeats = cfg.get("repeats")
                if cfg.get("kind") == "locator_combination_pointer" and "data" in payload and len(payload["data"]) > 0:
                    resolved_repeats = payload["data"][0].get("achieved_repeats")
                if resolved_repeats is None:
                    resolved_repeats = 0
                repo = LocatorResultsRepository(cat_cache)
                repo.purge_cached_combination(
                    generator=cfg.get("generator"),
                    noise=cfg.get("noise"),
                    strategy=cfg.get("strategy"),
                    repeats=resolved_repeats,
                    seed=cfg.get("seed", NVISION_RNG_SEED),
                    max_steps=cfg.get("max_steps"),
                    timeout_s=cfg.get("timeout_s"),
                )
            else:
                cat_cache.backend.delete(key)
            deleted_count += 1

        console.print(f"[green]Deleted {deleted_count} entries.[/green]")


@cache_app.command(name="clean-manifest")
def clean_manifest(  # noqa: C901
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show matches without deleting")] = False,
) -> None:
    """Remove old/invalid entries from plots_manifest.json and cache.

    Removes entries for generators that no longer exist (TwoPeak-*) and
    entries with outdated generator_type categorization. Also cleans
    corresponding cache entries.
    """
    import json

    manifest_path = out / "plots_manifest.json"
    if not manifest_path.exists():
        console.print("[yellow]No plots_manifest.json found.[/yellow]")
        return

    with open(manifest_path) as f:
        plots = json.load(f)

    original_count = len(plots)
    valid_generators = {g.value for g in GeneratorName}
    invalid_generators = set()

    # Identify invalid entries and collect invalid generator names
    valid_plots = []
    for p in plots:
        gen = p.get("generator", "")
        if gen not in valid_generators or p.get("generator_type") == "Supplemental":
            invalid_generators.add(gen)
        else:
            valid_plots.append(p)

    removed = original_count - len(valid_plots)

    if removed == 0:
        console.print("[green]No invalid entries found.[/green]")
        return

    # Also clean cache entries for invalid generators
    cache_removed = 0
    if invalid_generators:
        cache_root = out / "cache"
        for _cat_name, cat_cache in _get_caches(cache_root):
            backend = cat_cache.backend
            keys_to_delete = []
            for key in backend:
                payload = backend.get(key)
                if isinstance(payload, dict) and "config" in payload:
                    cfg = payload["config"]
                    if cfg.get("generator") in invalid_generators:
                        keys_to_delete.append(key)

            if dry_run:
                cache_removed += len(keys_to_delete)
            else:
                for key in keys_to_delete:
                    backend.delete(key)
                    cache_removed += 1

    if dry_run:
        console.print(f"[dim]Would remove {removed} manifest entries and {cache_removed} cache entries.[/dim]")
    else:
        with open(manifest_path, "w") as f:
            json.dump(valid_plots, f, indent=2)
        console.print(f"[green]Removed {removed} manifest entries and {cache_removed} cache entries.[/green]")


@cache_app.command(name="recalc")
def recalculate_metrics(  # noqa: C901
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    category: Annotated[str | None, typer.Option("--category", help="Category filter")] = None,
    strategy: Annotated[StrategyFilter | None, typer.Option("--strategy", help="Strategy filter")] = None,
    generator: Annotated[GeneratorName | None, typer.Option("--generator", help="Generator filter")] = None,
    noise: Annotated[NoiseName | None, typer.Option("--noise", help="Noise filter")] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps", help="Max steps filter")] = None,
    repeats: Annotated[int | None, typer.Option("--repeats", help="Repeats filter")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show matches without updating")] = False,
    force: Annotated[bool, typer.Option("--force", help="Update even if metrics already exist")] = False,
) -> None:
    """Recalculate metrics for cached simulation runs."""

    from nvision.models.experiment import CoreExperiment

    cache_root = out / "cache"
    grid = CombinationGrid()
    updated_count = 0
    total_repeats = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_id = progress.add_task("Recalculating metrics...", total=None)

        for cat_name, cat_cache in _get_caches(cache_root):
            if category and category.lower() not in cat_name.lower():
                continue

            backend = cat_cache.backend
            for key in backend:
                payload = backend.get(key)
                if not isinstance(payload, dict) or "config" not in payload:
                    continue

                cfg = payload["config"]
                cfg_copy = dict(cfg)
                if cfg.get("kind") == "locator_combination_pointer" and "data" in payload and len(payload["data"]) > 0:
                    cfg_copy["repeats"] = payload["data"][0].get("achieved_repeats")
                if not _matches_filter(cfg_copy, None, strategy, generator, noise, max_steps, repeats):
                    continue

                gen_name = cfg.get("generator")
                noise_name = cfg.get("noise")
                strat_name = cfg.get("strategy")
                seed = cfg.get("seed", NVISION_RNG_SEED)

                combo = grid.resolve(gen_name, noise_name, strat_name)
                if not combo:
                    continue

                # Reconstruct experiment
                rng = random.Random(seed)
                true_signal = combo.generator.generate(rng)
                x_min, x_max = 2.6e9, 3.1e9  # Matches _TaskRunner
                experiment = CoreExperiment(true_signal=true_signal, noise=combo.noise, x_min=x_min, x_max=x_max)

                # Load results
                if payload.get("__nvision_cache__") != "dataframe":
                    continue
                data = payload.get("data", [])
                if not data:
                    continue

                results_json = data[0].get("results")
                if not results_json:
                    continue

                try:
                    results = json.loads(results_json)
                except Exception:
                    continue

                if not results:
                    continue

                progress.update(task_id, description=f"Processing {gen_name}/{noise_name}/{strat_name}")

                new_results = []
                combo_updated = False

                for _rid, record in enumerate(results):
                    # Results are stored as list of dicts: [{"entries": ..., "main_result_row": ...}, ...]
                    if not isinstance(record, dict):
                        continue

                    entries = record.get("entries", [])
                    main_result_row = record.get("main_result_row", {})

                    # We'll use the main_result_row as the 'estimate' source.
                    # It contains the flattened results from run_result_to_finalize_record.
                    from nvision.runner.metrics import _scan_attempt_metrics, _truth_positions

                    truth_positions = _truth_positions(experiment)
                    new_metrics = _scan_attempt_metrics(truth_positions, main_result_row)

                    if not force and all(main_result_row.get(k) == v for k, v in new_metrics.items()):
                        new_results.append(record)
                        continue

                    # Update main_result_row
                    updated_row = dict(main_result_row)
                    updated_row.update(new_metrics)

                    # Update entries
                    new_entries = []
                    for entry in entries:
                        if "metrics" in entry:
                            # entry["metrics"] might be a dict or a list of dicts?
                            # Usually it's a dict.
                            entry["metrics"].update(new_metrics)
                        # Also update top-level metrics in entry
                        entry.update(new_metrics)
                        new_entries.append(entry)

                    new_results.append({"entries": new_entries, "main_result_row": updated_row})
                    combo_updated = True
                    total_repeats += 1

                if combo_updated and not dry_run:
                    data[0]["results"] = json.dumps(new_results)
                    backend.set(key, payload)
                    updated_count += 1
                elif combo_updated:
                    updated_count += 1

    if dry_run:
        console.print(f"[dim]Dry run: would update {updated_count} combinations ({total_repeats} repeats).[/dim]")
    else:
        console.print(f"[green]Updated {updated_count} combinations ({total_repeats} repeats).[/green]")
