from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from nvision.cache import CacheBridge
from nvision.cache.data_store import CategoryDataStore
from nvision.cli.app_instance import app
from nvision.sim.grid_enums import GeneratorName, NoiseName, StrategyFilter

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
    for cat_name, cat_cache in _get_caches(cache_root):
        backend = cat_cache.backend
        for key in backend:
            payload = backend.get(key)
            if isinstance(payload, dict) and "config" in payload:
                config = payload["config"]
                kind = config.get("kind")
                if kind == "locator_combination":
                    found_any = True
                    group_key = (
                        cat_name,
                        str(config.get("generator", "-")),
                        str(config.get("strategy", "-")),
                        str(config.get("repeats", "-")),
                        str(config.get("max_steps", "-")),
                        str(config.get("timeout_s", "-")),
                        str(config.get("schema_version", "-")),
                    )
                    grouped[group_key].add(str(config.get("noise", "-")))
                    row_counts[group_key] += 1

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
) -> bool:
    """Check if a config matches all the given filters."""
    if strategy and config.get("strategy") != strategy:
        return False
    if generator and config.get("generator") != generator:
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
                if _matches_filter(cfg, None, strategy, generator, noise):
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
        for _, cat_cache, key in keys_to_delete:
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
