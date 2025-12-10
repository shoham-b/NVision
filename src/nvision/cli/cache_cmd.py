from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from nvision.cache import CacheBridge, CategoryCache
from nvision.cli.main import app

console = Console()

# Create a Typer app for the cache command group
cache_app = typer.Typer(help="Manage simulation cache.")
app.add_typer(cache_app, name="cache")


def _get_caches(root: Path) -> list[tuple[str, CategoryCache]]:
    bridge = CacheBridge(root)
    return [("NVCenter", bridge.nv_center), ("Complementary", bridge.complementary)]


@cache_app.command(name="list")
def list_cache(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
) -> None:
    """List all cached simulations."""
    cache_root = out / "cache"

    table = Table(title="Cached Simulations")
    table.add_column("Category", style="cyan")
    table.add_column("Generator", style="green")
    table.add_column("Noise", style="magenta")
    table.add_column("Strategy", style="blue")
    table.add_column("Seed", justify="right")
    table.add_column("Repeats", justify="right")

    found_any = False
    for cat_name, cat_cache in _get_caches(cache_root):
        backend = cat_cache.backend
        for key in backend:
            payload = backend.get(key)
            if isinstance(payload, dict) and "config" in payload:
                config = payload["config"]
                kind = config.get("kind")
                if kind == "locator_combination":
                    found_any = True
                    table.add_row(
                        cat_name,
                        str(config.get("generator", "-")),
                        str(config.get("noise", "-")),
                        str(config.get("strategy", "-")),
                        str(config.get("seed", "-")),
                        str(config.get("repeats", "-")),
                    )

    if found_any:
        console.print(table)
    else:
        console.print("[yellow]No cached combinations found (or no metadata available).[/yellow]")


def _matches_filter(
    config: dict[str, Any],
    category: str | None,
    strategy: str | None,
    generator: str | None,
    noise: str | None,
) -> bool:
    """Check if a config matches all the given filters."""
    if strategy and config.get("strategy") != strategy:
        return False
    if generator and config.get("generator") != generator:
        return False
    if noise and config.get("noise") != noise:
        return False
    return True


@cache_app.command(name="clean")
def cache_clean(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    category: Annotated[
        str | None,
        typer.Option("--category", help="Category filter (e.g. 'NVCenter')"),
    ] = None,
    strategy: Annotated[
        str | None,
        typer.Option("--strategy", help="Strategy filter"),
    ] = None,
    generator: Annotated[
        str | None,
        typer.Option("--generator", help="Generator filter"),
    ] = None,
    noise: Annotated[str | None, typer.Option("--noise", help="Noise preset filter")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show matches without deleting")
    ] = False,
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

    if not dry_run and not force:
        if not Confirm.ask("Are you sure you want to delete these?"):
            return

    if dry_run:
        console.print("[dim]Dry run: no files deleted.[/dim]")
    else:
        deleted_count = 0
        for _, cat_cache, key in keys_to_delete:
            cat_cache.backend.delete(key)
            deleted_count += 1

        console.print(f"[green]Deleted {deleted_count} entries.[/green]")
