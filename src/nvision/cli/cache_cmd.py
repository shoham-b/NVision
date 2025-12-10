from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from nvision.cache import CategoryCache
from nvision.cli.main import app

console = Console()

# Create a Typer app for the cache command group
cache_app = typer.Typer(help="Manage simulation cache.")
app.add_typer(cache_app, name="cache")


@cache_app.command(name="list")
def list_cache(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
) -> None:
    """List all cached simulations."""
    cache_root = out / "cache"
    if not cache_root.exists():
        console.print("[yellow]No cache directory found.[/yellow]")
        return

    table = Table(title="Cached Simulations")
    table.add_column("Category", style="cyan")
    table.add_column("Generator", style="green")
    table.add_column("Noise", style="magenta")
    table.add_column("Strategy", style="blue")
    table.add_column("Seed", justify="right")
    table.add_column("Repeats", justify="right")

    # Iterate over all subdirectories in cache root
    found_any = False
    for category_dir in cache_root.iterdir():
        if category_dir.is_dir():
            cat_cache = CategoryCache(category_dir)
            items = cat_cache.list_content()
            for config in items:
                # We only want to list "locator_combination" items
                kind = config.get("kind")
                if kind == "locator_combination":
                    found_any = True
                    table.add_row(
                        category_dir.name,
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
    # Category is implicit if we are iterating a category dir
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
    if not cache_root.exists():
        console.print("[yellow]No cache directory found.[/yellow]")
        return

    files_to_delete: list[Path] = []

    # Iterate over relevant category directories
    dirs = []
    if category:
        # We need to find dir for category. Since names are slugified, we iterate and match?
        # Or just construct slug.
        from nvision.core.paths import slugify

        cat_slug = slugify(category)
        if (cache_root / cat_slug).exists():
            dirs.append(cache_root / cat_slug)
    else:
        dirs = [d for d in cache_root.iterdir() if d.is_dir()]

    for cat_dir in dirs:
        # Check if we need to filter by category (if user didn't specify category but we are iterating generic)
        # Actually logic is: iterate matching dirs, then inside match configs.

        # We need access to cache keys to delete files.
        # CategoryCache.list_content returns configs, but not keys...
        # We need a way to get key->config mapping or iterate.
        # Original cli implementation logic check:
        # _find_matching_files did it by iterating diskcache or files?
        # Let's verify how it was done or implement reasonably.
        # The diskcache stores data in .diskcache files? Or our files are separate?
        # CategoryCache stores dataframe in `cache_dir / key`.
        # Metadata is in the file or in diskcache.
        # Actually, `CategoryCache` uses `FanoutCache`. The dataframe payload is stored in cache?
        # `save_df` does `cache.set(key, payload)` AND returns `path`.
        # Wait, `CategoryCache.save_df`:
        # `cache.set(key, payload)` -> Payload is small dict with data.
        # Returns `self.cache_dir / key`.
        # So files ARE the keys? FanoutCache manages shards.
        # FanoutCache creates sharded directories.
        # If we just delete files in `cat_dir`, we corrupt the cache if it manages them?
        # `CategoryCache` uses `FanoutCache(self.cache_dir.as_posix())`.
        # Only safely delete via cache API? Or if payload is just `df` rows, it's inside diskcache.

        # Re-reading `cache_old.py`:
        # `save_df`: `cache.set(key, payload)`. `cache_dir / key` is returned but might not exist as a standalone file?
        # FanoutCache uses `.diskcache` or similar structure.
        # If we delete, we should use `cache.delete(key)`.

        # cat_cache = CategoryCache(cat_dir)
        # We need method to iterate keys and configs.
        # `cat_cache.list_content` iterates `for key in cache`.

        # Simplified deletion:
        # We can extend CategoryCache to support iteration with deletion logic,
        # or implement it here using `diskcache` directly if we import it.
        from diskcache import FanoutCache

        with FanoutCache(cat_dir.as_posix()) as cache:
            for key in cache:
                try:
                    payload = cache.get(key)
                    if isinstance(payload, dict) and "config" in payload:
                        cfg = payload["config"]
                        if _matches_filter(cfg, None, strategy, generator, noise):
                            # Mark for deletion
                            files_to_delete.append((cat_dir, key))
                except Exception:
                    pass

    if not files_to_delete:
        console.print("[yellow]No matching cache entries found.[/yellow]")
        return

    console.print(f"Found {len(files_to_delete)} entries to delete.")

    if not dry_run and not force:
        if not Confirm.ask("Are you sure you want to delete these?"):
            return

    if dry_run:
        console.print("[dim]Dry run: no files deleted.[/dim]")
    else:
        deleted_count = 0
        from diskcache import FanoutCache
        # Process deletions grouped by dir to avoid opening/closing cache too much
        # Or just one by one is fine for CLI command

        # Group by dir
        by_dir = {}
        for d, k in files_to_delete:
            by_dir.setdefault(d, []).append(k)

        for d, keys in by_dir.items():
            with FanoutCache(d.as_posix()) as cache:
                for k in keys:
                    cache.delete(k)
                    deleted_count += 1

        console.print(f"[green]Deleted {deleted_count} entries.[/green]")
