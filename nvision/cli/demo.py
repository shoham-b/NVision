"""Quick demo command for validating improvements.

Usage:
    uv run python -m nvision demo              # Run NV center zeeman demo (default)
    uv run python -m nvision demo --no-cache   # Run fresh comparison
    uv run python -m nvision demo --open       # Auto-open results in browser
    uv run python -m nvision demo --out ./my_dir  # Use custom artifact directory

Beta testing (isolated from demo artifacts, uses port 18082):
    uv run python -m nvision beta              # Run beta_artifacts on dedicated port
    uv run python -m nvision beta --no-cache   # Fresh beta run

To run all NV center generators (slower):
    uv run python -m nvision demo --filter-generator """

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from nvision.cli.app_instance import app
from nvision.cli.run import run
from nvision.tools.paths import ensure_out_dir

log = logging.getLogger("nvision")
console = Console()



def _clear_demo_artifacts(artifacts_root: Path, keep_logs: bool = True) -> None:
    """Clear demo artifacts directory to ensure only latest run is shown."""
    if not artifacts_root.exists():
        return

    for item in artifacts_root.iterdir():
        if keep_logs and item.name == "logs":
            continue
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except OSError:
            pass  # Best effort cleanup


@app.command()
def demo(
    repeats: Annotated[
        int,
        typer.Option("--repeats", help="Number of repeats (default: 3 for speed)"),
    ] = 3,
    loc_max_steps: Annotated[
        int,
        typer.Option("--loc-max-steps", help="Max steps per run (default: 50)"),
    ] = 60,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache/--cache", help="Disable cache for fresh run"),
    ] = False,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open results in browser after run"),
    ] = True,
    compare: Annotated[
        bool,
        typer.Option("--compare/--no-compare", help="Compare Bayesian vs SimpleSweep"),
    ] = False,
    runners: Annotated[
        int,
        typer.Option("--runners", min=1, help="Parallel runner processes"),
    ] = 8,
    filter_generator: Annotated[
        str,
        typer.Option("--filter-generator", help="Filter to specific generator (e.g., 'NVCenter-zeeman')"),
    ] = "NVCenter-zeeman",
    filter_noise: Annotated[
        str | None,
        typer.Option("--filter-noise", help="Filter to specific noise (e.g., 'NoNoise', 'Gauss')"),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output directory for demo artifacts (default: demo_artifacts)"),
    ] = None,
) -> int:
    """Quick demo to validate improvements - fast, focused, visual.

    Runs a lightweight NV-center scenario with reduced repeats/steps for quick
    feedback. Ideal for testing code changes before full benchmark runs.
    """
    # Set up artifact directories based on CLI option
    demo_artifacts_root = out if out is not None else Path("demo_artifacts")
    demo_logs_root = demo_artifacts_root / "logs"

    console.print("[bold cyan]NVision Quick Demo[/bold cyan]")
    console.print(f"Repeats: {repeats}, Steps: {loc_max_steps}, Cache: {not no_cache}")
    console.print(f"Artifacts: {demo_artifacts_root}")
    if filter_generator:
        console.print(f"Generator filter: {filter_generator}")
    if filter_noise:
        console.print(f"Noise filter: {filter_noise}")
    console.print()

    # Clear old demo artifacts to ensure only latest run is shown
    _clear_demo_artifacts(demo_artifacts_root, keep_logs=True)

    # Ensure demo artifacts directory exists
    ensure_out_dir(demo_artifacts_root)

    start_time = time.time()

    # Run all Bayesian strategies (SBED + MaximumLikelihood) in single run
    console.print("[bold]Running Bayesian strategies (SBED + MaximumLikelihood) on NV center variants...[/bold]")
    result = run(
        out=demo_artifacts_root,
        repeats=repeats,
        loc_max_steps=loc_max_steps,
        sweep_max_steps=loc_max_steps,  # Use same value for demo
        loc_timeout_s=300,
        no_cache=no_cache,
        ignore_cache_strategy=None,
        filter_category="NVCenter",
        filter_strategy="Bayesian",  # Matches both SBED and MaximumLikelihood
        filter_generator=filter_generator,
        filter_noise=filter_noise,
        filter_target=None,
        all_experiments=False,
        no_progress=False,  # Show unified progress bar
        require_cache=False,
        log_level="INFO",
        runners=runners,
        logs_root=demo_logs_root,
    )
    if result != 0:
        console.print("[bold red]Demo failed![/bold red]")
        return result

    # Optionally run SimpleSweep comparison
    if compare:
        console.print()
        console.print("[bold]Running SimpleSweep baseline for comparison...[/bold]")
        result = run(
            out=demo_artifacts_root,
            repeats=repeats,
            loc_max_steps=loc_max_steps,
            sweep_max_steps=loc_max_steps,
            loc_timeout_s=300,
            no_cache=no_cache,
            ignore_cache_strategy=None,
            filter_category="NVCenter",
            filter_strategy="SimpleSweep",
            filter_generator=filter_generator,
            filter_noise=filter_noise,
            filter_target=None,
            all_experiments=False,
            no_progress=False,  # Show unified progress bar
            require_cache=False,
            log_level="INFO",
            runners=runners,
            logs_root=demo_logs_root,
        )
        if result != 0:
            console.print("[yellow]Warning: Baseline run failed[/yellow]")

    elapsed = time.time() - start_time
    console.print()
    console.print(f"[bold green]Demo complete in {elapsed:.1f}s![/bold green]")

    # Display quick summary
    _display_summary(demo_artifacts_root)

    # Open browser via local HTTP server (runs in background so CLI finishes)
    if open_browser:
        ui_path = demo_artifacts_root / "index.html"
        if ui_path.exists():
            from nvision.cli.serve import serve as _serve_cmd

            console.print()
            _serve_cmd(directory=demo_artifacts_root, port=None, no_open=False, background=True)
        else:
            console.print(f"[yellow]UI not found at {ui_path}[/yellow]")

    return 0


def _display_summary(artifacts_root: Path) -> None:
    """Display a summary table of recent results."""
    results_path = artifacts_root / "locator_results.csv"

    if not results_path.exists():
        console.print("[dim]No results file found yet[/dim]")
        return

    try:
        import polars as pl

        df = pl.read_csv(results_path)
        if df.is_empty():
            return

        # Get most recent runs (last few rows)
        recent = df.tail(6)

        table = Table(title="Recent Results")
        table.add_column("Strategy", style="cyan")
        table.add_column("Generator", style="magenta")
        table.add_column("Steps", justify="right")
        table.add_column("Final Error", justify="right")
        table.add_column("Time (s)", justify="right")

        for row in recent.to_dicts():
            strategy = str(row.get("strategy", "N/A"))[:20]
            generator = str(row.get("generator", "N/A"))[:20]
            steps = str(row.get("measurements", "N/A"))
            error = f"{row.get('abs_err_x', 0):.6f}" if row.get("abs_err_x") is not None else "N/A"
            time_val = f"{row.get('duration_ms', 0) / 1000:.1f}" if row.get("duration_ms") is not None else "N/A"
            table.add_row(strategy, generator, steps, error, time_val)

        console.print(table)

    except Exception as e:
        console.print(f"[dim]Could not display summary: {e}[/dim]")


@app.command()
def beta(
    repeats: Annotated[
        int,
        typer.Option("--repeats", help="Number of repeats (default: 3 for speed)"),
    ] = 3,
    loc_max_steps: Annotated[
        int,
        typer.Option("--loc-max-steps", help="Max steps per run (default: 50)"),
    ] = 60,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache/--cache", help="Disable cache for fresh run"),
    ] = False,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open results in browser after run"),
    ] = True,
    compare: Annotated[
        bool,
        typer.Option("--compare/--no-compare", help="Compare Bayesian vs SimpleSweep"),
    ] = False,
    runners: Annotated[
        int,
        typer.Option("--runners", min=1, help="Parallel runner processes"),
    ] = 8,
    filter_generator: Annotated[
        str,
        typer.Option("--filter-generator", help="Filter to specific generator (e.g., 'NVCenter-zeeman')"),
    ] = "NVCenter-zeeman",
    filter_noise: Annotated[
        str | None,
        typer.Option("--filter-noise", help="Filter to specific noise (e.g., 'NoNoise', 'Gauss')"),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output directory for beta artifacts (default: beta_artifacts)"),
    ] = None,
) -> int:
    """Beta testing command - runs like demo but uses beta_artifacts by default.

    Ideal for testing experimental features or configurations without
    interfering with demo or main artifacts.
    """
    # Delegate to demo implementation with beta_artifacts as default
    return demo(
        repeats=repeats,
        loc_max_steps=loc_max_steps,
        no_cache=no_cache,
        open_browser=open_browser,
        compare=compare,
        runners=runners,
        filter_generator=filter_generator,
        filter_noise=filter_noise,
        out=out if out is not None else Path("beta_artifacts"),
    )


if __name__=="__main__":
    demo()