"""``nv graphs``: build the graphs still waiting in the graph queue.

``nv run`` saves each repeat's results immediately and queues its plot inputs; graph-worker
processes build a combination's graphs once all its repeats are saved (see
``nvision/runner/graph_queue.py``). This command drains whatever is left after an interrupted
run, a crash, or a run started with the workers stopped early.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from logging.handlers import QueueListener
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler

from nvision.cli import defaults as cli_defaults
from nvision.cli.app_instance import app
from nvision.runner import graph_queue
from nvision.tools.paths import ARTIFACTS_ROOT

console = Console()


@app.command(name="graphs")
def graphs(
    directory: Annotated[
        Path,
        typer.Option("--dir", help="Results directory whose graph queue to drain (default: artifacts)."),
    ] = ARTIFACTS_ROOT,
    workers: Annotated[
        int,
        typer.Option("--workers", min=1, help="Graph-worker processes to use (default: NVISION_GRAPH_WORKERS or 1)."),
    ] = max(1, cli_defaults.GRAPH_WORKERS),
    retry_failed: Annotated[
        bool,
        typer.Option("--retry-failed", help="Also re-queue repeats whose graphs failed earlier."),
    ] = False,
) -> None:
    """Build the graphs still queued in DIRECTORY, then archive the completed combinations."""
    from nvision.cli.run import _graph_worker_entry

    cache_dir = directory / "cache"
    spool = graph_queue.spool_dir(cache_dir)
    if not spool.is_dir():
        console.print("[green]No graph queue here -- nothing to do.[/green]")
        return

    graph_queue.prepare_spool(spool, retry_failed=retry_failed)
    # Every combination with queued repeats is worked on, complete or not: its graphs are useful
    # either way, and archiving is a no-op until the combination has all its repeats.
    for slug in graph_queue.slugs_with_state(spool):
        graph_queue.mark_ready(spool, slug)
    combos, repeats = graph_queue.pending_counts(spool)
    if combos == 0:
        console.print("[green]No graphs queued.[/green]")
        return

    n_workers = min(workers, combos)
    console.print(f"Building graphs for {combos} combination(s) ({repeats} repeat(s)) with {n_workers} worker(s)...")

    ctx = multiprocessing.get_context("spawn")
    log_queue = ctx.Queue()
    handler = RichHandler(console=console, show_path=False, rich_tracebacks=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    listener = QueueListener(log_queue, handler)
    listener.start()
    procs = [
        ctx.Process(
            target=_graph_worker_entry,
            args=(log_queue, logging.INFO, str(spool), str(cache_dir), None),
            daemon=True,
        )
        for _ in range(n_workers)
    ]
    try:
        for proc in procs:
            proc.start()
        graph_queue.mark_producers_done(spool)
        last = 0.0
        while any(p.is_alive() for p in procs):
            now = time.monotonic()
            if now - last >= 10.0:
                left_combos, left_repeats = graph_queue.pending_counts(spool)
                console.print(f"  {left_combos} combination(s), {left_repeats} repeat(s) left...")
                last = now
            time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted; the remaining graphs stay queued for the next `nv graphs`.[/yellow]")
        raise typer.Exit(code=130) from None
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=5.0)
        listener.stop()

    _, left = graph_queue.pending_counts(spool)
    failed = len(list(spool.glob("*.failed")))
    if failed:
        console.print(f"[red]{failed} repeat(s) failed; see the log. Use --retry-failed to try them again.[/red]")
        raise typer.Exit(code=1)
    if left:
        console.print(f"[yellow]{left} repeat(s) still queued.[/yellow]")
        raise typer.Exit(code=1)
    console.print("[green]All graphs built.[/green]")
