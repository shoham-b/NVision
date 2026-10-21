from __future__ import annotations

from datetime import UTC, datetime

import typer
from rich.console import Console
from rich.table import Table

from nvision.cache import CacheBridge
from nvision.cli import run_history
from nvision.cli.app_instance import app
from nvision.tools.paths import ARTIFACTS_ROOT, LOGS_ROOT

console = Console()


@app.command("runs")
def list_runs() -> None:
    """List recorded runs (newest first) with their ids, for ``--resume-run <id>``."""
    bridge = CacheBridge(ARTIFACTS_ROOT / "cache")
    store = bridge.nv_center
    try:
        runs = run_history.load_runs(store)
        if run_history.finalize_runs(runs, LOGS_ROOT, min_idle_s=60):
            run_history.save_runs(store, runs)
    finally:
        store.close()
    if not runs:
        console.print("[yellow]No runs recorded yet (runs are recorded from the next `nv run`/`nv groups`).[/yellow]")
        raise typer.Exit()

    table = Table(title="Recorded runs (resume one with --resume-run <id>)")
    for col in (
        "ref",
        "id",
        "started (local)",
        "group",
        "repeats",
        "started combos",
        "resumable combos",
        "resumed from",
    ):
        table.add_column(col)
    for i, run in enumerate(reversed(runs)):
        started_local = (
            datetime.strptime(run["started_at"], "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=UTC)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M")
        )
        table.add_row(
            "latest" if i == 0 else f"~{i}",
            run["id"],
            started_local,
            str(run["group"] or "-"),
            str(run["repeats"]),
            "running?" if run["started"] is None else str(len(run["started"])),
            "-" if run["chain"] is None else str(len(run["chain"])),
            str(run["resumed_from"] or "-"),
        )
    console.print(table)
