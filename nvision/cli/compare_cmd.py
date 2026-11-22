"""``nv compare`` -- per-locator summary (calibration, accuracy, dip detection) of a finished run."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from nvision.cli.app_instance import app
from nvision.metrics.compare import compare_locators
from nvision.tools.artifacts import read_locator_results
from nvision.tools.paths import ARTIFACTS_ROOT

console = Console()


def _fmt(value: object, column: str) -> str:
    if value is None:
        return "-"
    if column in {"detected", "cover_2sigma"}:
        return f"{100 * float(value):.0f}%"
    if column.startswith(("err_", "unc_")):
        return f"{float(value) / 1e3:.0f} kHz"
    return f"{float(value):.0f}"


@app.command("compare")
def compare(
    out: Annotated[Path, typer.Option("--out", help="Results directory (as used for nv run)")] = ARTIFACTS_ROOT,
    filter_signal: Annotated[str | None, typer.Option("--filter-signal", help="Generator-name substring")] = None,
    filter_noise: Annotated[str | None, typer.Option("--filter-noise", help="Noise-name prefix")] = None,
    by: Annotated[
        str | None,
        typer.Option("--by", help="Also hyperfine_shift by 'generator' and/or 'noise' (comma-separated)"),
    ] = None,
    csv: Annotated[Path | None, typer.Option("--csv", help="Also write the table to this CSV file")] = None,
) -> None:
    """Compare every locator on the same footing: calibration, final accuracy and dip detection.

    The article's claims (search, converge, refine, speedup) are in the UI's Simulation view
    (docs/evaluation_metrics.md).
    """
    path = next((p for p in (out / "locator_results.parquet", out / "locator_results.csv") if p.exists()), None)
    if path is None:
        raise typer.BadParameter(f"No locator_results file in {out}. Run `nv run` first.")
    df = read_locator_results(path)
    if filter_signal:
        df = df.filter(pl.col("generator").str.contains(filter_signal, literal=True))
    if filter_noise:
        df = df.filter(pl.col("noise").str.starts_with(filter_noise))
    group_by = tuple(g.strip() for g in (by or "").split(",") if g.strip())
    unknown = [g for g in group_by if g not in {"generator", "noise"}]
    if unknown:
        raise typer.BadParameter(f"--by accepts 'generator' and/or 'noise', got {unknown}")
    table_df = compare_locators(df, group_by=group_by)
    if table_df.is_empty():
        console.print("[yellow]No usable runs to compare.[/yellow]")
        return

    labels = {"strategy", *group_by}
    sections = {
        "Accuracy (final primary-parameter error vs reported uncertainty)": [
            "runs",
            "measurements",
            "err_median",
            "err_p90",
            "unc_median",
            "cover_2sigma",
        ],
        "Search (share of runs whose dip test fired, median step)": ["detected", "detect_step"],
    }
    for title, columns in sections.items():
        table = Table(title=title)
        for col in [*group_by, "strategy", *columns]:
            table.add_column(col, justify="left" if col in labels else "right")
        for row in table_df.iter_rows(named=True):
            table.add_row(*(str(row[c]) if c in labels else _fmt(row[c], c) for c in [*group_by, "strategy", *columns]))
        console.print(table)
    if csv is not None:
        table_df.write_csv(csv)
        console.print(f"Wrote {csv}")
