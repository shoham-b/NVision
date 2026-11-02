"""``nv lab-add``: register a Google Drive folder of the lab's runs, shown by ``nv serve`` from then on."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from nvision.cli.app_instance import app
from nvision.tools.paths import ARTIFACTS_ROOT


@app.command("lab-add")
def lab_add(
    drive_url: Annotated[str, typer.Argument(help="Google Drive folder link (or id) holding the lab's runs.")],
    directory: Annotated[
        Path,
        typer.Option("--dir", help="Artifacts directory the labs are stored under (the one given to `nv serve`)."),
    ] = ARTIFACTS_ROOT,
) -> None:
    """Download a Drive folder with rclone, process its Bayesian ESR runs, and keep it under ``<dir>/labs``.

    Run once per link (again to pick up new runs). ``nv serve`` lists every lab found there under its Study
    dropdown, so nothing else needs configuring. Needs ``rclone`` with a Drive remote (``rclone config``).
    """
    from nvision.metrics.lab_process import process_lab_tree
    from nvision.metrics.lab_study import labs_dir
    from nvision.tools.drive_fetch import fetch_drive_folder, parse_drive_folder_id

    try:
        dest = labs_dir(directory) / parse_drive_folder_id(drive_url)
        fetch_drive_folder(drive_url, dest)
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    outcomes = process_lab_tree(dest)
    for folder, outcome in outcomes.items():
        typer.echo(f"{folder.relative_to(dest)}: {outcome}")
    if "processed" not in outcomes.values() and not list(dest.rglob("window*MHz_statistics_perRun.csv")):
        typer.echo(f"Error: no lab runs found in {dest}")
        raise typer.Exit(code=1)
    typer.echo(f'Added. View: uv run nv serve --dir "{directory}"')
