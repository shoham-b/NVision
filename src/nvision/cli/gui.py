from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from nvision.cli.main import app
from nvision.gui.app import run_gui


@app.command()
def gui(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    port: Annotated[int, typer.Option("--port", help="Port to run the server on")] = 8080,
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Do not open the browser automatically"),
    ] = False,
) -> None:
    """Launch the NiceGUI results viewer."""
    run_gui(out, port=port, show=not no_browser)
