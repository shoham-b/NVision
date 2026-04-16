from __future__ import annotations

import typer

# Centralized Typer app instance to avoid circular imports between main.py and subcommands.
app = typer.Typer(help="NVision simulation runner", pretty_exceptions_show_locals=False)
