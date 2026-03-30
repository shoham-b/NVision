from __future__ import annotations

import typer

from nvision import install_rich_tracebacks

# Centralized Typer app instance to avoid circular imports between main.py and subcommands.
app = typer.Typer(help="NVision simulation runner", pretty_exceptions_show_locals=False)

# Capture any startup/registration errors with rich tracebacks immediately.
install_rich_tracebacks()
