from __future__ import annotations

import sys

import typer

from nvision.cli.app_instance import app


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """NVision CLI root callback."""
    if ctx.resilient_parsing:
        return

    # If no subcommand is provided, show help.
    if ctx.invoked_subcommand is None:
        # Avoid accidents during help requests.
        if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
            return

        print("No command provided. Try 'nvision --help'.")
        raise typer.Exit()


if __name__ == "__main__":
    app()
