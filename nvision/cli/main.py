from __future__ import annotations

import sys

import typer

from nvision.cli.app_instance import app


def _install_tracebacks() -> None:
    """Install rich tracebacks for better error display."""
    from nvision import install_rich_tracebacks

    install_rich_tracebacks()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """NVision CLI root callback."""
    # Install rich tracebacks on first command invocation (not at import time).
    _install_tracebacks()

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
