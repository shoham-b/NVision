from __future__ import annotations

import sys

import typer

app = typer.Typer(help="NVision simulation runner", pretty_exceptions_show_locals=False)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """NVision CLI root callback."""
    if ctx.resilient_parsing:
        return

    # If no subcommand is provided, run the "main use" default workflow.
    if ctx.invoked_subcommand is None:
        # Avoid accidentally running when user asks for help.
        if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
            return

        from nvision.cli.run import run

        # Keep prior behavior: default NVCenter run, no cache.
        run(filter_category="NVCenter", no_cache=True)


if __name__ == "__main__":
    app()
