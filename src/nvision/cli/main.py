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
        from nvision.sim import cases as sim_cases

        default_case = sim_cases.default_run_case()
        run(
            repeats=default_case.repeats,
            loc_max_steps=default_case.loc_max_steps,
            loc_timeout_s=default_case.loc_timeout_s,
            no_cache=default_case.no_cache,
            filter_category=default_case.filter_category,
            filter_strategy=default_case.filter_strategy,
            require_cache=default_case.require_cache,
            log_level=default_case.log_level,
            no_progress=default_case.no_progress,
        )


if __name__ == "__main__":
    app()
