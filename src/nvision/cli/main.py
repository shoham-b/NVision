import typer

app = typer.Typer(help="NVision simulation runner")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """NVision CLI root callback."""
    if ctx.invoked_subcommand is None:
        from nvision.cli.run import run

        run(filter_category="NVCenter", no_cache=True)
