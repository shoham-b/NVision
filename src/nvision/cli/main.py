import typer

app = typer.Typer(help="NVision simulation runner")

# Import commands to register them
# These imports must happen after `app` is defined to avoid circular deps if they import `app`.
# Ideally `app` is defined here and commands import it.
