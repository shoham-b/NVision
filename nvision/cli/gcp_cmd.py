"""CLI subcommands for GCP verification and upload."""

from __future__ import annotations

import subprocess
from typing import Annotated

import typer
from rich.console import Console

from nvision.cli.app_instance import app
from nvision.tools.gcp import verify_bucket, verify_credentials

console = Console()

gcp_app = typer.Typer(
    help="Verify and manage GCP integration.",
    pretty_exceptions_show_locals=False,
)
app.add_typer(gcp_app, name="gcp")


@gcp_app.command("verify")
def gcp_verify(
    bucket: Annotated[
        str | None,
        typer.Option("--bucket", help="Also verify bucket access"),
    ] = None,
) -> None:
    """Check that GCP credentials and (optionally) the bucket are configured.

    This validates that GOOGLE_APPLICATION_CREDENTIALS is set and that the
    service account can authenticate. If --bucket is provided, it also checks
    bucket existence and permissions.
    """
    try:
        creds_msg = verify_credentials()
        console.print(f"[bold green]{creds_msg}[/bold green]")
    except RuntimeError as exc:
        console.print(f"[bold red]Credential check failed:[/bold red]\n{exc}")
        raise typer.Exit(1) from exc

    if bucket:
        try:
            bucket_msg = verify_bucket(bucket)
            console.print(f"[bold green]{bucket_msg}[/bold green]")
        except RuntimeError as exc:
            console.print(f"[bold red]Bucket check failed:[/bold red]\n{exc}")
            raise typer.Exit(1) from exc

    console.print("[bold green]All GCP checks passed.[/bold green]")


@gcp_app.command("create-bucket")
def gcp_create_bucket(
    bucket_name: Annotated[
        str,
        typer.Option("--bucket-name", help="Name of the GCS bucket to create"),
    ],
    project: Annotated[
        str | None,
        typer.Option("--project", help="GCP project ID (uses gcloud default if omitted)"),
    ] = None,
    location: Annotated[
        str,
        typer.Option("--location", help="GCS bucket location"),
    ] = "US",
) -> None:
    """Create a new GCS bucket using gcloud CLI."""
    cmd = ["gcloud", "storage", "buckets", "create", f"gs://{bucket_name}", "--location", location]
    if project:
        cmd.extend(["--project", project])

    console.print(f"[bold cyan]Creating bucket gs://{bucket_name}...[/bold cyan]")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[bold red]Failed to create bucket:[/bold red]\n{result.stderr}")
        raise typer.Exit(1)
    console.print(f"[bold green]Bucket created:[/bold green] gs://{bucket_name}")
    if result.stdout:
        console.print(result.stdout)
