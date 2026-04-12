"""Serve artifacts via a local HTTP server.

Usage:
    uv run python -m nvision serve                       # Serve main artifacts (port 8080)
    uv run python -m nvision serve --dir demo_artifacts  # Serve demo artifacts (port 8081)
    uv run python -m nvision serve --port 9000           # Custom port
"""

from __future__ import annotations

import http.server
import logging
import socketserver
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from nvision.cli.app_instance import app
from nvision.tools.paths import ARTIFACTS_ROOT

log = logging.getLogger("nvision")
console = Console()

# Well-known ports for each artifacts directory (high numbers to avoid conflicts)
PORT_MAIN = 18080
PORT_DEMO = 18081


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that suppresses per-request log lines."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _default_port_for_dir(directory: Path) -> int:
    """Return the well-known port for a directory, or PORT_MAIN as fallback."""
    name = directory.resolve().name.lower()
    if "demo" in name:
        return PORT_DEMO
    return PORT_MAIN


def _port_is_open(port: int) -> bool:
    """Check if a port is already bound (i.e. a server is already listening)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect(("localhost", port))
            return True
        except (ConnectionRefusedError, OSError):
            return False


@app.command()
def serve(
    directory: Annotated[
        Path,
        typer.Option("--dir", help="Directory to serve (default: artifacts)"),
    ] = ARTIFACTS_ROOT,
    port: Annotated[
        int | None,
        typer.Option("--port", help="Port to serve on (auto-detected if omitted)"),
    ] = None,
    no_open: Annotated[
        bool,
        typer.Option("--no-open", help="Don't auto-open browser"),
    ] = False,
) -> None:
    """Start a local HTTP server for viewing NVision results.

    Serves the artifacts directory so that all graphs, iframes, and
    fetch-based resources load correctly in the browser.
    Uses port 8080 for main artifacts and 8081 for demo artifacts.
    """
    directory = directory.resolve()
    if not directory.exists():
        console.print(f"[bold red]Directory not found:[/bold red] {directory}")
        raise typer.Exit(1)

    index = directory / "index.html"
    if not index.exists():
        console.print(f"[yellow]Warning: no index.html in {directory}[/yellow]")
        console.print("[dim]Run 'nvision run' or 'nvision demo' first to generate results.[/dim]")
        raise typer.Exit(1)

    if port is None:
        port = _default_port_for_dir(directory)

    url = f"http://localhost:{port}"

    # If port is already in use, assume existing server — just open browser
    if _port_is_open(port):
        console.print(f"[bold cyan]Server already running:[/bold cyan] {url}")
        if not no_open:
            webbrowser.open(url)
        return

    console.print(f"[bold cyan]Serving:[/bold cyan] {directory}")
    console.print(f"[bold cyan]URL:[/bold cyan]     {url}")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    if not no_open:
        webbrowser.open(url)

    import os

    original_dir = os.getcwd()
    try:
        os.chdir(directory)
        with socketserver.TCPServer(("", port), _QuietHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[bold]Server stopped.[/bold]")
    except OSError as e:
        if "Address already in use" in str(e) or "Only one usage" in str(e):
            console.print(f"[bold cyan]Server already running:[/bold cyan] {url}")
            if not no_open:
                webbrowser.open(url)
        else:
            raise
    finally:
        os.chdir(original_dir)
