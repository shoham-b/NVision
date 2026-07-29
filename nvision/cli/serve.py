"""Serve artifacts via a local FastAPI/uvicorn server, backed live by the SQLite cache.

Usage:
    uv run python -m nvision serve                       # Serve main artifacts (port 18080)
    uv run python -m nvision serve --dir demo_artifacts  # Serve demo artifacts (port 18081)
    uv run python -m nvision serve --port 9000           # Custom port

Keyboard shortcuts (in browser):
    'r' - Reload/recalculate results

Nothing is pre-rendered to disk: the manifest and every graph/aggregate view are
built on demand from ``<directory>/cache`` on each request (see
``nvision.cli.api_server``).
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console

from nvision.cli.api_server import build_app
from nvision.cli.app_instance import app
from nvision.tools.paths import ARTIFACTS_ROOT

log = logging.getLogger("nvision")
console = Console()

# Well-known ports for each artifacts directory (high numbers to avoid conflicts)
PORT_MAIN = 18080
PORT_DEMO = 18081
PORT_BETA = 18082

# Global server instance for shutdown control
_server_instance: uvicorn.Server | None = None


def _default_port_for_dir(directory: Path) -> int:
    """Return the well-known port for a directory, or PORT_MAIN as fallback."""
    name = directory.resolve().name.lower()
    if "demo" in name:
        return PORT_DEMO
    if "beta" in name:
        return PORT_BETA
    return PORT_MAIN


def _port_is_open(port: int) -> bool:
    """Check if a healthy server is already running on the port.

    Verifies actual HTTP response to avoid false positives from
    TIME_WAIT sockets or zombie processes on Windows.
    """
    import socket
    import urllib.error
    import urllib.request

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect(("localhost", port))
        except (ConnectionRefusedError, OSError):
            return False

    try:
        req = urllib.request.Request(f"http://localhost:{port}/api/status", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
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
    demo: Annotated[
        bool,
        typer.Option("--demo", help="Run demo first, then serve results"),
    ] = False,
    background: Annotated[
        bool,
        typer.Option("--background", help="Run server in background and exit immediately"),
    ] = False,
    host: Annotated[
        str,
        typer.Option("--host", help="Interface to bind (use 0.0.0.0 to serve outside localhost, e.g. in a container)"),
    ] = "127.0.0.1",
) -> None:
    """Start a local API server for viewing NVision results, live from the cache.

    Serves the frontend assets plus a FastAPI backend that answers every
    manifest/graph/aggregate request straight from ``<directory>/cache`` — nothing
    is written to disk. Uses port 18080 for main artifacts and 18081 for demo.

    Press 'r' in the browser to reload/recalculate results.
    Use --background to run server in background and return immediately.
    """
    if demo:
        from nvision.cli.demo import DEMO_ARTIFACTS_ROOT
        from nvision.cli.demo import demo as demo_cmd

        directory = DEMO_ARTIFACTS_ROOT
        if not (directory / "cache").exists():
            console.print("[bold cyan]Running demo first...[/bold cyan]")
            result = demo_cmd(open_browser=False)
            if result != 0:
                console.print("[bold red]Demo failed![/bold red]")
                raise typer.Exit(result)

    directory = directory.resolve()
    if not directory.exists():
        console.print(f"[bold red]Directory not found:[/bold red] {directory}")
        raise typer.Exit(1)

    if not (directory / "cache").exists():
        console.print(f"[yellow]Warning: no cache found in {directory}[/yellow]")
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
    console.print("[dim]Keyboard: 'r' = reload/recalculate | Ctrl+C = stop[/dim]")

    if not no_open:
        webbrowser.open(url)

    fastapi_app = build_app(cache_dir=directory / "cache", run_dir=directory)
    config = uvicorn.Config(fastapi_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    fastapi_app.state.uvicorn_server = server

    global _server_instance
    _server_instance = server

    if background:
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        console.print("[bold green]Server running in background.[/bold green]")
        return

    try:
        server.run()
    except KeyboardInterrupt:
        log.warning("Server interrupted by user (Ctrl+C)")
        console.print("\n[yellow]Interrupted by user. Stopping server...[/yellow]")


@app.command(name="serve-stop")
def serve_stop(
    port: Annotated[
        int | None,
        typer.Option("--port", help="Port of the server to stop (auto-detected if omitted)"),
    ] = None,
    directory: Annotated[
        Path,
        typer.Option("--dir", help="Directory the server was serving (for port auto-detection)"),
    ] = ARTIFACTS_ROOT,
) -> None:
    """Stop a running background server.

    Sends a shutdown signal to the server on the specified port.
    If port is not provided, auto-detects based on the directory.
    """
    if port is None:
        port = _default_port_for_dir(directory)
    url = f"http://localhost:{port}"

    if not _port_is_open(port):
        console.print(f"[yellow]No server running on port {port}[/yellow]")
        raise typer.Exit(1)

    try:
        import urllib.request

        req = urllib.request.Request(f"{url}/api/stop", method="POST")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            console.print(f"[bold green]Server stopped:[/bold green] {url}")
            console.print(f"[dim]Response: {data.get('message', 'OK')}[/dim]")
    except Exception as e:
        console.print(f"[bold red]Failed to stop server:[/bold red] {e}")
        raise typer.Exit(1) from e
