"""Serve artifacts via a local HTTP server.

Usage:
    uv run python -m nvision serve                       # Serve main artifacts (port 8080)
    uv run python -m nvision serve --dir demo_artifacts  # Serve demo artifacts (port 8081)
    uv run python -m nvision serve --port 9000           # Custom port

Keyboard shortcuts (in browser):
    'r' - Reload/recalculate results
"""

from __future__ import annotations

import http.server
import json
import logging
import socketserver
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Annotated, ClassVar

import typer
from rich.console import Console

from nvision.cli.app_instance import app
from nvision.tools.paths import ARTIFACTS_ROOT

log = logging.getLogger("nvision")
console = Console()

# Well-known ports for each artifacts directory (high numbers to avoid conflicts)
PORT_MAIN = 18080
PORT_DEMO = 18081

# Global state for reload tracking
_reload_state: dict = {"running": False, "last_output": ""}


class _APIHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with API endpoints for reload functionality."""

    # Class variables set by the server
    directory_to_serve: ClassVar[Path] = Path(".")
    is_demo: ClassVar[bool] = False

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:
        """Handle GET requests - check for API endpoints first."""
        if self.path == "/api/status":
            self._send_json({
                "reload_running": _reload_state["running"],
                "last_output": _reload_state["last_output"],
            })
            return
        # Fall through to static file serving
        super().do_GET()

    def do_POST(self) -> None:
        """Handle POST requests for API endpoints."""
        if self.path == "/api/reload":
            self._handle_reload()
            return
        self.send_error(404, "Not found")

    def _handle_reload(self) -> None:
        """Trigger a reload/recalculation in the background."""
        global _reload_state

        if _reload_state["running"]:
            self._send_json({"status": "already_running", "message": "Reload already in progress"})
            return

        _reload_state["running"] = True
        _reload_state["last_output"] = ""

        # Start reload in background thread
        thread = threading.Thread(target=self._run_reload, daemon=True)
        thread.start()

        self._send_json({"status": "started", "message": "Reload started"})

    def _run_reload(self) -> None:
        """Run the actual reload command."""
        global _reload_state

        try:
            if self.is_demo:
                # Run demo command
                cmd = [sys.executable, "-m", "nvision", "demo", "--no-open"]
            else:
                # Run render to regenerate from cache
                cmd = [sys.executable, "-m", "nvision", "render"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.directory_to_serve.parent),
            )
            _reload_state["last_output"] = result.stdout + result.stderr
            if result.returncode != 0:
                _reload_state["last_output"] += f"\n[Exit code: {result.returncode}]"
        except Exception as e:
            _reload_state["last_output"] = f"Error: {e}"
        finally:
            _reload_state["running"] = False

    def _send_json(self, data: dict) -> None:
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def end_headers(self) -> None:
        """Add CORS headers for API requests."""
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


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
    Uses port 18080 for main artifacts and 18081 for demo artifacts.

    Press 'r' in the browser to reload/recalculate results.
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

    # Configure handler class variables
    _APIHandler.directory_to_serve = directory
    _APIHandler.is_demo = "demo" in directory.name.lower()

    console.print(f"[bold cyan]Serving:[/bold cyan] {directory}")
    console.print(f"[bold cyan]URL:[/bold cyan]     {url}")
    console.print("[dim]Keyboard: 'r' = reload/recalculate | Ctrl+C = stop[/dim]")

    if not no_open:
        webbrowser.open(url)

    import os

    original_dir = os.getcwd()
    try:
        os.chdir(directory)
        with socketserver.TCPServer(("", port), _APIHandler) as httpd:
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
