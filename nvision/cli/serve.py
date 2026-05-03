"""Serve artifacts via a local HTTP server.

Usage:
    uv run python -m nvision serve                       # Serve main artifacts (port 8080)
    uv run python -m nvision serve --dir demo_artifacts  # Serve demo artifacts (port 8081)
    uv run python -m nvision serve --port 9000           # Custom port
    uv run python -m nvision serve --demo                # Run demo then serve results

Keyboard shortcuts (in browser):
    'r' - Reload/recalculate results
"""

from __future__ import annotations

import contextlib
import http.server
import json
import logging
import socketserver
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from nvision.cli import ClassVar, defaults
from nvision.cli.app_instance import app
from nvision.tools.paths import ARTIFACTS_ROOT

log = logging.getLogger("nvision")
console = Console()

# Well-known ports for each artifacts directory (high numbers to avoid conflicts)
PORT_MAIN = 18080
PORT_DEMO = 18081
PORT_BETA = 18082

# Global state for reload tracking
_reload_state: dict = {"running": False, "last_output": ""}

# Global server instance for shutdown control
_server_instance: socketserver.TCPServer | None = None
_server_thread: threading.Thread | None = None


class _APIHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with API endpoints for reload functionality."""

    # Class variables set by the server
    directory_to_serve: ClassVar[Path] = Path(".")
    is_demo: ClassVar[bool] = False

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests - check for API endpoints first."""
        if self.path == "/api/status":
            self._send_json(
                {
                    "reload_running": _reload_state["running"],
                    "last_output": _reload_state["last_output"],
                }
            )
            return
        # Fall through to static file serving
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST requests for API endpoints."""
        if self.path == "/api/reload":
            self._handle_reload()
            return
        if self.path == "/api/stop":
            self._handle_stop()
            return
        self.send_error(404, "Not found")

    def _handle_stop(self) -> None:
        """Trigger server shutdown."""
        global _server_instance

        self._send_json({"status": "stopping", "message": "Server shutting down"})

        # Shutdown server in a thread to avoid blocking the response
        if _server_instance is not None:
            thread = threading.Thread(target=_server_instance.shutdown, daemon=True)
            thread.start()

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

    def handle(self) -> None:
        """Handle requests with graceful client disconnect handling."""
        with contextlib.suppress(ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            super().handle()  # Silently ignore benign client disconnects (browser refresh/close)


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

    # First: can we connect at TCP level?
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect(("localhost", port))
        except (ConnectionRefusedError, OSError):
            return False

    # Second: does it actually respond to HTTP requests?
    # This filters out TIME_WAIT sockets and zombie processes
    try:
        req = urllib.request.Request(f"http://localhost:{port}/api/status", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


@app.command()
def serve(  # noqa: C901
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
    gcp: Annotated[
        bool,
        typer.Option("--gcp", help="Serve from GCP instead of local"),
    ] = defaults.DEFAULT_GCP,
    gcp_bucket: Annotated[
        str | None,
        typer.Option("--gcp-bucket", help="GCP bucket to serve results from"),
    ] = defaults.DEFAULT_GCP_BUCKET,
) -> None:
    """Start a local HTTP server for viewing NVision results.

    Serves the artifacts directory so that all graphs, iframes, and
    fetch-based resources load correctly in the browser.
    Uses port 18080 for main artifacts and 18081 for demo artifacts.

    Press 'r' in the browser to reload/recalculate results.
    Use --background to run server in background and return immediately.
    """
    # Run demo first if requested
    if demo:
        from nvision.cli.demo import DEMO_ARTIFACTS_ROOT
        from nvision.cli.demo import demo as demo_cmd

        directory = DEMO_ARTIFACTS_ROOT
        index = directory / "index.html"
        if not index.exists():
            console.print("[bold cyan]Running demo first...[/bold cyan]")
            result = demo_cmd(open_browser=False, gcp=gcp, gcp_bucket=gcp_bucket)
            if result != 0:
                console.print("[bold red]Demo failed![/bold red]")
                raise typer.Exit(result)

    if gcp:
        if not gcp_bucket:
            console.print("[bold red]Error:[/bold red] --gcp requires --gcp-bucket to be specified")
            raise typer.Exit(1)
        from nvision.tools.gcp import get_public_url

        url = get_public_url(gcp_bucket, directory.resolve().name)
        console.print(f"[bold cyan]Serving from GCP:[/bold cyan] {url}")
        if not no_open:
            webbrowser.open(url)
        return

    directory = directory.resolve()
    if not directory.exists():
        console.print(f"[bold red]Directory not found:[/bold red] {directory}")
        raise typer.Exit(1)

    index = directory / "index.html"
    if not index.exists():
        console.print(f"[yellow]Warning: no index.html in {directory}[/yellow]")
        console.print("[dim]Run 'nvision run' or 'nvision demo' first to generate results.[/dim]")
        raise typer.Exit(1)

    import os

    is_render = os.environ.get("RENDER") == "true"
    if is_render:
        no_open = True

    if port is None:
        port = int(os.environ["PORT"]) if is_render and "PORT" in os.environ else _default_port_for_dir(directory)

    if is_render and "RENDER_EXTERNAL_URL" in os.environ:
        url = os.environ["RENDER_EXTERNAL_URL"]
    else:
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

    class _ReuseAddrTCPServer(socketserver.TCPServer):
        """TCP server that allows address reuse (critical on Windows)."""

        allow_reuse_address = True

    def _run_server() -> None:
        global _server_instance
        is_render = os.environ.get("RENDER") == "true"
        try:
            os.chdir(directory)
            host = "0.0.0.0" if is_render else ""
            with _ReuseAddrTCPServer((host, port), _APIHandler) as httpd:
                _server_instance = httpd
                httpd.serve_forever(poll_interval=0.1)
        except OSError as e:
            if "Address already in use" in str(e) or "Only one usage" in str(e):
                log.debug("Server already running on port %s", port)
            else:
                log.error("Server error: %s", e)
        finally:
            _server_instance = None
            os.chdir(original_dir)

    if background:
        thread = threading.Thread(target=_run_server, daemon=True)
        thread.start()
        console.print("[bold green]Server running in background.[/bold green]")
        return

    try:
        _run_server()
    except KeyboardInterrupt:
        log.warning("Server interrupted by user (Ctrl+C)")
        console.print("\n[yellow]Interrupted by user. Stopping server...[/yellow]")
        # Graceful shutdown of server
        if _server_instance is not None:
            _server_instance.shutdown()
    except OSError as e:
        if "Address already in use" in str(e) or "Only one usage" in str(e):
            console.print(f"[bold cyan]Server already running:[/bold cyan] {url}")
            if not no_open:
                webbrowser.open(url)
        else:
            raise


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
    import os

    is_render = os.environ.get("RENDER") == "true"

    if port is None:
        port = int(os.environ["PORT"]) if is_render and "PORT" in os.environ else _default_port_for_dir(directory)

    if is_render and "RENDER_EXTERNAL_URL" in os.environ:
        url = os.environ["RENDER_EXTERNAL_URL"]
    else:
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
