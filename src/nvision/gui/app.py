from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Flask, send_from_directory
from dash import Dash, html, dcc, Input, Output


def load_manifest(out_dir: Path) -> list[dict[str, Any]]:
    manifest_path = out_dir / "plots_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def run_gui(out_dir: Path, port: int = 8080, show: bool = True) -> None:
    # Initialize Flask server for custom static serving
    server = Flask(__name__)

    # Serve static files from out_dir
    @server.route("/plots/<path:path>")
    def serve_plots(path):
        return send_from_directory(out_dir, path)

    # Initialize Dash app
    app = Dash(__name__, server=server, title="NVision Results")

    # Load initial data
    plots = load_manifest(out_dir)
    scan_plots = [p for p in plots if p.get("type") == "scan"]

    # Extract unique options
    generators = sorted(list({p.get("generator", "") for p in scan_plots if p.get("generator")}))
    noises = sorted(list({p.get("noise", "") for p in scan_plots if p.get("noise")}))
    strategies = sorted(list({p.get("strategy", "") for p in scan_plots if p.get("strategy")}))

    # Defaults
    default_gen = generators[0] if generators else None
    default_noise = noises[0] if noises else None
    default_strat = strategies[0] if strategies else None
    default_rep = 1

    app.layout = html.Div(
        className="w-full h-screen flex flex-row",
        style={"display": "flex", "height": "100vh", "width": "100%"},
        children=[
            # Sidebar
            html.Div(
                className="w-64 p-4 border-r bg-gray-50 h-full",
                style={
                    "width": "300px",
                    "padding": "20px",
                    "borderRight": "1px solid #ddd",
                    "backgroundColor": "#f9fafb",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "15px",
                },
                children=[
                    html.H1(
                        "NVision Results",
                        style={"fontSize": "1.25rem", "fontWeight": "bold", "marginBottom": "1rem"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Generator", style={"fontWeight": "bold", "fontSize": "0.9rem"}
                            ),
                            dcc.Dropdown(
                                id="generator-dropdown",
                                options=generators,
                                value=default_gen,
                                clearable=False,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Noise", style={"fontWeight": "bold", "fontSize": "0.9rem"}),
                            dcc.Dropdown(
                                id="noise-dropdown",
                                options=noises,
                                value=default_noise,
                                clearable=False,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Strategy", style={"fontWeight": "bold", "fontSize": "0.9rem"}
                            ),
                            dcc.Dropdown(
                                id="strategy-dropdown",
                                options=strategies,
                                value=default_strat,
                                clearable=False,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Repeat", style={"fontWeight": "bold", "fontSize": "0.9rem"}
                            ),
                            dcc.Input(
                                id="repeat-input",
                                type="number",
                                min=1,
                                value=default_rep,
                                style={"width": "100%"},
                            ),
                        ]
                    ),
                ],
            ),
            # Main Content
            html.Div(
                className="flex-grow p-4 h-full flex flex-col",
                style={
                    "flexGrow": 1,
                    "padding": "20px",
                    "display": "flex",
                    "flexDirection": "column",
                    "height": "100%",
                },
                children=[
                    html.Div(
                        id="metrics-display", style={"marginBottom": "10px", "fontSize": "1.1rem"}
                    ),
                    html.Iframe(
                        id="plot-frame",
                        style={
                            "width": "100%",
                            "flexGrow": 1,
                            "border": "1px solid #ddd",
                            "borderRadius": "4px",
                        },
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        [Output("plot-frame", "src"), Output("metrics-display", "children")],
        [
            Input("generator-dropdown", "value"),
            Input("noise-dropdown", "value"),
            Input("strategy-dropdown", "value"),
            Input("repeat-input", "value"),
        ],
    )
    def update_view(gen, noise, strat, rep):
        target_plot = None
        for p in scan_plots:
            if (
                p.get("generator") == gen
                and p.get("noise") == noise
                and p.get("strategy") == strat
                and p.get("repeat") == rep
            ):
                target_plot = p
                break

        if target_plot:
            path = target_plot.get("path", "")
            src = f"/plots/{path}" if path else ""

            metrics = (
                f"Measurements: {target_plot.get('measurements', 'N/A')} | "
                f"Duration: {target_plot.get('duration_ms', 'N/A')} ms | "
                f"Abs Error: {target_plot.get('abs_err_x', 'N/A')} | "
                f"Uncertainty: {target_plot.get('uncert', 'N/A')}"
            )
            return src, metrics
        else:
            return "", "No data found for selection"

    # Turn off reloader if we want to mimic simple "run and hold" behavior without dev tools usually
    # But usually 'show' implies opening browser.
    if show:
        import webbrowser
        from threading import Timer

        def open_browser():
            webbrowser.open_new(f"http://localhost:{port}")

        Timer(1, open_browser).start()

    app.run(port=port, debug=False)
