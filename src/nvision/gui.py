from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nicegui import app, ui


def load_manifest(out_dir: Path) -> list[dict[str, Any]]:
    manifest_path = out_dir / "plots_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def run_gui(out_dir: Path, port: int = 8080, show: bool = True) -> None:
    # Serve static files from out_dir so iframe can access them
    # We mount out_dir to /plots
    app.add_static_files("/plots", out_dir)

    @ui.page("/")
    def index() -> None:
        plots = load_manifest(out_dir)

        # Filter for scan plots
        scan_plots = [p for p in plots if p.get("type") == "scan"]

        # Extract unique options
        generators = sorted(
            list({p.get("generator", "") for p in scan_plots if p.get("generator")})
        )
        noises = sorted(list({p.get("noise", "") for p in scan_plots if p.get("noise")}))
        strategies = sorted(list({p.get("strategy", "") for p in scan_plots if p.get("strategy")}))

        # State
        state = {
            "generator": generators[0] if generators else None,
            "noise": noises[0] if noises else None,
            "strategy": strategies[0] if strategies else None,
            "repeat": 1,
        }

        def get_current_plot() -> dict[str, Any] | None:
            gen = state["generator"]
            noise = state["noise"]
            strat = state["strategy"]
            rep = state["repeat"]

            for p in scan_plots:
                if (
                    p.get("generator") == gen
                    and p.get("noise") == noise
                    and p.get("strategy") == strat
                    and p.get("repeat") == rep
                ):
                    return p
            return None

        def update_content() -> None:
            plot = get_current_plot()
            if plot:
                # Update iframe
                plot_path = plot.get("path", "")
                if plot_path:
                    iframe.props(f'src="/plots/{plot_path}"')
                else:
                    iframe.props('src=""')

                # Update metrics
                metrics_label.text = (
                    f"Measurements: {plot.get('measurements', 'N/A')} | "
                    f"Duration: {plot.get('duration_ms', 'N/A')} ms | "
                    f"Abs Error: {plot.get('abs_err_x', 'N/A')} | "
                    f"Uncertainty: {plot.get('uncert', 'N/A')}"
                )
            else:
                iframe.props('src=""')
                metrics_label.text = "No data found for selection"

        def on_change(key: str, value: Any) -> None:
            state[key] = value
            update_content()

        # Layout
        with ui.row().classes("w-full h-screen"):
            # Sidebar
            with ui.column().classes("w-64 p-4 border-r bg-gray-50 h-full"):
                ui.label("NVision Results").classes("text-xl font-bold mb-4")

                ui.select(
                    generators,
                    label="Generator",
                    value=state["generator"],
                    on_change=lambda e: on_change("generator", e.value),
                ).classes("w-full")
                ui.select(
                    noises,
                    label="Noise",
                    value=state["noise"],
                    on_change=lambda e: on_change("noise", e.value),
                ).classes("w-full")
                ui.select(
                    strategies,
                    label="Strategy",
                    value=state["strategy"],
                    on_change=lambda e: on_change("strategy", e.value),
                ).classes("w-full")

                ui.number(
                    label="Repeat",
                    value=state["repeat"],
                    min=1,
                    on_change=lambda e: on_change("repeat", e.value),
                ).classes("w-full")

            # Main Content
            with ui.column().classes("flex-grow p-4 h-full"):
                metrics_label = ui.label("").classes("text-lg mb-2")
                iframe = ui.element("iframe").classes("w-full h-full border rounded")

        update_content()

    ui.run(title="NVision Results", reload=False, port=port, show=show)
