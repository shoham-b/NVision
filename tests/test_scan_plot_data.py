"""Manifest ``plot_data`` for static head-to-head must be strict JSON."""

from __future__ import annotations

import json

import polars as pl

from nvision.models.experiment import CoreExperiment
from nvision.signal.gaussian import GaussianModel
from nvision.signal.signal import Parameter, TrueSignal
from nvision.viz.measurements import (
    backfill_scan_plot_data_if_missing,
    compute_scan_plot_data,
    plot_data_from_scan_figure,
)


def _minimal_scan() -> CoreExperiment:
    model = GaussianModel()
    parameters = [
        Parameter(name="frequency", bounds=(0.0, 1.0), value=0.5),
        Parameter(name="sigma", bounds=(0.01, 0.3), value=0.1),
        Parameter(name="amplitude", bounds=(0.0, 1.5), value=1.0),
        Parameter(name="background", bounds=(0.0, 0.5), value=0.0),
    ]
    true_signal = TrueSignal(model=model, parameters=parameters)
    return CoreExperiment(true_signal=true_signal, noise=None, x_min=0.0, x_max=1.0)


def test_compute_scan_plot_data_json_serializable() -> None:
    scan = _minimal_scan()
    hist = pl.DataFrame({"x": [0.1, 0.5], "signal_values": [0.02, 0.3]})
    data = compute_scan_plot_data(scan, hist, None)
    json.dumps(data)  # must not raise (no NaN)


def test_compute_scan_plot_data_phases() -> None:
    scan = _minimal_scan()
    hist = pl.DataFrame(
        {
            "x": [0.1, 0.2],
            "signal_values": [0.02, 0.04],
            "phase": ["coarse", "fine"],
        }
    )
    data = compute_scan_plot_data(scan, hist, None)
    assert data["measurements"]["mode"] == "phases"
    json.dumps(data)


def test_plot_data_from_scan_figure_minimal() -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.0, 1.0], y=[0.1, 0.2], mode="lines", name="true signal"))
    fig.add_trace(
        go.Scatter(
            x=[0.1],
            y=[0.15],
            mode="markers",
            name="measurements (noisy)",
            marker=dict(color=[0], size=8),
        )
    )
    pd = plot_data_from_scan_figure(fig)
    assert pd is not None
    assert pd["measurements"]["mode"] == "steps"


def test_backfill_scan_plot_data_from_saved_html(tmp_path) -> None:
    import plotly.graph_objects as go
    from plotly.io import write_html

    sub = tmp_path / "graphs" / "scans"
    sub.mkdir(parents=True)
    p = sub / "scan.html"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.0, 1.0], y=[0.1, 0.2], mode="lines", name="true signal"))
    write_html(fig, str(p), include_plotlyjs=False)
    entry = {"type": "scan", "path": "graphs/scans/scan.html"}
    backfill_scan_plot_data_if_missing(entry, tmp_path)
    assert "plot_data" in entry
    assert entry["plot_data"]["x_dense"] == [0.0, 1.0]
