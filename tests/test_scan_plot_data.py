"""Manifest ``plot_data`` for static head-to-head must be strict JSON."""

from __future__ import annotations

import json

import polars as pl

from nvision.models.experiment import CoreExperiment
from nvision.spectra.gaussian import GaussianModel, GaussianParams
from nvision.spectra.lorentzian import LorentzianModel, LorentzianParams
from nvision.spectra.signal import TrueSignal
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.viz.measurements import (
    backfill_scan_plot_data_if_missing,
    compute_scan_plot_data,
    plot_data_from_scan_figure,
)


def _minimal_scan() -> CoreExperiment:
    bounds = {
        "frequency": (0.0, 1.0),
        "sigma": (0.01, 0.3),
        "amplitude": (0.0, 1.5),
        "background": (0.0, 0.5),
    }
    typed = GaussianParams(frequency=0.5, sigma=0.1, amplitude=1.0, background=0.0)
    true_signal = TrueSignal.from_typed(model=GaussianModel(), params=typed, bounds=bounds)
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


def test_compute_scan_plot_data_mode_curve_uses_belief_unit_cube() -> None:
    """MAP overlay uses the inference model when ground-truth parameter names differ."""
    x_min, x_max = 2.6e9, 3.1e9
    lorentz_bounds = {
        "frequency": (x_min, x_max),
        "linewidth": (5e6, 100e6),
        "amplitude": (1e-6, 1.0),
        "background": (0.5, 1.2),
    }
    typed = LorentzianParams(
        frequency=2.85e9,
        linewidth=30e6,
        amplitude=0.01,
        background=1.0,
    )
    true_signal = TrueSignal.from_typed(
        model=LorentzianModel(),
        params=typed,
        bounds=lorentz_bounds,
    )
    scan = CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)

    phys = {
        "frequency": (x_min, x_max),
        "sigma": (5e6, 100e6),
        "amplitude": (0.1, 1.4),
        "background": (0.0, 0.5),
    }
    belief_uc = UnitCubeSignalModel(GaussianModel(), phys, (x_min, x_max))
    mode_estimates = {
        "frequency": 2.85e9,
        "sigma": 30e6,
        "amplitude": 1.0,
        "background": 0.0,
    }
    hist = pl.DataFrame({"x": [2.8e9], "signal_values": [0.5]})
    without = compute_scan_plot_data(scan, hist, None, mode_estimates=mode_estimates)
    assert "y_dense_mode" not in without
    with_belief = compute_scan_plot_data(scan, hist, None, mode_estimates=mode_estimates, belief_unit_cube=belief_uc)
    assert "y_dense_mode" in with_belief
    assert len(with_belief["y_dense_mode"]) == len(with_belief["x_dense"])
    json.dumps(with_belief)


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
