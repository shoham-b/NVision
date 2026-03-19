from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl

from nvision.models.noise import CompositeOverFrequencyNoise
from nvision.sim.batch import DataBatch
from nvision.tools.paths import ensure_out_dir


def _make_scan_figure(has_metrics: bool) -> go.Figure:
    if has_metrics:
        from plotly.subplots import make_subplots

        return make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.15,
            subplot_titles=("Scan Measurements", "Convergence Metrics"),
            row_heights=[0.7, 0.3],
        )
    return go.Figure()


def _row_col(has_metrics: bool) -> tuple[int | None, int | None]:
    return (1, 1) if has_metrics else (None, None)


def _add_true_and_noisy_traces(
    fig: go.Figure,
    *,
    xs: np.ndarray,
    ys: list[float],
    has_metrics: bool,
    over_frequency_noise: CompositeOverFrequencyNoise | None,
) -> None:
    row, col = _row_col(has_metrics)
    fig.add_trace(
        go.Scatter(x=xs, y=ys, mode="lines", name="true signal", line=dict(color="blue")),
        row=row,
        col=col,
    )
    if over_frequency_noise is None:
        return
    dense_batch = DataBatch.from_arrays(xs, ys, meta={})
    noisy_batch = over_frequency_noise.apply(dense_batch, random.Random(0))
    noisy_values = [float(v) for v in noisy_batch.signal_values]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=noisy_values,
            mode="lines",
            name="simulated noisy signal (over-frequency)",
            line=dict(color="orange", dash="dot"),
        ),
        row=row,
        col=col,
    )


def _add_history_traces(fig: go.Figure, history: pl.DataFrame, has_metrics: bool) -> None:
    if history.height == 0:
        return
    row, col = _row_col(has_metrics)
    xs_s = history.get_column("x").to_list() if "x" in history.columns else []
    ys_s = history.get_column("signal_values").to_list() if "signal_values" in history.columns else []

    if "phase" in history.columns:
        phases = history.get_column("phase").to_list()
        coarse_x = [x for x, p in zip(xs_s, phases, strict=False) if p == "coarse"]
        coarse_y = [y for y, p in zip(ys_s, phases, strict=False) if p == "coarse"]
        fine_x = [x for x, p in zip(xs_s, phases, strict=False) if p == "fine"]
        fine_y = [y for y, p in zip(ys_s, phases, strict=False) if p == "fine"]

        if coarse_x:
            fig.add_trace(
                go.Scatter(
                    x=coarse_x,
                    y=coarse_y,
                    mode="markers",
                    name="measurements (coarse)",
                    marker=dict(size=7, color="rgba(150,150,150,0.8)", line=dict(width=0.5, color="black")),
                ),
                row=row,
                col=col,
            )

        if fine_x:
            fig.add_trace(
                go.Scatter(
                    x=fine_x,
                    y=fine_y,
                    mode="markers",
                    name="measurements (inference)",
                    marker=dict(size=8, color="rgba(214,39,40,0.9)", line=dict(width=0.5, color="black")),
                ),
                row=row,
                col=col,
            )
        return

    steps = list(range(history.height))
    fig.add_trace(
        go.Scatter(
            x=xs_s,
            y=ys_s,
            mode="markers",
            name="measurements (noisy)",
            marker=dict(
                size=8,
                color=steps,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="step", len=0.6, y=0.8) if has_metrics else dict(title="step"),
                line=dict(width=0.5, color="black"),
            ),
        ),
        row=row,
        col=col,
    )


def _add_metric_traces(fig: go.Figure, history: pl.DataFrame, has_metrics: bool) -> None:
    if not has_metrics or history.height == 0:
        return
    steps_arr = np.arange(history.height)
    if "entropy" in history.columns:
        entropy = history.get_column("entropy").to_list()
        valid_mask = [x is not None and not np.isnan(x) for x in entropy]
        if any(valid_mask):
            fig.add_trace(
                go.Scatter(
                    x=steps_arr,
                    y=entropy,
                    mode="lines+markers",
                    name="Entropy",
                    line=dict(color="red"),
                ),
                row=2,
                col=1,
            )

    if "uncertainty" in history.columns:
        uncert = history.get_column("uncertainty").to_list()
        valid_mask = [x is not None and not np.isnan(x) for x in uncert]
        if any(valid_mask):
            fig.add_trace(
                go.Scatter(
                    x=steps_arr,
                    y=uncert,
                    mode="lines+markers",
                    name="Uncertainty",
                    line=dict(color="green"),
                    yaxis="y3" if "entropy" in history.columns else "y2",
                ),
                row=2,
                col=1,
            )


def _json_safe_float(v: Any) -> float | None:
    """Finite floats only; NaN/inf become None for strict JSON manifests."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return f


def compute_scan_plot_data(
    scan: Any,
    history: pl.DataFrame,
    over_frequency_noise: CompositeOverFrequencyNoise | None,
) -> dict[str, Any]:
    """Dense curve + measurement points for static UI head-to-head (matches ``plot_scan_measurements``)."""
    xs = np.linspace(scan.x_min, scan.x_max, 1000)
    ys = [float(scan.signal(x)) for x in xs]
    out: dict[str, Any] = {
        "x_dense": [float(x) for x in xs],
        "y_dense": ys,
    }
    if over_frequency_noise is not None:
        dense_batch = DataBatch.from_arrays(xs, ys, meta={})
        noisy_batch = over_frequency_noise.apply(dense_batch, random.Random(0))
        noisy_vals: list[float] = []
        for i, v in enumerate(noisy_batch.signal_values):
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv):
                noisy_vals.append(ys[i])
            else:
                noisy_vals.append(fv)
        out["y_dense_noisy"] = noisy_vals

    has_metrics = history.height > 0 and any(col in history.columns for col in ["entropy", "max_prob", "uncertainty"])
    out["has_metrics"] = has_metrics

    if history.height == 0:
        out["measurements"] = {"mode": "empty"}
        return out

    if "phase" in history.columns:
        phases = history.get_column("phase").to_list()
        xs_s = history.get_column("x").to_list() if "x" in history.columns else []
        ys_s = history.get_column("signal_values").to_list() if "signal_values" in history.columns else []
        coarse_x: list[float] = []
        coarse_y: list[float | None] = []
        fine_x: list[float] = []
        fine_y: list[float | None] = []
        for x, y, p in zip(xs_s, ys_s, phases, strict=False):
            if p == "coarse":
                coarse_x.append(float(x))
                coarse_y.append(_json_safe_float(y))
            elif p == "fine":
                fine_x.append(float(x))
                fine_y.append(_json_safe_float(y))
        out["measurements"] = {
            "mode": "phases",
            "coarse_x": coarse_x,
            "coarse_y": coarse_y,
            "fine_x": fine_x,
            "fine_y": fine_y,
        }
        return out

    xs_s = history.get_column("x").to_list() if "x" in history.columns else []
    ys_s = history.get_column("signal_values").to_list() if "signal_values" in history.columns else []
    steps = list(range(history.height))
    out["measurements"] = {
        "mode": "steps",
        "x": [float(x) for x in xs_s],
        "y": [_json_safe_float(y) for y in ys_s],
        "step": steps,
    }
    return out


def _scan_layout(fig: go.Figure, has_metrics: bool) -> None:
    layout_args: dict = dict(
        title="Scan with sampled measurements",
        template="plotly_white",
    )
    if has_metrics:
        layout_args.update(
            dict(
                xaxis_title="frequency",
                yaxis_title="intensity",
                xaxis2_title="step",
                yaxis2_title="metric value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=800,
            )
        )
    else:
        layout_args.update(
            dict(
                xaxis_title="frequency",
                yaxis_title="intensity (photon count)",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
        )
    fig.update_layout(**layout_args)


class MeasurementsMixin:
    """Mixin for scan measurement plotting."""

    # Typing for mixin dependency (self.out_dir from VizBase)
    out_dir: Path

    def plot_scan_measurements(
        self,
        scan,
        history: pl.DataFrame,
        out_path: Path,
        over_frequency_noise: CompositeOverFrequencyNoise | None = None,
    ) -> Path:
        """Plot the true scan signal distribution and overlay sampled measurements.

        - True signal: computed densely across [x_min, x_max].
        - Measurements (noisy): points from `history` colored by step order (gradient).
        - Measurements (ideal): corresponding points on the true signal curve.
        - Noisy curve: simulated by applying the provided CompositeNoise to the dense grid.
        """
        ensure_out_dir(out_path.parent)

        xs = np.linspace(scan.x_min, scan.x_max, 1000)
        ys = [float(scan.signal(x)) for x in xs]

        has_metrics = history.height > 0 and any(
            col in history.columns for col in ["entropy", "max_prob", "uncertainty"]
        )

        fig = _make_scan_figure(has_metrics)
        _add_true_and_noisy_traces(
            fig,
            xs=xs,
            ys=ys,
            has_metrics=has_metrics,
            over_frequency_noise=over_frequency_noise,
        )
        _add_history_traces(fig, history, has_metrics)
        _add_metric_traces(fig, history, has_metrics)
        _scan_layout(fig, has_metrics)

        fig.write_html(out_path.as_posix(), include_plotlyjs="cdn")
        return out_path
