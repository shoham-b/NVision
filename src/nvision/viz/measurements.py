from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from nvision.models.noise import CompositeOverFrequencyNoise
from nvision.sim.batch import DataBatch
from nvision.tools.paths import ensure_out_dir


def _make_scan_figure(has_metrics: bool) -> go.Figure:
    if has_metrics:
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
    measurement_xs: list[float] | None = None,
    measurement_ys: list[float] | None = None,
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
    # UI expectation: noisy curve should visually pass through the sampled measurement points.
    # We "splice" the noisy curve values at the measurement x locations.
    if measurement_xs and measurement_ys and len(measurement_xs) == len(measurement_ys):
        for x_m, y_m in zip(measurement_xs, measurement_ys, strict=False):
            try:
                xm = float(x_m)
                ym = float(y_m)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(ym):
                continue
            idxs = np.nonzero(np.isclose(xs, xm, rtol=0.0, atol=1e-12))[0]
            if idxs.size:
                noisy_values[int(idxs[0])] = ym
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


def _extract_history_xy(history: pl.DataFrame) -> tuple[list[Any], list[Any]]:
    xs_s = history.get_column("x").to_list() if "x" in history.columns else []
    ys_s = history.get_column("signal_values").to_list() if "signal_values" in history.columns else []
    return xs_s, ys_s


def _dense_xs_with_measurements(scan: Any, history_xs: list[Any], *, n_dense: int = 1000) -> np.ndarray:
    xs_base = np.linspace(scan.x_min, scan.x_max, n_dense)
    if not history_xs:
        return xs_base
    history_xs_arr = np.asarray([float(x) for x in history_xs if x is not None], dtype=float)
    if history_xs_arr.size == 0:
        return xs_base
    return np.unique(np.concatenate([xs_base, history_xs_arr]))


def _compute_noisy_dense_values(
    xs: np.ndarray,
    ys: list[float],
    over_frequency_noise: CompositeOverFrequencyNoise,
) -> list[float]:
    dense_batch = DataBatch.from_arrays(xs, ys, meta={})
    noisy_batch = over_frequency_noise.apply(dense_batch, random.Random(0))
    noisy_vals: list[float] = []
    for i, v in enumerate(noisy_batch.signal_values):
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv):
            noisy_vals.append(ys[i])
        else:
            noisy_vals.append(fv)
    return noisy_vals


def _splice_noisy_dense_at_measurements(
    xs: np.ndarray,
    noisy_vals: list[float],
    history_xs: list[Any],
    history_ys: list[Any],
) -> None:
    """In-place splice noisy dense curve so it passes through measurement points."""
    for x_m, y_m in zip(history_xs, history_ys, strict=False):
        try:
            xm = float(x_m)
            ym = float(y_m)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ym):
            continue
        idxs = np.nonzero(np.isclose(xs, xm, rtol=0.0, atol=1e-12))[0]
        if idxs.size:
            noisy_vals[int(idxs[0])] = ym


def _measurements_from_history(history: pl.DataFrame) -> dict[str, Any]:
    if history.height == 0:
        return {"mode": "empty"}

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
        return {
            "mode": "phases",
            "coarse_x": coarse_x,
            "coarse_y": coarse_y,
            "fine_x": fine_x,
            "fine_y": fine_y,
        }

    xs_s = history.get_column("x").to_list() if "x" in history.columns else []
    ys_s = history.get_column("signal_values").to_list() if "signal_values" in history.columns else []
    steps = list(range(history.height))
    return {
        "mode": "steps",
        "x": [float(x) for x in xs_s],
        "y": [_json_safe_float(y) for y in ys_s],
        "step": steps,
    }


def compute_scan_plot_data(
    scan: Any,
    history: pl.DataFrame,
    over_frequency_noise: CompositeOverFrequencyNoise | None,
) -> dict[str, Any]:
    """Dense curve + measurement points for static UI head-to-head (matches ``plot_scan_measurements``)."""
    history_xs_s, history_ys_s = _extract_history_xy(history)
    xs = _dense_xs_with_measurements(scan, history_xs_s)
    ys = [float(scan.signal(x)) for x in xs]
    out: dict[str, Any] = {
        "x_dense": [float(x) for x in xs],
        "y_dense": ys,
    }
    if over_frequency_noise is not None:
        noisy_vals = _compute_noisy_dense_values(xs, ys, over_frequency_noise)
        _splice_noisy_dense_at_measurements(xs, noisy_vals, history_xs_s, history_ys_s)
        out["y_dense_noisy"] = noisy_vals

    has_metrics = history.height > 0 and any(col in history.columns for col in ["entropy", "max_prob", "uncertainty"])
    out["has_metrics"] = has_metrics
    out["measurements"] = _measurements_from_history(history)
    return out


def _parse_figure_from_scan_html(html: str) -> go.Figure | None:
    """Rebuild a :class:`plotly.graph_objects.Figure` from ``Plotly.newPlot`` JSON in saved HTML."""
    m = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', html)
    if not m:
        return None
    pos = m.end()
    decoder = json.JSONDecoder()
    try:
        data, pos = decoder.raw_decode(html, pos)
    except json.JSONDecodeError:
        return None
    while pos < len(html) and html[pos] in " \t\n\r,":
        pos += 1
    try:
        layout, _pos = decoder.raw_decode(html, pos)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not isinstance(layout, dict):
        return None
    return go.Figure(data=data, layout=layout)


def _trace_xy_lists(tr: Any) -> tuple[list[float], list[float | None]]:
    xs: list[float] = []
    ys: list[float | None] = []
    tx = getattr(tr, "x", None)
    ty = getattr(tr, "y", None)
    if tx is not None:
        for v in tx:
            xs.append(float(v))
    if ty is not None:
        for v in ty:
            ys.append(_json_safe_float(v))
    n = min(len(xs), len(ys))
    return xs[:n], ys[:n]


def plot_data_from_scan_figure(fig: go.Figure) -> dict[str, Any] | None:  # noqa: C901
    """Rebuild ``plot_data`` from a scan figure (must match ``plot_scan_measurements``)."""
    x_dense: list[float] | None = None
    y_dense: list[float] | None = None
    y_dense_noisy: list[float] | None = None
    coarse_x: list[float] = []
    coarse_y: list[float | None] = []
    fine_x: list[float] = []
    fine_y: list[float | None] = []
    step_x: list[float] = []
    step_y: list[float | None] = []
    step_idx: list[float] = []

    has_metrics = any(getattr(t, "name", None) in ("Entropy", "Uncertainty") for t in fig.data)

    for tr in fig.data:
        name = getattr(tr, "name", None) or ""
        mode = getattr(tr, "mode", "") or ""
        if name == "true signal" and "lines" in mode:
            x_dense, y_dense = _trace_xy_lists(tr)
            continue
        if name == "simulated noisy signal (over-frequency)" and "lines" in mode:
            _, yn = _trace_xy_lists(tr)
            y_dense_noisy = []
            for i, v in enumerate(yn):
                if v is not None:
                    y_dense_noisy.append(float(v))
                elif y_dense is not None and i < len(y_dense) and y_dense[i] is not None:
                    y_dense_noisy.append(float(y_dense[i]))
                else:
                    y_dense_noisy.append(0.0)
            continue
        if name == "measurements (coarse)":
            coarse_x, coarse_y = _trace_xy_lists(tr)
            continue
        if name == "measurements (inference)":
            fine_x, fine_y = _trace_xy_lists(tr)
            continue
        if name == "measurements (noisy)":
            step_x, step_y = _trace_xy_lists(tr)
            mk = getattr(tr, "marker", None)
            if mk is not None:
                c = getattr(mk, "color", None)
                if c is not None and hasattr(c, "__iter__") and not isinstance(c, (str, bytes)):
                    step_idx = [float(v) for v in c]
            continue

    if x_dense is None or y_dense is None:
        return None

    out: dict[str, Any] = {
        "x_dense": [float(x) for x in x_dense],
        "y_dense": [float(y) for y in y_dense if y is not None],
        "has_metrics": has_metrics,
    }
    if y_dense_noisy is not None and len(y_dense_noisy) == len(out["x_dense"]):
        out["y_dense_noisy"] = y_dense_noisy

    if coarse_x or fine_x:
        out["measurements"] = {
            "mode": "phases",
            "coarse_x": coarse_x,
            "coarse_y": coarse_y,
            "fine_x": fine_x,
            "fine_y": fine_y,
        }
    elif step_x:
        if len(step_idx) != len(step_x):
            step_idx = [float(i) for i in range(len(step_x))]
        out["measurements"] = {
            "mode": "steps",
            "x": step_x,
            "y": step_y,
            "step": [int(s) for s in step_idx],
        }
    else:
        out["measurements"] = {"mode": "empty"}

    return out


def backfill_scan_plot_data_if_missing(entry: dict[str, Any], out_dir: Path) -> None:
    """If a scan manifest entry has no ``plot_data``, rebuild it from the saved scan HTML on disk."""
    if entry.get("type") != "scan" or entry.get("plot_data"):
        return
    rel = entry.get("path")
    if not isinstance(rel, str) or not rel.strip():
        return
    path = out_dir / rel
    if not path.exists():
        return
    try:
        html = path.read_text(encoding="utf-8")
        fig = _parse_figure_from_scan_html(html)
        if fig is None:
            return
        plot_data = plot_data_from_scan_figure(fig)
    except Exception:
        return
    if plot_data:
        entry["plot_data"] = plot_data


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

        history_xs_raw = history.get_column("x").to_list() if "x" in history.columns else []
        history_ys_raw = history.get_column("signal_values").to_list() if "signal_values" in history.columns else []
        history_xs_arr: np.ndarray | None = None
        if history_xs_raw:
            history_xs_arr = np.asarray([float(x) for x in history_xs_raw if x is not None], dtype=float)
            if history_xs_arr.size == 0:
                history_xs_arr = None

        xs_base = np.linspace(scan.x_min, scan.x_max, 1000)
        xs = np.unique(np.concatenate([xs_base, history_xs_arr])) if history_xs_arr is not None else xs_base
        ys = [float(scan.signal(x)) for x in xs]

        measurement_xs: list[float] = []
        measurement_ys: list[float] = []
        for x_m, y_m in zip(history_xs_raw, history_ys_raw, strict=False):
            try:
                xm = float(x_m)
                ym = float(y_m)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(ym):
                continue
            measurement_xs.append(xm)
            measurement_ys.append(ym)

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
            measurement_xs=measurement_xs if measurement_xs else None,
            measurement_ys=measurement_ys if measurement_ys else None,
        )
        _add_history_traces(fig, history, has_metrics)
        _add_metric_traces(fig, history, has_metrics)
        _scan_layout(fig, has_metrics)

        fig.write_html(out_path.as_posix(), include_plotlyjs="cdn")
        return out_path
