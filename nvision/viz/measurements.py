from __future__ import annotations

import contextlib
import json
import logging
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from nvision.models.noise import CompositeOverFrequencyNoise
from nvision.sim.batch import DataBatch
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.tools.paths import ensure_out_dir

# Match ``nvision.spectra.numba_kernels.nv_center_lorentzian_eval`` / Voigt no-split branch.
_NV_SPLIT_ZERO_TOL = 1e-10


def _align_mode_split_with_ground_truth(
    scan: Any,
    mode_estimates: Mapping[str, float],
) -> Mapping[str, float]:
    """If the true spectrum is zero-field NV (split≈0), evaluate the overlay at split=0.

    ``belief_mode_estimates`` uses a **marginal** argmax per grid parameter. The marginal
    mode for ``split`` can sit above zero even when the generated signal is a single
    combined dip, which would draw three dips in the dashed \"locator most likely\" curve.
    """
    if "split" not in mode_estimates:
        return mode_estimates
    ts = getattr(scan, "true_signal", None)
    if ts is None:
        return mode_estimates
    getv = getattr(ts, "get_param_value", None)
    if not callable(getv):
        return mode_estimates
    try:
        truth_split = float(getv("split"))
    except (KeyError, TypeError, ValueError):
        return mode_estimates
    if abs(truth_split) >= _NV_SPLIT_ZERO_TOL:
        return mode_estimates
    out = dict(mode_estimates)
    out["split"] = 0.0
    return out


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


def _noise_scale_for_scan(scan: Any, over_frequency_noise: CompositeOverFrequencyNoise | None) -> float:
    """Return noise scale (fixed to 1.0 to match the non-scaling CoreExperiment.measure)."""
    return 1.0


def _add_true_and_noisy_traces(
    fig: go.Figure,
    *,
    xs: np.ndarray,
    ys: list[float],
    has_metrics: bool,
    over_frequency_noise: CompositeOverFrequencyNoise | None,
    measurement_xs: list[float] | None = None,
    measurement_ys: list[float] | None = None,
    noise_scale: float = 1.0,
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
    noisy_values: list[float] = []
    for i, v in enumerate(noisy_batch.signal_values):
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv):
            noisy_values.append(ys[i])
        elif noise_scale != 1.0:
            noisy_values.append(ys[i] + (fv - ys[i]) * noise_scale)
        else:
            noisy_values.append(fv)
    # Bypassed splicing to prevent wrong noise / spurious spikes artifact
    # in physical experiments with drift and coordinate/time dependencies.
    pass
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


def _dense_model_curve(model: Any, xs: np.ndarray, values_in_order: Sequence[float]) -> np.ndarray | None:
    """Evaluate one parameter set over many ``xs`` in a single vectorized call.

    Uses the model's ``compute_vectorized_many`` kernel with length-1 parameter
    arrays — one batched kernel call instead of thousands of scalar
    ``compute`` calls. Returns ``None`` when the model has no compatible
    vectorized interface; callers fall back to the per-point loop.
    """
    fn = getattr(model, "compute_vectorized_many", None)
    if fn is None:
        return None
    try:
        arrays = [np.full(1, float(v), dtype=np.float32) for v in values_in_order]
        out = np.asarray(fn(np.asarray(xs, dtype=np.float32), arrays))
    except Exception:
        return None
    if out.ndim == 2 and out.shape[0] == len(xs) and out.shape[1] == 1:
        return out[:, 0].astype(float)
    return None


def _true_signal_dense_y(scan: Any, xs: np.ndarray) -> np.ndarray:
    """Dense true-signal curve, vectorized when the model supports it."""
    true_signal = scan.true_signal
    model = getattr(true_signal, "model", None)
    if model is not None and hasattr(true_signal, "typed_parameters"):
        try:
            values = model.spec.pack_params(true_signal.typed_parameters)
        except Exception:
            values = None
        if values is not None:
            curve = _dense_model_curve(model, xs, values)
            if curve is not None:
                return curve
    return np.asarray([float(scan.signal(x)) for x in xs], dtype=float)


def _mode_dense_y_unit_cube(
    model: UnitCubeSignalModel,
    xs: np.ndarray,
    mode_estimates: Mapping[str, float],
) -> np.ndarray | list[float] | None:
    """MAP curve using a unit-cube forward model (physical ``xs`` and physical marginal modes)."""
    names = list(model.parameter_names())
    if not names or not all(name in mode_estimates for name in names):
        return None
    x_lo, x_hi = model.x_bounds_phys
    w = float(x_hi - x_lo)
    if w <= 0:
        return None
    xs_u = (np.asarray(xs, dtype=float) - x_lo) / w
    u_values: list[float] = []
    for name in names:
        lo, hi = model.param_bounds_phys[name]
        hw = float(hi - lo)
        v = float(mode_estimates[name])
        # Treat small split values as zero to show correct number of dips
        if name == "split" and v < _NV_SPLIT_ZERO_TOL:
            v = 0.0
        u = (v - lo) / hw if hw > 0 else 0.5
        u_values.append(min(max(u, 0.0), 1.0))
    curve = _dense_model_curve(model, xs_u, u_values)
    if curve is not None:
        return curve
    typed = model.spec.unpack_params(u_values)
    return [float(model.compute_from_params(float(xu), typed)) for xu in xs_u]


def _mode_belief_dense_y(
    scan: Any,
    xs: np.ndarray,
    mode_estimates: Mapping[str, float],
    *,
    belief_unit_cube: UnitCubeSignalModel | None = None,
) -> np.ndarray | list[float] | None:
    """Evaluate the forward model at ``mode_estimates`` along ``xs`` (physical domain).

    For :class:`~nvision.spectra.unit_cube.UnitCubeSignalModel`, ``mode_estimates`` are
    physical parameters and ``xs`` are physical probe positions (same as the true-signal
    plot); internally normalized coordinates are applied for evaluation.

    When ``belief_unit_cube`` is set (Bayesian runs), it is used instead of
    ``scan.true_signal.model`` so the dashed curve matches the inference model — e.g. NV
    Voigt ground truth with Lorentzian belief still gets a consistent MAP overlay.
    """
    mode_estimates = _align_mode_split_with_ground_truth(scan, mode_estimates)
    if belief_unit_cube is not None:
        return _mode_dense_y_unit_cube(belief_unit_cube, xs, mode_estimates)

    model = getattr(scan.true_signal, "model", None)
    bounds = getattr(scan.true_signal, "bounds", None)
    if model is None or bounds is None:
        return None
    names = list(model.parameter_names())
    if not names or not all(name in mode_estimates for name in names):
        return None

    if isinstance(model, UnitCubeSignalModel):
        return _mode_dense_y_unit_cube(model, xs, mode_estimates)

    values = [float(mode_estimates[name]) for name in names]
    # Treat small split values as zero to show correct number of dips
    if "split" in mode_estimates and float(mode_estimates["split"]) < _NV_SPLIT_ZERO_TOL:
        split_idx = names.index("split")
        values[split_idx] = 0.0
    curve = _dense_model_curve(model, xs, values)
    if curve is not None:
        return curve
    typed = model.spec.unpack_params(values)
    return [float(model.compute_from_params(float(x), typed)) for x in xs]


def _add_mode_belief_trace(
    fig: go.Figure,
    *,
    scan: Any,
    xs: np.ndarray,
    mode_estimates: Mapping[str, float] | None,
    has_metrics: bool,
    belief_unit_cube: UnitCubeSignalModel | None = None,
) -> None:
    """Overlay signal from the locator's approximate MAP / marginal-mode parameters."""
    if not mode_estimates:
        return
    y_mode = _mode_belief_dense_y(scan, xs, mode_estimates, belief_unit_cube=belief_unit_cube)
    if y_mode is None or len(y_mode) == 0:
        return
    row, col = _row_col(has_metrics)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=y_mode,
            mode="lines",
            name="locator most likely signal",
            line=dict(color="#d62728", width=2, dash="dash"),
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
    finite_ys = [float(y) for y in ys_s if y is not None and np.isfinite(float(y))]
    baseline = max(finite_ys) if finite_ys else 1.0
    baseline = baseline if baseline > 1e-12 else 1.0

    def depth_percent(value: float) -> float:
        """Percent drop from baseline at this measurement."""
        return max(0.0, (baseline - float(value)) / baseline * 100.0)

    if "phase" in history.columns:
        phases = history.get_column("phase").to_list()
        coarse_x = [x for x, p in zip(xs_s, phases, strict=False) if p == "coarse"]
        coarse_y = [y for y, p in zip(ys_s, phases, strict=False) if p == "coarse"]
        secondary_x = [x for x, p in zip(xs_s, phases, strict=False) if p == "secondary"]
        secondary_y = [y for y, p in zip(ys_s, phases, strict=False) if p == "secondary"]
        tertiary_x = [x for x, p in zip(xs_s, phases, strict=False) if p == "tertiary"]
        tertiary_y = [y for y, p in zip(ys_s, phases, strict=False) if p == "tertiary"]
        fine_x = [x for x, p in zip(xs_s, phases, strict=False) if p == "fine"]
        fine_y = [y for y, p in zip(ys_s, phases, strict=False) if p == "fine"]
        coarse_depth = [depth_percent(y) for y in coarse_y]
        secondary_depth = [depth_percent(y) for y in secondary_y]
        tertiary_depth = [depth_percent(y) for y in tertiary_y]
        fine_depth = [depth_percent(y) for y in fine_y]

        if coarse_x:
            fig.add_trace(
                go.Scatter(
                    x=coarse_x,
                    y=coarse_y,
                    mode="markers",
                    name="measurements (coarse)",
                    marker=dict(size=7, color="rgba(176,176,176,0.95)", line=dict(width=0.6, color="#4a4a4a")),
                    customdata=coarse_depth,
                    hovertemplate="x=%{x}<br>y=%{y:.4f}<br>down=%{customdata:.1f}%<extra></extra>",
                ),
                row=row,
                col=col,
            )

        if secondary_x:
            fig.add_trace(
                go.Scatter(
                    x=secondary_x,
                    y=secondary_y,
                    mode="markers",
                    name="measurements (secondary)",
                    marker=dict(size=7, color="rgba(255,127,14,0.9)", line=dict(width=0.6, color="#8B4513")),
                    customdata=secondary_depth,
                    hovertemplate="x=%{x}<br>y=%{y:.4f}<br>down=%{customdata:.1f}%<extra></extra>",
                ),
                row=row,
                col=col,
            )

        if tertiary_x:
            fig.add_trace(
                go.Scatter(
                    x=tertiary_x,
                    y=tertiary_y,
                    mode="markers",
                    name="measurements (tertiary)",
                    marker=dict(size=7, color="rgba(148,0,211,0.9)", line=dict(width=0.6, color="#4B0082")),
                    customdata=tertiary_depth,
                    hovertemplate="x=%{x}<br>y=%{y:.4f}<br>down=%{customdata:.1f}%<extra></extra>",
                ),
                row=row,
                col=col,
            )

        if fine_x:
            fine_steps = list(range(len(fine_x)))
            fig.add_trace(
                go.Scatter(
                    x=fine_x,
                    y=fine_y,
                    mode="markers",
                    name="measurements (inference)",
                    marker=dict(
                        size=8,
                        color=fine_steps,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="inference step", len=0.6, y=0.8)
                        if has_metrics
                        else dict(title="inference step"),
                        line=dict(width=0.5, color="black"),
                    ),
                    customdata=fine_depth,
                    hovertemplate="x=%{x}<br>y=%{y:.4f}<br>down=%{customdata:.1f}%<extra></extra>",
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
            customdata=[depth_percent(y) for y in ys_s],
            hovertemplate="x=%{x}<br>y=%{y:.4f}<br>down=%{customdata:.1f}%<br>step=%{marker.color}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def _add_focus_window_overlay(
    fig: go.Figure,
    *,
    focus_window: tuple[float, float] | None,
    has_metrics: bool,
) -> None:
    """Overlay focused Bayesian acquisition region as a shaded vertical band."""
    if focus_window is None:
        return
    x0, x1 = float(focus_window[0]), float(focus_window[1])
    if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
        return
    row, col = _row_col(has_metrics)
    fig.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor="rgba(46, 204, 113, 0.16)",
        line_width=1,
        line_color="rgba(46, 204, 113, 0.7)",
        layer="below",
        annotation_text="Bayesian focus area",
        annotation_position="top left",
        row=row,
        col=col,
    )


def _add_per_dip_windows_overlay(
    fig: go.Figure,
    *,
    per_dip_windows: list[tuple[float, float]] | None,
    has_metrics: bool,
) -> None:
    """Overlay individual per-dip focus windows as colored vertical bands.

    Each dip window is shown with a distinct color to distinguish individual dips.
    """
    if per_dip_windows is None or not per_dip_windows:
        return
    row, col = _row_col(has_metrics)
    # Use distinct colors for different dips (orange, purple, cyan for up to 3 dips)
    colors = [
        ("rgba(255, 159, 64, 0.20)", "rgba(255, 159, 64, 0.8)"),  # Orange
        ("rgba(153, 102, 255, 0.20)", "rgba(153, 102, 255, 0.8)"),  # Purple
        ("rgba(54, 162, 235, 0.20)", "rgba(54, 162, 235, 0.8)"),  # Blue
    ]
    for i, (lo, hi) in enumerate(per_dip_windows):
        x0, x1 = float(lo), float(hi)
        if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
            continue
        fill_color, line_color = colors[i % len(colors)]
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=fill_color,
            line_width=1,
            line_color=line_color,
            layer="below",
            annotation_text=f"Dip {i + 1}",
            annotation_position="top left",
            row=row,
            col=col,
        )


def _detect_dip_segments(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    max_dips: int | None = None,
) -> list[tuple[float, float]]:
    """Detect dip segments from dense signal evaluation using percentile thresholding."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if len(xs) < 3 or len(ys) < 3:
        return []
    order = np.argsort(xs)
    xs_s = xs[order]
    ys_s = ys[order]
    # Find significant local maxima (near baseline) and minima (deep dips),
    # then pair each minimum with its nearest flanking maxima.
    ymax = float(np.max(ys))
    ymin = float(np.min(ys))
    # Significant maxima must be near the baseline (top 90%)
    max_threshold = ymax - 0.1 * (ymax - ymin)
    # Significant minima must be well into the dip (below 40%)
    min_threshold = ymax - 0.4 * (ymax - ymin)

    maxima_indices = []
    minima_indices = []
    for i in range(1, len(ys_s) - 1):
        if ys_s[i] > ys_s[i - 1] and ys_s[i] > ys_s[i + 1] and float(ys_s[i]) > max_threshold:
            maxima_indices.append(i)
        elif ys_s[i] < ys_s[i - 1] and ys_s[i] < ys_s[i + 1] and float(ys_s[i]) < min_threshold:
            minima_indices.append(i)

    if not minima_indices:
        return []

    if max_dips is not None and len(minima_indices) > max_dips:
        minima_indices = sorted(minima_indices, key=lambda i: float(ys_s[i]))[:max_dips]
        minima_indices.sort()

    segments: list[tuple[float, float]] = []
    for min_idx in minima_indices:
        left_max = None
        for m in reversed(maxima_indices):
            if m < min_idx:
                left_max = m
                break
        right_max = None
        for m in maxima_indices:
            if m > min_idx:
                right_max = m
                break
        if left_max is not None and right_max is not None:
            segments.append((float(xs_s[left_max]), float(xs_s[right_max])))

    # Merge overlapping or adjacent segments
    if len(segments) <= 1:
        return segments
    segments.sort(key=lambda x: x[0])
    merged: list[tuple[float, float]] = [segments[0]]
    for lo, hi in segments[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _add_dip_boundary_lines(
    fig: go.Figure,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    has_metrics: bool,
    max_dips: int | None = None,
) -> None:
    """Overlay vertical dashed lines at true-signal dip boundaries."""
    segments = _detect_dip_segments(xs, ys, max_dips=max_dips)
    if not segments:
        return
    row, col = _row_col(has_metrics)
    boundary_colors = [
        "rgba(231, 76, 60, 0.85)",  # Red
        "rgba(46, 204, 113, 0.85)",  # Green
        "rgba(155, 89, 182, 0.85)",  # Purple
    ]

    # We want annotations to appear around the middle/top
    # Let's find the maximum Y value so we can place an annotation near the top, or just use paper coords

    for i, (lo, hi) in enumerate(segments):
        color = boundary_colors[i % len(boundary_colors)]
        for xb, side in ((lo, "start"), (hi, "end")):
            fig.add_vline(
                x=xb,
                line=dict(color=color, width=2, dash="dash"),
                annotation_text=f"Dip {i + 1} {side}",
                annotation_position="top" if side == "start" else "bottom",
                annotation_font=dict(size=12, color="white"),
                annotation_bgcolor=color,
                annotation_bordercolor=color,
                annotation_borderwidth=1,
                annotation_borderpad=4,
                row=row,
                col=col,
            )

        # Add a width annotation in the middle of the dip
        width = hi - lo
        mid_x = (lo + hi) / 2.0

        # Format width: use scientific notation if very small, otherwise precision 3
        w_str = f"w={width:.2e}" if width < 0.0001 else f"w={width:.3g}"

        fig.add_annotation(
            x=mid_x,
            y=0.9,
            yref=f"y{row} domain" if row is not None and row > 1 else "y domain",
            xref=f"x{col}" if col is not None and col > 1 else "x",
            text=f"↔ {w_str}",
            showarrow=False,
            font=dict(size=11, color=color),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
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


def _compute_meas_dist_data(
    xs: np.ndarray,
    ys: np.ndarray | list[float],
    history: pl.DataFrame,
) -> dict[str, Any] | None:
    """Compute measurement distribution curve data for the lean data format."""
    if history.height == 0 or "x" not in history.columns:
        return None
    x_vals = history.get_column("x").to_list()
    x_meas = np.asarray([float(x) for x in x_vals if x is not None], dtype=float)
    if x_meas.size < 2:
        return None
    n_bins = max(60, min(400, int(np.sqrt(x_meas.size) * 20)))
    counts, edges = np.histogram(x_meas, bins=n_bins, range=(float(xs.min()), float(xs.max())))
    if counts.sum() <= 0:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts.astype(float) / max(1.0, float(counts.max()))
    y_min = float(min(ys))
    y_max = float(max(ys))
    y_span = max(1e-9, y_max - y_min)
    y_band_base = y_min + 0.02 * y_span
    y_band_height = 0.25 * y_span
    y_curve = y_band_base + density * y_band_height
    return {
        "x": centers,
        "y_baseline": float(y_band_base),
        "y_curve": y_curve,
        "customdata": density * 100.0,
    }


def _add_measurement_distribution_trace(
    fig: go.Figure,
    *,
    history: pl.DataFrame,
    xs_dense: np.ndarray,
    ys_dense: list[float],
    has_metrics: bool,
) -> None:
    """Add a hidden-by-default measurement-density overlay."""
    md = _compute_meas_dist_data(xs_dense, ys_dense, history)
    if md is None:
        return
    row, col = _row_col(has_metrics)
    fig.add_trace(
        go.Scatter(
            x=md["x"],
            y=np.full(len(md["x"]), md["y_baseline"]),
            mode="lines",
            line=dict(width=0, color="rgba(111,66,193,0)"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=md["x"],
            y=md["y_curve"],
            mode="lines",
            name="measurement distribution",
            visible="legendonly",
            line=dict(color="rgba(111,66,193,0.95)", width=1.5),
            fill="tonexty",
            fillcolor="rgba(111,66,193,0.30)",
            customdata=md["customdata"],
            hovertemplate="x=%{x}<br>relative density=%{customdata:.1f}%<extra></extra>",
        ),
        row=row,
        col=col,
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


def _dense_xs_with_measurements(scan: Any, history_xs: list[Any], *, n_dense: int = 5000) -> np.ndarray:
    xs_base = np.linspace(scan.x_min, scan.x_max, n_dense)
    if not history_xs:
        return xs_base
    history_xs_arr = np.asarray([float(x) for x in history_xs if x is not None], dtype=float)
    if history_xs_arr.size == 0:
        return xs_base
    # Only include measurement xs that fall within the physical scan domain.
    # Old cached runs may have stored normalized [0, 1] x values; exclude them.
    in_range = (history_xs_arr >= scan.x_min) & (history_xs_arr <= scan.x_max)
    history_xs_arr = history_xs_arr[in_range]
    if history_xs_arr.size == 0:
        return xs_base
    return np.unique(np.concatenate([xs_base, history_xs_arr]))


def _compute_noisy_dense_values(
    xs: np.ndarray,
    ys: np.ndarray | list[float],
    over_frequency_noise: CompositeOverFrequencyNoise,
    noise_scale: float = 1.0,
    rng: random.Random | None = None,
) -> np.ndarray:
    dense_batch = DataBatch.from_arrays(xs, ys, meta={})
    noisy_batch = over_frequency_noise.apply(dense_batch, rng if rng is not None else random.Random(0))
    vals = np.asarray(noisy_batch.signal_values, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    bad = ~np.isfinite(vals)
    if noise_scale != 1.0:
        vals = ys_arr + (vals - ys_arr) * noise_scale
    vals[bad] = ys_arr[bad]
    return vals


def _compute_noisy_dense_band(
    xs: np.ndarray,
    ys: np.ndarray | list[float],
    over_frequency_noise: CompositeOverFrequencyNoise,
    noise_scale: float = 1.0,
    n_draws: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Monte-Carlo envelope of where a re-measurement could plausibly land.

    Draws ``n_draws`` independent noise realizations over the dense true-signal
    curve and returns two nested percentile bands per x -- an inner ~1 sigma
    band (15.87/84.13 pct) and an outer ~2 sigma band (2.28/97.72 pct), so a
    single outlier draw doesn't dominate the width.
    """
    seed_rng = random.Random(seed)
    draws = np.empty((n_draws, len(xs)), dtype=float)
    for i in range(n_draws):
        draws[i] = _compute_noisy_dense_values(
            xs, ys, over_frequency_noise, noise_scale, rng=random.Random(seed_rng.randint(0, 2**31 - 1))
        )
    lo1 = np.percentile(draws, 15.87, axis=0)
    hi1 = np.percentile(draws, 84.13, axis=0)
    lo2 = np.percentile(draws, 2.28, axis=0)
    hi2 = np.percentile(draws, 97.72, axis=0)
    return lo1, hi1, lo2, hi2


def _splice_noisy_dense_at_measurements(
    xs: np.ndarray,
    noisy_vals: list[float],
    history_xs: list[Any],
    history_ys: list[Any],
) -> None:
    """In-place splice noisy dense curve so it passes through measurement points."""
    # Bypassed splicing to prevent wrong noise / spurious spikes artifact
    # in physical experiments with drift and coordinate/time dependencies.
    return


def _measurements_from_history(history: pl.DataFrame) -> dict[str, Any]:
    if history.height == 0:
        return {"mode": "empty"}

    if "phase" in history.columns:
        phases = history.get_column("phase").to_list()
        xs_s = history.get_column("x").to_list() if "x" in history.columns else []
        ys_s = history.get_column("signal_values").to_list() if "signal_values" in history.columns else []
        coarse_x: list[float] = []
        coarse_y: list[float | None] = []
        secondary_x: list[float] = []
        secondary_y: list[float | None] = []
        tertiary_x: list[float] = []
        tertiary_y: list[float | None] = []
        fine_x: list[float] = []
        fine_y: list[float | None] = []
        fine_step: list[int] = []
        for x, y, p in zip(xs_s, ys_s, phases, strict=False):
            if p == "coarse":
                coarse_x.append(float(x))
                coarse_y.append(_json_safe_float(y))
            elif p == "secondary":
                secondary_x.append(float(x))
                secondary_y.append(_json_safe_float(y))
            elif p == "tertiary":
                tertiary_x.append(float(x))
                tertiary_y.append(_json_safe_float(y))
            elif p == "fine":
                fine_x.append(float(x))
                fine_y.append(_json_safe_float(y))
                fine_step.append(len(fine_step))
        return {
            "mode": "phases",
            "coarse_x": coarse_x,
            "coarse_y": coarse_y,
            "secondary_x": secondary_x,
            "secondary_y": secondary_y,
            "tertiary_x": tertiary_x,
            "tertiary_y": tertiary_y,
            "fine_x": fine_x,
            "fine_y": fine_y,
            "fine_step": fine_step,
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
    focus_window: tuple[float, float] | None = None,
    mode_estimates: Mapping[str, float] | None = None,
    belief_unit_cube: UnitCubeSignalModel | None = None,
    narrowed_param_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Dense curve + measurement points for static UI head-to-head (matches ``plot_scan_measurements``).

    Unlike the internal gz-writer path, this public helper keeps a plain-JSON
    contract: all arrays are converted to Python lists before returning.
    """
    history_xs_s, history_ys_s = _extract_history_xy(history)
    xs = _dense_xs_with_measurements(scan, history_xs_s)
    ys = _true_signal_dense_y(scan, xs)
    out: dict[str, Any] = {
        "x_dense": xs.tolist(),
        "y_dense": ys.tolist(),
    }
    if mode_estimates:
        y_mode = _mode_belief_dense_y(scan, xs, mode_estimates, belief_unit_cube=belief_unit_cube)
        if y_mode is not None and len(y_mode) == len(xs):
            out["y_dense_mode"] = np.asarray(y_mode).tolist()
    if over_frequency_noise is not None:
        noise_scale = _noise_scale_for_scan(scan, over_frequency_noise)
        noisy_vals = _compute_noisy_dense_values(xs, ys, over_frequency_noise, noise_scale)
        _splice_noisy_dense_at_measurements(xs, noisy_vals, history_xs_s, history_ys_s)
        out["y_dense_noisy"] = noisy_vals.tolist()

    has_metrics = history.height > 0 and any(col in history.columns for col in ("entropy", "max_prob", "uncertainty"))
    out["has_metrics"] = has_metrics
    out["measurements"] = _measurements_from_history(history)
    if focus_window is not None:
        lo, hi = float(focus_window[0]), float(focus_window[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out["focus_window"] = [lo, hi]
    if narrowed_param_bounds:
        safe_bounds: dict[str, list[float]] = {}
        for name, (lo, hi) in narrowed_param_bounds.items():
            flo, fhi = _json_safe_float(lo), _json_safe_float(hi)
            if flo is not None and fhi is not None and fhi > flo:
                safe_bounds[name] = [flo, fhi]
        if safe_bounds:
            out["narrowed_param_bounds"] = safe_bounds
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


def _decode_plotly_array(v: Any) -> list[Any] | None:
    if v is None:
        return None
    if hasattr(v, "to_plotly_json"):
        v = v.to_plotly_json()
    if isinstance(v, dict) and "bdata" in v and "dtype" in v:
        import base64

        import numpy as np

        try:
            raw = base64.b64decode(v["bdata"])
            arr = np.frombuffer(raw, dtype=v["dtype"])
            return arr.tolist()
        except Exception:
            pass
    return v


def _trace_xy_lists(tr: Any) -> tuple[list[float], list[float | None]]:
    xs: list[float] = []
    ys: list[float | None] = []
    tx = getattr(tr, "x", None)
    ty = getattr(tr, "y", None)
    tx = _decode_plotly_array(tx)
    ty = _decode_plotly_array(ty)
    if tx is not None:
        for v in tx:
            xs.append(float(v))
    if ty is not None:
        for v in ty:
            ys.append(_json_safe_float(v))
    n = min(len(xs), len(ys))
    return xs[:n], ys[:n]


def plot_data_from_scan_figure(fig: go.Figure) -> dict[str, Any] | None:
    """Rebuild ``plot_data`` from a scan figure (must match ``plot_scan_measurements``)."""
    x_dense: list[float] | None = None
    y_dense: list[float] | None = None
    y_dense_noisy: list[float] | None = None
    coarse_x: list[float] = []
    coarse_y: list[float | None] = []
    fine_x: list[float] = []
    fine_y: list[float | None] = []
    fine_step: list[float] = []
    step_x: list[float] = []
    step_y: list[float | None] = []
    step_idx: list[float] = []

    has_metrics = any(getattr(t, "name", None) in ("Entropy", "Uncertainty") for t in fig.data)

    y_dense_mode: list[float] | None = None
    y_dense_sobol_mode: list[float] | None = None
    sobol_x: list[float] = []
    sobol_y: list[float | None] = []
    for tr in fig.data:
        name = getattr(tr, "name", None) or ""
        mode = getattr(tr, "mode", "") or ""
        if name in ("locator most likely signal", "locator mode belief signal") and "lines" in mode:
            _, ym = _trace_xy_lists(tr)
            if ym and all(v is not None for v in ym):
                y_dense_mode = [float(v) for v in ym if v is not None]
            continue
        if name == "sobol most likely signal" and "lines" in mode:
            _, ym = _trace_xy_lists(tr)
            if ym and all(v is not None for v in ym):
                y_dense_sobol_mode = [float(v) for v in ym if v is not None]
            continue
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
            mk = getattr(tr, "marker", None)
            if mk is not None:
                c = getattr(mk, "color", None)
                c = _decode_plotly_array(c)
                if c is not None and hasattr(c, "__iter__") and not isinstance(c, str | bytes):
                    fine_step = [float(v) for v in c]
            continue
        if name == "measurements (noisy)":
            step_x, step_y = _trace_xy_lists(tr)
            mk = getattr(tr, "marker", None)
            if mk is not None:
                c = getattr(mk, "color", None)
                c = _decode_plotly_array(c)
                if c is not None and hasattr(c, "__iter__") and not isinstance(c, str | bytes):
                    step_idx = [float(v) for v in c]
            continue
        if name == "sobol measurements (noisy)":
            sobol_x, sobol_y = _trace_xy_lists(tr)
            continue

    if x_dense is None or y_dense is None:
        return None

    out: dict[str, Any] = {
        "x_dense": [float(x) for x in x_dense],
        "y_dense": [float(y) for y in y_dense if y is not None],
        "has_metrics": has_metrics,
    }
    if y_dense_mode is not None and len(y_dense_mode) == len(out["x_dense"]):
        out["y_dense_mode"] = y_dense_mode
    if y_dense_sobol_mode is not None and len(y_dense_sobol_mode) == len(out["x_dense"]):
        out["y_dense_sobol_mode"] = y_dense_sobol_mode
    if y_dense_noisy is not None and len(y_dense_noisy) == len(out["x_dense"]):
        out["y_dense_noisy"] = y_dense_noisy

    if sobol_x:
        out["sobol_measurements"] = {
            "x": sobol_x,
            "y": [float(y) for y in sobol_y if y is not None],
        }

    if coarse_x or fine_x:
        if len(fine_step) != len(fine_x):
            fine_step = [float(i) for i in range(len(fine_x))]
        out["measurements"] = {
            "mode": "phases",
            "coarse_x": coarse_x,
            "coarse_y": coarse_y,
            "fine_x": fine_x,
            "fine_y": fine_y,
            "fine_step": [int(s) for s in fine_step],
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
    """If a scan manifest entry has no ``plot_data``, rebuild it from the saved scan file on disk."""
    if entry.get("type") != "scan" or entry.get("plot_data"):
        return
    rel = entry.get("path")
    if not isinstance(rel, str) or not rel.strip():
        return
    path = out_dir / rel
    if not path.exists():
        return
    try:
        if rel.endswith(".json.gz"):
            from nvision.viz._f32_json import from_gz_bytes

            raw = from_gz_bytes(path.read_bytes())
            if isinstance(raw, dict) and raw.get("_graph_type") == "scan":
                # New lean format — plot_data is a subset of the lean data
                plot_data: dict[str, Any] = {
                    k: raw[k] for k in ("x_dense", "y_dense", "has_metrics", "measurements") if k in raw
                }
                for k in (
                    "focus_window",
                    "narrowed_param_bounds",
                    "y_dense_noisy",
                    "y_dense_noisy_lo",
                    "y_dense_noisy_hi",
                    "y_dense_mode",
                ):
                    if k in raw:
                        plot_data[k] = raw[k]
                entry["plot_data"] = plot_data
                return
            # Legacy full Plotly figure JSON
            from nvision.viz._f32_json import figure_from_gz_bytes

            fig = figure_from_gz_bytes(path.read_bytes())
        else:
            html = path.read_text(encoding="utf-8")
            fig = _parse_figure_from_scan_html(html)
            if fig is None:
                return
        plot_data = plot_data_from_scan_figure(fig)
    except Exception as exc:
        logging.warning("Failed to backfill scan plot data for %s: %s", rel, exc)
        return
    if plot_data:
        entry["plot_data"] = plot_data


def _setup_scan_layout(
    fig: go.Figure,
    has_metrics: bool,
    *,
    focus_window: tuple[float, float] | None = None,
    narrowed_param_bounds: dict[str, tuple[float, float]] | None = None,
    per_dip_windows: list[tuple[float, float]] | None = None,
    true_params: dict | None = None,
) -> None:
    # Note: focus_window is used for overlay only, not for setting x-axis range
    # The plot should show the full scan range (scan.x_min to scan.x_max)
    layout_args: dict = dict(
        title="Scan with sampled measurements",
        template="plotly_white",
    )
    if has_metrics:
        layout_args.update(
            dict(
                xaxis_title="frequency",
                xaxis_tickformat="~s",
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
                xaxis_tickformat="~s",
                yaxis_title="intensity (photon count)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0.0,
                ),
                margin=dict(t=110),
            )
        )
    fig.update_layout(**layout_args)

    # Embed narrowed_param_bounds and per_dip_windows in figure meta so the UI can retrieve them
    # without needing a separate manifest field or extra HTTP request.
    meta_dict: dict[str, object] = {}
    if narrowed_param_bounds:
        safe_meta: dict[str, list[float]] = {}
        for name, (lo, hi) in narrowed_param_bounds.items():
            flo, fhi = _json_safe_float(lo), _json_safe_float(hi)
            if flo is not None and fhi is not None and fhi > flo:
                safe_meta[name] = [flo, fhi]
        if safe_meta:
            meta_dict["narrowed_param_bounds"] = safe_meta
    if per_dip_windows:
        safe_windows: list[list[float]] = []
        for lo, hi in per_dip_windows:
            flo, fhi = _json_safe_float(lo), _json_safe_float(hi)
            if flo is not None and fhi is not None and fhi > flo:
                safe_windows.append([flo, fhi])
        if safe_windows:
            meta_dict["per_dip_windows"] = safe_windows
    if true_params and isinstance(true_params, dict):
        meta_dict["true_params"] = true_params
    if meta_dict:
        fig.update_layout(meta=meta_dict)


def _compute_scan_data_dict(
    scan: Any,
    history: pl.DataFrame,
    over_frequency_noise: CompositeOverFrequencyNoise | None,
    focus_window: tuple[float, float] | None = None,
    mode_estimates: Mapping[str, float] | None = None,
    belief_unit_cube: UnitCubeSignalModel | None = None,
    narrowed_param_bounds: dict[str, tuple[float, float]] | None = None,
    per_dip_windows: list[tuple[float, float]] | None = None,
    sobol_xs: list[float] | None = None,
    sobol_ys: list[float] | None = None,
    sobol_mode_estimates: Mapping[str, float] | None = None,
    sweep_xs: list[float] | None = None,
    sweep_ys: list[float] | None = None,
    sweep_mode_estimates: Mapping[str, float] | None = None,
    true_params: dict | None = None,
) -> dict[str, Any]:
    """Build the lean scan data dict written to disk (replaces the full Plotly figure)."""
    history_xs_raw, _history_ys_raw = _extract_history_xy(history)
    xs = _dense_xs_with_measurements(scan, history_xs_raw, n_dense=5000)
    ys = _true_signal_dense_y(scan, xs)

    out: dict[str, Any] = {
        "_graph_type": "scan",
        "x_dense": xs,
        "y_dense": ys,
    }

    if mode_estimates:
        y_mode = _mode_belief_dense_y(scan, xs, mode_estimates, belief_unit_cube=belief_unit_cube)
        if y_mode is not None and len(y_mode) == len(xs):
            out["y_dense_mode"] = y_mode

    if over_frequency_noise is not None:
        noise_scale = _noise_scale_for_scan(scan, over_frequency_noise)
        noisy_lo1, noisy_hi1, noisy_lo2, noisy_hi2 = _compute_noisy_dense_band(
            xs, ys, over_frequency_noise, noise_scale
        )
        out["y_dense_noisy_lo"] = noisy_lo1
        out["y_dense_noisy_hi"] = noisy_hi1
        out["y_dense_noisy_lo2"] = noisy_lo2
        out["y_dense_noisy_hi2"] = noisy_hi2

    has_metrics = history.height > 0 and any(col in history.columns for col in ("entropy", "max_prob", "uncertainty"))
    out["has_metrics"] = has_metrics
    out["measurements"] = _measurements_from_history(history)

    meas_dist = _compute_meas_dist_data(xs, ys, history)
    if meas_dist is not None:
        out["meas_dist"] = meas_dist

    if focus_window is not None:
        lo, hi = float(focus_window[0]), float(focus_window[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out["focus_window"] = [lo, hi]

    # Filter per-dip windows to expected dip count (same guard as old plotting code)
    if per_dip_windows:
        expected_dips: int | None = None
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            expected_dips = int(scan.true_signal.model.expected_dip_count())
        filtered = per_dip_windows
        if expected_dips is not None and len(per_dip_windows) > expected_dips:
            filtered = None
        if filtered:
            safe_windows: list[list[float]] = []
            for lo, hi in filtered:
                flo, fhi = _json_safe_float(lo), _json_safe_float(hi)
                if flo is not None and fhi is not None and fhi > flo:
                    safe_windows.append([flo, fhi])
            if safe_windows:
                out["per_dip_windows"] = safe_windows

    if narrowed_param_bounds:
        safe_bounds: dict[str, list[float]] = {}
        for name, (lo, hi) in narrowed_param_bounds.items():
            flo, fhi = _json_safe_float(lo), _json_safe_float(hi)
            if flo is not None and fhi is not None and fhi > flo:
                safe_bounds[name] = [flo, fhi]
        if safe_bounds:
            out["narrowed_param_bounds"] = safe_bounds

    if true_params and isinstance(true_params, dict):
        out["true_params"] = true_params

    if sobol_xs and sobol_ys:
        width = float(scan.x_max - scan.x_min)
        out["sobol_measurements"] = {
            "x": float(scan.x_min) + np.asarray(sobol_xs, dtype=float) * width,
            "y": np.asarray(sobol_ys, dtype=float),
        }
    if sobol_mode_estimates:
        y_sobol_mode = _mode_belief_dense_y(scan, xs, sobol_mode_estimates, belief_unit_cube=belief_unit_cube)
        if y_sobol_mode is not None and len(y_sobol_mode) > 0:
            out["sobol_mode_y"] = y_sobol_mode

    if sweep_xs and sweep_ys:
        width = float(scan.x_max - scan.x_min)
        out["sweep_measurements"] = {
            "x": float(scan.x_min) + np.asarray(sweep_xs, dtype=float) * width,
            "y": np.asarray(sweep_ys, dtype=float),
        }
    if sweep_mode_estimates:
        y_sweep_mode = _mode_belief_dense_y(scan, xs, sweep_mode_estimates, belief_unit_cube=belief_unit_cube)
        if y_sweep_mode is not None and len(y_sweep_mode) > 0:
            out["sweep_mode_y"] = y_sweep_mode

    if has_metrics:
        if "entropy" in history.columns:
            entropy = history.get_column("entropy").to_list()
            if any(x is not None and not np.isnan(float(x)) for x in entropy if x is not None):
                out["entropy"] = [_json_safe_float(x) for x in entropy]
        if "uncertainty" in history.columns:
            uncert = history.get_column("uncertainty").to_list()
            if any(x is not None and not np.isnan(float(x)) for x in uncert if x is not None):
                out["uncertainty"] = [_json_safe_float(x) for x in uncert]

    return out


class MeasurementsMixin:
    """Mixin for scan measurement plotting."""

    # Typing for mixin dependency (self.out_dir from VizBase)
    out_dir: Path

    def plot_scan_measurements(
        self,
        scan,
        history: pl.DataFrame,
        out_path: Path | None = None,
        over_frequency_noise: CompositeOverFrequencyNoise | None = None,
        mode_estimates: Mapping[str, float] | None = None,
        focus_window: tuple[float, float] | None = None,
        per_dip_windows: list[tuple[float, float]] | None = None,
        belief_unit_cube: UnitCubeSignalModel | None = None,
        narrowed_param_bounds: dict[str, tuple[float, float]] | None = None,
        sobol_xs: list[float] | None = None,
        sobol_ys: list[float] | None = None,
        sobol_mode_estimates: Mapping[str, float] | None = None,
        sweep_xs: list[float] | None = None,
        sweep_ys: list[float] | None = None,
        sweep_mode_estimates: Mapping[str, float] | None = None,
        true_params: dict | None = None,
    ) -> bytes:
        """Serialize scan data as a lean JSON.gz (definition lives in static/graphs/scan.json)."""
        if out_path is not None:
            ensure_out_dir(out_path.parent)

        data = _compute_scan_data_dict(
            scan,
            history,
            over_frequency_noise,
            focus_window=focus_window,
            mode_estimates=mode_estimates,
            belief_unit_cube=belief_unit_cube,
            narrowed_param_bounds=narrowed_param_bounds,
            per_dip_windows=per_dip_windows,
            sobol_xs=sobol_xs,
            sobol_ys=sobol_ys,
            sobol_mode_estimates=sobol_mode_estimates,
            sweep_xs=sweep_xs,
            sweep_ys=sweep_ys,
            sweep_mode_estimates=sweep_mode_estimates,
            true_params=true_params,
        )

        from nvision.viz._f32_json import dump_gz

        return dump_gz(data, out_path)
