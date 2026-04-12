from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _add_true_vline_single_axis(fig: go.Figure, true_value: float | None) -> None:
    if true_value is None or not math.isfinite(float(true_value)):
        return
    tv = float(true_value)
    fig.add_vline(
        x=tv,
        line_width=2,
        line_dash="dash",
        line_color="green",
        annotation_text=f"true: {tv:.6g}",
        annotation_position="top",
    )


def _add_acquisition_window_subplot(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    window: tuple[float, float],
    full_domain: tuple[float, float] | None,
    annotation_text: str = "acquisition window",
) -> None:
    """Shade the acquisition interval on one marginal."""
    x0, x1 = float(window[0]), float(window[1])
    if not math.isfinite(x0) or not math.isfinite(x1) or x1 <= x0:
        return
    fig.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor="rgba(46, 204, 113, 0.18)",
        line_width=1,
        line_color="rgba(46, 204, 113, 0.75)",
        layer="below",
        annotation_text=annotation_text,
        annotation_position="top left",
        row=row,
        col=col,
    )
    if full_domain is not None:
        flo, fhi = float(full_domain[0]), float(full_domain[1])
        if math.isfinite(flo) and math.isfinite(fhi) and fhi > flo:
            lo = min(flo, x0)
            hi = max(fhi, x1)
            fig.update_xaxes(range=[lo, hi], row=row, col=col)


def _add_acquisition_window_single(
    fig: go.Figure,
    window: tuple[float, float],
    full_domain: tuple[float, float] | None,
) -> None:
    x0, x1 = float(window[0]), float(window[1])
    if not math.isfinite(x0) or not math.isfinite(x1) or x1 <= x0:
        return
    fig.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor="rgba(46, 204, 113, 0.18)",
        line_width=1,
        line_color="rgba(46, 204, 113, 0.75)",
        layer="below",
        annotation_text="post-sweep acquisition",
        annotation_position="top left",
    )
    if full_domain is not None:
        flo, fhi = float(full_domain[0]), float(full_domain[1])
        if math.isfinite(flo) and math.isfinite(fhi) and fhi > flo:
            lo = min(flo, x0)
            hi = max(fhi, x1)
            fig.update_xaxes(range=[lo, hi])


def _add_true_vline_subplots(
    fig: go.Figure,
    param_names: list[str],
    true_params: Mapping[str, float] | None,
) -> None:
    if not true_params:
        return
    for i, name in enumerate(param_names, start=1):
        raw = true_params.get(name)
        if raw is None:
            continue
        tv = float(raw)
        if not math.isfinite(tv):
            continue
        fig.add_vline(
            x=tv,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text=f"true: {tv:.6g}",
            annotation_position="top right",
            row=i,
            col=1,
        )


def _trace_one_marginal_posterior(
    posterior: np.ndarray,
    grid: np.ndarray,
    param: str,
) -> go.Histogram | go.Scatter:
    if posterior.ndim == 2:
        return go.Histogram(
            x=posterior[:, 0],
            histnorm="probability density",
            name=f"{param} (particles)",
            opacity=0.75,
            nbinsx=40,
            showlegend=False,
        )
    return go.Scatter(
        x=grid,
        y=posterior,
        mode="lines",
        name=param,
        line=dict(color="blue"),
        showlegend=False,
    )


class BayesianMixin:
    """Mixin for Bayesian visualization."""

    # Typing for mixin dependency
    out_dir: Path

    def plot_posterior_animation(  # noqa: C901
        self,
        posterior_history: list[np.ndarray],
        freq_grid: np.ndarray,
        out_path: Path,
        model_history: list[np.ndarray] | None = None,
        *,
        true_value: float | None = None,
        acquisition_window: tuple[float, float] | None = None,
        experiment_domain: tuple[float, float] | None = None,
    ) -> None:
        """Create an interactive Plotly animation of the posterior distribution evolution.

        If ``true_value`` is set, draws a vertical line at the ground-truth parameter value.
        If ``acquisition_window`` is set, shades the post-sweep search interval (optionally
        with ``experiment_domain`` widening the x-axis to the full sweep range).
        """
        if not posterior_history:
            return

        is_particles = posterior_history[0].ndim == 2

        # Prepare frames
        frames = []
        # Downsample frames if too many for performance (e.g., max 100 frames)
        total_steps = len(posterior_history)
        step_indices = range(total_steps)
        if total_steps > 100:
            step_indices = np.linspace(0, total_steps - 1, 100, dtype=int)

        max_prob = 0.0
        if not is_particles:
            for p in posterior_history:
                m = np.max(p)
                if m > max_prob:
                    max_prob = m

        for i in step_indices:
            posterior = posterior_history[i]

            if is_particles:
                data = [
                    go.Histogram(
                        x=posterior[:, 0],
                        histnorm="probability density",
                        name="Posterior Particles",
                        marker_color="blue",
                        opacity=0.7,
                        nbinsx=50,
                    )
                ]
            else:
                data = [
                    go.Scatter(
                        x=freq_grid,
                        y=posterior,
                        mode="lines",
                        name="Posterior",
                        line=dict(color="blue"),
                    )
                ]

            if model_history and i < len(model_history):
                # Normalize model for visualization scale if needed, or plotting separately?
                # Usually model is in signal units, posterior in probability density.
                # Put model on secondary y-axis?
                # For simplicity, let's just plot posterior for now or scale model.
                pass

            frames.append(
                go.Frame(
                    data=data,
                    name=str(i),
                    layout=go.Layout(title_text=f"Step {i + 1}/{total_steps}"),
                )
            )

        slider_steps = [
            {
                "args": [
                    [frame.name],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
                "label": str(int(frame.name) + 1),
                "method": "animate",
            }
            for frame in frames
        ]

        # Initial plot
        initial_posterior = posterior_history[0]
        if is_particles:
            initial_data = [
                go.Histogram(
                    x=initial_posterior[:, 0],
                    histnorm="probability density",
                    name="Posterior Particles",
                    marker_color="blue",
                    opacity=0.7,
                    nbinsx=50,
                )
            ]
            yaxis_layout = dict(title="Probability Density")
        else:
            initial_data = [
                go.Scatter(
                    x=freq_grid,
                    y=initial_posterior,
                    mode="lines",
                    name="Posterior",
                    line=dict(color="blue"),
                )
            ]
            yaxis_layout = dict(title="Probability Density", range=[0, max_prob * 1.1])

        fig = go.Figure(
            data=initial_data,
            layout=go.Layout(
                xaxis=dict(title="Frequency / Parameter"),
                yaxis=yaxis_layout,
                title="Posterior Evolution",
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        x=0.0,
                        y=1.15,
                        xanchor="left",
                        yanchor="top",
                        showactive=False,
                        pad={"r": 10, "t": 0},
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": 100, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                        ],
                    )
                ],
                sliders=[
                    dict(
                        active=0,
                        pad={"t": 50, "b": 10},
                        currentvalue={"prefix": "Step: "},
                        steps=slider_steps,
                    )
                ],
            ),
            frames=frames,
        )

        if acquisition_window is not None:
            _add_acquisition_window_single(fig, acquisition_window, experiment_domain)

        _add_true_vline_single_axis(fig, true_value)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path)

    def plot_posterior_animation_all_params(
        self,
        posterior_inputs_by_param: dict[str, tuple[list[np.ndarray], np.ndarray]],
        out_path: Path,
        *,
        true_params: Mapping[str, float] | None = None,
        acquisition_window: tuple[float, float] | None = None,
        acquisition_param: str | None = None,
        experiment_domain: tuple[float, float] | None = None,
        narrowed_param_bounds: dict[str, tuple[float, float]] | None = None,
        per_step_narrowed_bounds: list[dict[str, tuple[float, float]]] | None = None,
    ) -> None:
        """Animate marginal posterior evolution for every parameter (one subplot each, own x-axis).

        ``true_params`` maps parameter names to :class:`~nvision.spectra.signal.TrueSignal` values
        (physical units); a dashed vertical line is drawn on each subplot when a value is given.

        When ``acquisition_window`` and ``acquisition_param`` are set (post-sweep focus from the
        Bayesian locator), a green band marks that interval on the corresponding marginal; if
        ``experiment_domain`` is the full sweep ``(x_min, x_max)``, the x-axis is expanded so the
        band appears in context of the original domain.

        When ``narrowed_param_bounds`` is provided, additional green bands are drawn on each
        non-scan marginal that was narrowed by the sweep estimator, giving a visual cue of
        which parameter regions survived the coarse sweep.
        """
        if not posterior_inputs_by_param:
            return

        param_names = list(posterior_inputs_by_param.keys())
        first_param = param_names[0]
        total_steps = len(posterior_inputs_by_param[first_param][0])
        if total_steps == 0:
            return

        step_indices = list(range(total_steps))
        if total_steps > 100:
            step_indices = [int(x) for x in np.linspace(0, total_steps - 1, 100)]

        n = len(param_names)

        def traces_for_step(step_idx: int) -> list[object]:
            traces: list[object] = []
            for param in param_names:
                posterior_history, grid = posterior_inputs_by_param[param]
                posterior = posterior_history[-1] if step_idx >= len(posterior_history) else posterior_history[step_idx]
                traces.append(_trace_one_marginal_posterior(posterior, grid, param))
            return traces

        fig = make_subplots(
            rows=n,
            cols=1,
            subplot_titles=tuple(param_names),
            vertical_spacing=min(0.12, 0.5 / max(n, 1)),
        )
        for i, tr in enumerate(traces_for_step(step_indices[0]), start=1):
            fig.add_trace(tr, row=i, col=1)
        for i, name in enumerate(param_names, start=1):
            fig.update_yaxes(title_text="density", row=i, col=1)
            fig.update_xaxes(title_text=name, row=i, col=1)

        frames = []
        refocusing_steps = set()  # Track steps where refocusing occurs
        
        for si, step_idx in enumerate(step_indices):
            # Get narrowed bounds for this step if available
            step_bounds = None
            if per_step_narrowed_bounds is not None and step_idx < len(per_step_narrowed_bounds):
                step_bounds = per_step_narrowed_bounds[step_idx]
            elif narrowed_param_bounds is not None:
                # Fallback to final bounds if per-step not available
                step_bounds = narrowed_param_bounds

            frame_data = traces_for_step(step_idx)
            
            # Add acquisition window overlays for this step to frame data
            if step_bounds and acquisition_param and acquisition_param in step_bounds:
                window = step_bounds[acquisition_param]
                if acquisition_param in posterior_inputs_by_param:
                    row_nb = param_names.index(acquisition_param) + 1
                    # Create a vrect shape for this frame
                    x0, x1 = float(window[0]), float(window[1])
                    if math.isfinite(x0) and math.isfinite(x1) and x1 > x0:
                        # Determine annotation text based on step
                        if step_idx < 32:  # Initial sweep steps
                            annotation = "post-sweep acquisition"
                        else:
                            annotation = "posterior refocusing"
                        
                        frame_data.append(
                            go.Scatter(
                                x=[x0, x1, x1, x0, x0],
                                y=[0, 0, 1, 1, 0],
                                fill="toself",
                                fillcolor="rgba(46, 204, 113, 0.18)",
                                line=dict(color="rgba(46, 204, 113, 0.75)", width=1),
                                hoverinfo="skip",
                                showlegend=False,
                                xaxis=f"x{row_nb}" if row_nb > 1 else "x",
                                yaxis=f"y{row_nb}" if row_nb > 1 else "y",
                                name=annotation,  # Store annotation in name for hover
                            )
                        )
                        
                        # Track refocusing events (after initial sweep, every 20 steps)
                        if step_idx >= 32 and (step_idx - 32) % 20 == 0:
                            refocusing_steps.add(si)
            
            # If no per-step bounds but we have acquisition window, show it as post-sweep
            elif not step_bounds and acquisition_window is not None and acquisition_param and acquisition_param in posterior_inputs_by_param:
                if step_idx >= 32:  # Only show after sweep is complete
                    row_nb = param_names.index(acquisition_param) + 1
                    x0, x1 = float(acquisition_window[0]), float(acquisition_window[1])
                    if math.isfinite(x0) and math.isfinite(x1) and x1 > x0:
                        frame_data.append(
                            go.Scatter(
                                x=[x0, x1, x1, x0, x0],
                                y=[0, 0, 1, 1, 0],
                                fill="toself",
                                fillcolor="rgba(46, 204, 113, 0.18)",
                                line=dict(color="rgba(46, 204, 113, 0.75)", width=1),
                                hoverinfo="skip",
                                showlegend=False,
                                xaxis=f"x{row_nb}" if row_nb > 1 else "x",
                                yaxis=f"y{row_nb}" if row_nb > 1 else "y",
                                name="post-sweep acquisition",
                            )
                        )
                        
                        # Manually track refocusing events every 20 steps after initial sweep
                        if step_idx >= 32 and (step_idx - 32) % 20 == 0:
                            refocusing_steps.add(si)

            # Add refocusing indicator to title
            title_text = f"Posterior evolution (all parameters) — step {step_idx + 1}/{total_steps}"
            if si in refocusing_steps:
                title_text += " [REFOCUSING]"
                
            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(si),
                    layout=go.Layout(
                        title_text=title_text,
                    ),
                )
            )

        slider_steps = []
        for si, frame in enumerate(frames):
            step_num = step_indices[si] + 1
            # Highlight refocusing steps with brackets
            if si in refocusing_steps:
                label = f"[{step_num}]"  # Brackets mark refocusing steps
            else:
                label = str(step_num)
                
            slider_steps.append(
                {
                    "args": [
                        [frame.name],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": label,
                    "method": "animate",
                }
            )

        # Initial acquisition window for frequency (will be overridden per-step)
        if acquisition_window is not None and acquisition_param and acquisition_param in posterior_inputs_by_param:
            row_ap = param_names.index(acquisition_param) + 1
            _add_acquisition_window_subplot(
                fig,
                row=row_ap,
                col=1,
                window=acquisition_window,
                full_domain=experiment_domain,
                annotation_text="post-sweep acquisition",
            )

        _add_true_vline_subplots(fig, param_names, true_params)

        fig.update_layout(
            title_text=f"Posterior evolution (all parameters) — step {step_indices[0] + 1}/{total_steps}",
            template="plotly_white",
            height=max(260, 180 * n),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.0,
                    y=1.08,
                    xanchor="left",
                    yanchor="top",
                    showactive=False,
                    pad={"r": 10, "t": 0},
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    pad={"t": 40, "b": 10},
                    currentvalue={"prefix": "Measurement step: "},
                    steps=slider_steps,
                )
            ],
        )
        fig.frames = frames

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path)

    def plot_parameter_convergence(
        self,
        parameter_history: list[dict[str, float]],
        out_path: Path,
    ) -> None:
        """Plot uncertainty trajectories over steps for Bayesian parameters."""
        if not parameter_history:
            return

        # Use all uncertainty keys provided by the belief implementation.
        keys = []
        for step in parameter_history:
            for k in step:
                if k not in keys:
                    keys.append(k)
        if not keys:
            return

        steps = list(range(len(parameter_history)))

        raw_y: list[list[float]] = []
        norm_y: list[list[float]] = []
        for key in keys:
            values = [step.get(key, math.nan) for step in parameter_history]
            raw_y.append(values)
            v0 = next((v for v in values if not math.isnan(v)), math.nan)
            if not math.isnan(v0) and v0 != 0:
                norm_y.append([v / v0 if not math.isnan(v) else math.nan for v in values])
            else:
                norm_y.append([math.nan] * len(values))

        fig = go.Figure()

        for key, y in zip(keys, raw_y, strict=True):
            fig.add_trace(go.Scatter(x=steps, y=y, mode="lines", name=key))

        fig.update_layout(
            title="Parameter Uncertainty Convergence",
            xaxis_title="Step",
            yaxis_title="Uncertainty (std dev)",
            template="plotly_white",
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    pad={"r": 8, "t": 8},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.12,
                    yanchor="bottom",
                    buttons=[
                        dict(
                            label="Absolute",
                            method="update",
                            args=[
                                {"y": raw_y},
                                {
                                    "yaxis.title.text": "Uncertainty (std dev)",
                                },
                            ],
                        ),
                        dict(
                            label="Normalized (÷ initial)",
                            method="update",
                            args=[
                                {"y": norm_y},
                                {
                                    "yaxis.title.text": "Relative uncertainty (u / u₀)",
                                },
                            ],
                        ),
                    ],
                )
            ],
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path)
