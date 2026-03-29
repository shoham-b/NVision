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

    def plot_posterior_animation(
        self,
        posterior_history: list[np.ndarray],
        freq_grid: np.ndarray,
        out_path: Path,
        model_history: list[np.ndarray] | None = None,
        *,
        true_value: float | None = None,
    ) -> None:
        """Create an interactive Plotly animation of the posterior distribution evolution.

        If ``true_value`` is set, draws a vertical line at the ground-truth parameter value.
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

        _add_true_vline_single_axis(fig, true_value)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path)

    def plot_posterior_animation_all_params(
        self,
        posterior_inputs_by_param: dict[str, tuple[list[np.ndarray], np.ndarray]],
        out_path: Path,
        *,
        true_params: Mapping[str, float] | None = None,
    ) -> None:
        """Animate marginal posterior evolution for every parameter (one subplot each, own x-axis).

        ``true_params`` maps parameter names to :class:`~nvision.signal.signal.TrueSignal` values
        (physical units); a dashed vertical line is drawn on each subplot when a value is given.
        """
        if not posterior_inputs_by_param:
            return

        param_names = sorted(posterior_inputs_by_param.keys())
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
        for si, step_idx in enumerate(step_indices):
            frames.append(
                go.Frame(
                    data=traces_for_step(step_idx),
                    name=str(si),
                    layout=go.Layout(
                        title_text=f"Posterior evolution (all parameters) — step {step_idx + 1}/{total_steps}",
                    ),
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
                "label": str(step_indices[si] + 1),
                "method": "animate",
            }
            for si, frame in enumerate(frames)
        ]

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
        keys = sorted({k for step in parameter_history for k in step})
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
