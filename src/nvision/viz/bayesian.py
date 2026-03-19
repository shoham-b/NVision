from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


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
    ) -> None:
        """Create an interactive Plotly animation of the posterior distribution evolution."""
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
            ),
            frames=frames,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path)

    def plot_parameter_convergence(
        self,
        parameter_history: list[dict[str, float]],
        out_path: Path,
    ) -> None:
        """Plot the convergence of parameters with uncertainty bands."""
        if not parameter_history:
            return

        # Extract keys (parameters)
        keys = list(parameter_history[0].keys())
        # Filter strictly for param keys (exclude entropy etc if mixed, though normally pure params)
        # Typically params are x1, x2, etc.

        steps = list(range(len(parameter_history)))

        fig = go.Figure()

        for key in keys:
            values = [step.get(key, math.nan) for step in parameter_history]
            fig.add_trace(go.Scatter(x=steps, y=values, mode="lines", name=key))

        fig.update_layout(
            title="Parameter Convergence",
            xaxis_title="Step",
            yaxis_title="Parameter Value",
            template="plotly_white",
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path)
