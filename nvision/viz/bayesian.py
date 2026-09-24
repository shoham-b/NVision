from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from nvision.viz._f32_json import write_plotly_gz


def _safe_histogram(vals, bins=80, weights=None, density=True):
    """Safely calculate histogram without raising ValueError for zero range, NaNs or empty arrays."""
    vals = np.asarray(vals)
    # Filter to only finite values
    if weights is not None:
        weights = np.asarray(weights)
        finite_mask = np.isfinite(vals) & np.isfinite(weights)
        vals = vals[finite_mask]
        weights = weights[finite_mask]
    else:
        vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        counts = np.zeros(bins)
        bin_edges = np.linspace(-1.0, 1.0, bins + 1)
        return counts, bin_edges

    # Try regular histogram
    try:
        counts, bin_edges = np.histogram(vals, bins=bins, weights=weights, density=density)
        # Check if all counts are finite. If not, fallback
        if np.all(np.isfinite(counts)) and np.all(np.isfinite(bin_edges)):
            return counts, bin_edges
    except ValueError:
        pass

    # If it fails (flat values or infinite ranges), compute a fallback range
    v_min = np.min(vals)
    v_max = np.max(vals)
    if not np.isfinite(v_min) or not np.isfinite(v_max):
        counts = np.zeros(bins)
        bin_edges = np.linspace(-1.0, 1.0, bins + 1)
        return counts, bin_edges

    diff = v_max - v_min
    if diff <= 0:
        half_width = max(0.05 * abs(v_min), 1e-5)
        hist_range = (float(v_min - half_width), float(v_min + half_width))
    else:
        hist_range = (float(v_min), float(v_max))

    try:
        counts, bin_edges = np.histogram(vals, bins=bins, range=hist_range, weights=weights, density=density)
        if np.all(np.isfinite(counts)) and np.all(np.isfinite(bin_edges)):
            return counts, bin_edges
    except ValueError:
        pass

    # Complete fallback
    counts = np.zeros(bins)
    bin_edges = np.linspace(-1.0, 1.0, bins + 1)
    return counts, bin_edges


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


def _trace_one_marginal_posterior(
    posterior: np.ndarray,
    grid: np.ndarray,
    param: str,
    color_idx: int = 0,
) -> list[go.Scatter]:
    # Unified, harmonized color palette matching Plotly's default color cycle
    colors = [
        "rgba(99, 110, 250, 1.0)",  # Blue
        "rgba(239, 85, 59, 1.0)",  # Red
        "rgba(0, 204, 150, 1.0)",  # Green
        "rgba(171, 99, 250, 1.0)",  # Purple
        "rgba(255, 161, 90, 1.0)",  # Orange
        "rgba(25, 211, 243, 1.0)",  # Cyan
    ]
    fill_colors = [
        "rgba(99, 110, 250, 0.25)",
        "rgba(239, 85, 59, 0.25)",
        "rgba(0, 204, 150, 0.25)",
        "rgba(171, 99, 250, 0.25)",
        "rgba(255, 161, 90, 0.25)",
        "rgba(25, 211, 243, 0.25)",
    ]
    c = colors[color_idx % len(colors)]
    fc = fill_colors[color_idx % len(fill_colors)]

    if posterior.ndim == 2:
        # Check if particles (N, 2 or 1) or mixture (K+1, N_grid)
        if posterior.shape[1] in (1, 2):
            weights = posterior[:, 1] if posterior.shape[1] == 2 else None
            # Pre-calculate histogram to avoid client-side Plotly animation bugs with go.Histogram/go.Bar
            vals = posterior[:, 0]
            counts, bin_edges = _safe_histogram(
                vals,
                bins=80,
                weights=weights,
                density=True,
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

            # Subsample particles for the jitter/rug plot to keep HTML size lightweight and fast
            rng = np.random.default_rng(42)
            sub_indices = rng.choice(len(posterior[:, 0]), size=min(150, len(posterior[:, 0])), replace=False)
            jitter_x = posterior[sub_indices, 0]
            max_density = np.max(counts) if len(counts) > 0 else 1.0
            jitter_y = -0.04 * max_density + rng.uniform(-0.02 * max_density, 0.02 * max_density, size=len(jitter_x))

            # Using step-like shape 'hvh' to look exactly like a histogram outline
            customdata = np.stack((bin_edges[:-1], bin_edges[1:]), axis=-1).tolist()
            return [
                go.Scatter(
                    x=bin_centers,
                    y=counts,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor=fc,
                    line=dict(color=c, width=2, shape="hvh"),
                    name=f"{param} (particles)",
                    showlegend=False,
                    customdata=customdata,
                    hovertemplate=(
                        "Density: %{y:.3g}<br>Bin: [%{customdata[0]:.4g}, %{customdata[1]:.4g}]<extra></extra>"
                    ),
                ),
                go.Scatter(
                    x=jitter_x,
                    y=jitter_y,
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=c,
                        opacity=0.45,
                    ),
                    name=f"{param} particles",
                    showlegend=False,
                    hoverinfo="skip",
                ),
            ]
        # Mixture: each row is a component, last row is total
        # Supporting new (2*K + 1) format and backward-compatible (K + 1) format
        total_rows = posterior.shape[0]
        if total_rows % 2 == 1 and total_rows >= 3:
            K = (total_rows - 1) // 2  # noqa: N806
            is_new_format = True
        else:
            K = total_rows - 1  # noqa: N806
            is_new_format = False

        traces: list[go.Scatter] = []

        # Curated harmonious qualitative colors for the mixture components (experts)
        expert_colors = [
            "rgba(26, 188, 156, 1.0)",  # Turquoise
            "rgba(155, 89, 182, 1.0)",  # Amethyst/Purple
            "rgba(230, 126, 34, 1.0)",  # Orange
            "rgba(52, 152, 219, 1.0)",  # Bright Blue
            "rgba(241, 196, 15, 1.0)",  # Sunflower Yellow
            "rgba(231, 76, 60, 1.0)",  # Alizarin Red
        ]

        # Calculate expert weights by numerical integration (sum * dx) over grid support
        dx = grid[1] - grid[0] if len(grid) > 1 else 1.0
        if is_new_format:
            raw_weights = [float(np.sum(posterior[K + j]) * dx) for j in range(K)]
        else:
            raw_weights = [float(np.sum(posterior[j]) * dx) for j in range(K)]
        sum_w = sum(raw_weights)

        # Plot individual experts with distinct colors and semi-transparent lines
        for k in range(K):
            comp_color = expert_colors[k % len(expert_colors)]
            weight = raw_weights[k] / sum_w if sum_w > 0 else 1.0 / K

            # In the new format, posterior[k] is the unweighted expert PDF.
            # In the old format, posterior[k] is the weighted expert PDF.
            y_vals = posterior[k]

            traces.append(
                go.Scatter(
                    x=grid,
                    y=y_vals,
                    mode="lines",
                    name=f"Expert {k + 1} (w={weight:.2f})",
                    line=dict(dash="dash", width=2, color=comp_color),
                    opacity=0.8,
                    legendgroup=param,
                    showlegend=True,
                )
            )
        # Plot total mixture with solid line and fill
        traces.append(
            go.Scatter(
                x=grid,
                y=posterior[-1],
                mode="lines",
                name=f"{param} (combined)",
                line=dict(color=c, width=2.5),
                fill="tozeroy",
                fillcolor=fc,
                legendgroup=param,
                showlegend=True,
            )
        )
        return traces

    return [
        go.Scatter(
            x=grid,
            y=posterior,
            mode="lines",
            fill="tozeroy",
            fillcolor=fc,
            name=param,
            line=dict(color=c, width=2.5),
            showlegend=False,
        )
    ]


class BayesianMixin:
    """Mixin for Bayesian visualization."""

    # Typing for mixin dependency
    out_dir: Path

    def _prepare_animation_frames(
        self,
        posterior_history: list[np.ndarray],
        freq_grid: np.ndarray,
        model_history: list[np.ndarray] | None,
        resampled_steps: list[int] | None,
        is_particles: bool,
    ) -> tuple[list[go.Frame], set[int], float]:
        frames = []
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

        resampled_set = set(resampled_steps) if resampled_steps else set()
        resampling_indices = set()

        for i in step_indices:
            posterior = posterior_history[i]

            data = _trace_one_marginal_posterior(posterior, freq_grid, "Posterior")

            if model_history and i < len(model_history):
                pass

            title_text = f"Step {i + 1}/{total_steps}"
            if i in resampled_set:
                resampling_indices.add(i)
                title_text += " [RESAMPLED] ↺"

            frames.append(
                go.Frame(
                    data=data,
                    name=str(i),
                    layout=go.Layout(title_text=title_text),
                )
            )

        return frames, resampling_indices, max_prob

    def _create_slider_steps(self, frames: list[go.Frame], resampling_indices: set[int]) -> list[dict]:
        return [
            {
                "args": [
                    [frame.name],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
                "label": f"<b><span style='color:red;'>🔴{int(frame.name) + 1}↺</span></b>"
                if int(frame.name) in resampling_indices
                else str(int(frame.name) + 1),
                "method": "animate",
            }
            for frame in frames
        ]

    def _create_speed_buttons(self) -> list[dict]:
        speed_options_single = [
            ("0.5×", 200),
            ("1×", 100),
            ("1.5×", 67),
            ("2×", 50),
        ]

        return [
            dict(
                label=label,
                method="animate",
                args=[
                    None,
                    {
                        "frame": {"duration": duration, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                        "mode": "immediate",
                    },
                ],
            )
            for label, duration in speed_options_single
        ]

    def _build_animation_figure(
        self,
        initial_posterior: np.ndarray,
        freq_grid: np.ndarray,
        is_particles: bool,
        max_prob: float,
        frames: list[go.Frame],
        slider_steps: list[dict],
        speed_buttons_single: list[dict],
    ) -> go.Figure:
        initial_data = _trace_one_marginal_posterior(initial_posterior, freq_grid, "Posterior")
        if is_particles:
            yaxis_layout = dict(title="Probability Density", automargin=True)
        else:
            yaxis_layout = dict(title="Probability Density", automargin=True, range=[0, max_prob * 1.1])

        return go.Figure(
            data=initial_data,
            layout=go.Layout(
                xaxis=dict(title="Frequency / Parameter"),
                yaxis=yaxis_layout,
                margin=dict(l=80, r=40, t=80, b=80),
                title="Posterior Evolution",
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        x=0.02,
                        y=-0.12,
                        xanchor="left",
                        yanchor="top",
                        showactive=False,
                        pad={"r": 10, "t": 0},
                        buttons=[
                            dict(
                                label="▶ Play",
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
                                label="⏸ Pause",
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
                    ),
                    dict(
                        type="buttons",
                        direction="left",
                        x=0.18,
                        y=-0.12,
                        xanchor="left",
                        yanchor="top",
                        showactive=True,
                        active=1,
                        pad={"r": 10, "t": 0},
                        buttons=speed_buttons_single,
                    ),
                ],
                sliders=[
                    dict(
                        active=0,
                        pad={"t": 30, "b": 10},
                        currentvalue={"prefix": "Step: "},
                        steps=slider_steps,
                    )
                ],
            ),
            frames=frames,
        )

    def plot_posterior_animation(
        self,
        posterior_history: list[np.ndarray],
        freq_grid: np.ndarray,
        out_path: Path,
        model_history: list[np.ndarray] | None = None,
        *,
        true_value: float | None = None,
        acquisition_window: tuple[float, float] | None = None,
        experiment_domain: tuple[float, float] | None = None,
        resampled_steps: list[int] | None = None,
    ) -> None:
        """Create an interactive Plotly animation of the posterior distribution evolution.

        If ``true_value`` is set, draws a vertical line at the ground-truth parameter value.
        If ``acquisition_window`` is set, shades the post-sweep search interval (optionally
        with ``experiment_domain`` widening the x-axis to the full sweep range).
        """
        if not posterior_history:
            return

        is_particles = posterior_history[0].ndim == 2

        frames, resampling_indices, max_prob = self._prepare_animation_frames(
            posterior_history, freq_grid, model_history, resampled_steps, is_particles
        )
        slider_steps = self._create_slider_steps(frames, resampling_indices)
        speed_buttons = self._create_speed_buttons()

        fig = self._build_animation_figure(
            posterior_history[0], freq_grid, is_particles, max_prob, frames, slider_steps, speed_buttons
        )

        if acquisition_window is not None:
            _add_acquisition_window_single(fig, acquisition_window, experiment_domain)

        _add_true_vline_single_axis(fig, true_value)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_plotly_gz(fig, out_path)
