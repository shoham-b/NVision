from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nvision.sim.defaults import PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS


@dataclass(frozen=True)
class SubplotOptions:
    row: int
    col: int
    annotation_text: str = "acquisition window"
    zoom_to_window: bool = False


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


def _get_signal_formula(model: Any) -> str:
    """Return a LaTeX string with the full signal formula for the given model.

    Plotly renders LaTeX in titles when wrapped in $...$.
    """
    # Unwrap UnitCubeSignalModel to get the inner physical model
    inner = getattr(model, "inner", model)
    inner_name = type(inner).__name__

    if "NVCenterLorentzian" in inner_name:
        return r"$\mu = 1 - \frac{ak_{NP}}{(f - f_B - \Delta f_{HF})^2 + \Omega^2} - \frac{a}{(f - f_B)^2 + \Omega^2} - \frac{a/k_{NP}}{(f - f_B + \Delta f_{HF})^2 + \Omega^2}$"  # noqa: E501
    if "VoigtZeeman" in inner_name:
        return (
            r"$S(f) = B - \frac{A}{k}(L*G)(f\!-\!f_0\!+\!\Delta) - A(L*G)(f\!-\!f_0) - Ak(L*G)(f\!-\!f_0\!-\!\Delta)$, "
            r"$(L*G)(x) = \int_{-\infty}^{\infty}\!\frac{\eta W/2\pi}{t^2+(\eta W/2)^2} \cdot "
            r"\frac{\exp\!\left[-\frac{(x-t)^2}{2((1-\eta)W/2\sqrt{2\ln 2})^2}\right]}{\frac{(1-\eta)W}{2\sqrt{2\ln 2}}\sqrt{2\pi}}\,dt$"  # noqa: E501
        )
    if "NVCenterVoigt" in inner_name:
        return (
            r"$S(f) = B - \frac{A}{k}(L*G)(f\!-\!f_0\!+\!\Delta) - A(L*G)(f\!-\!f_0) - Ak(L*G)(f\!-\!f_0\!-\!\Delta)$, "
            r"$(L*G)(x) = \int_{-\infty}^{\infty}\!\frac{\eta W/2\pi}{t^2+(\eta W/2)^2} \cdot "
            r"\frac{\exp\!\left[-\frac{(x-t)^2}{2((1-\eta)W/2\sqrt{2\ln 2})^2}\right]}{\frac{(1-\eta)W}{2\sqrt{2\ln 2}}\sqrt{2\pi}}\,dt$"  # noqa: E501
        )
    if "Lorentzian" in inner_name:
        return r"$S(f) = B - A \frac{\omega^2}{(f - f_0)^2 + \omega^2}$"
    if "Gaussian" in inner_name:
        return r"$S(f) = B + A \exp\left[-\frac{(f - f_0)^2}{2\sigma^2}\right]$"
    if "Composite" in inner_name:
        return r"$S(f) = \sum_i S_i(f)$  (composite)"
    return ""


def _get_nv_parameter_descriptions(model: Any) -> dict[str, str]:
    """Generate parameter descriptions with formulas for NV center models.

    Returns a mapping from parameter name to rich description including
    the parameter's role in the signal equation.
    """
    inner = getattr(model, "inner", model)
    inner_name = type(inner).__name__

    is_voigt = "Voigt" in inner_name

    base_descriptions: dict[str, str] = {
        "frequency": "f₀ — central frequency (center of main dip)",
        "linewidth": "ω — Lorentzian linewidth (HWHM)" if not is_voigt else "γ — Lorentzian FWHM",
        "fwhm_total": "W — total effective linewidth (Lorentzian + Gaussian)",
        "lorentz_frac": "η — Lorentzian fraction [0, 1]",
        "fwhm_lorentz": "γ_L — Lorentzian FWHM",
        "fwhm_gauss": "γ_G — Gaussian FWHM (inhomogeneous broadening)",
        "split": "Δ — hyperfine splitting (outer peak distance)",
        "k_np": "k — non-polarization factor (asymmetry ratio)",
        "dip_depth": "A — dip depth (peak amplitude scaling)",
        "background": "B — background signal level",
    }

    return base_descriptions


def _build_subplot_title(param: str, descriptions: dict[str, str] | None) -> str:
    """Build rich subplot title with parameter name and description."""
    param_units = {
        "frequency": " (GHz)",
        "linewidth": " (MHz)",
        "split": " (MHz)",
        "fwhm_total": " (MHz)",
        "fwhm_lorentz": " (MHz)",
        "fwhm_gauss": " (MHz)",
    }
    unit_str = param_units.get(param, "")
    param_display = f"{param}{unit_str}"

    if descriptions is None:
        return param_display
    desc = descriptions.get(param, "")
    if desc:
        return f"<b>{param_display}</b><br><span style='font-size:10px;color:#666;'>{desc}</span>"
    return param_display


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
    window: tuple[float, float],
    full_domain: tuple[float, float] | None,
    opts: SubplotOptions,
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
        annotation_text=opts.annotation_text,
        annotation_position="top left",
        row=opts.row,
        col=opts.col,
    )
    if opts.zoom_to_window:
        # Focus only on the acquisition window with 10% padding
        pad = (x1 - x0) * 0.1
        fig.update_xaxes(range=[x0 - pad, x1 + pad], row=opts.row, col=opts.col)
    elif full_domain is not None:
        flo, fhi = float(full_domain[0]), float(full_domain[1])
        if math.isfinite(flo) and math.isfinite(fhi) and fhi > flo:
            lo = min(flo, x0)
            hi = max(fhi, x1)
            fig.update_xaxes(range=[lo, hi], row=opts.row, col=opts.col)


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
    param_units = {
        "frequency": " GHz",
        "linewidth": " MHz",
        "split": " MHz",
        "fwhm_total": " MHz",
        "fwhm_lorentz": " MHz",
        "fwhm_gauss": " MHz",
    }
    for i, name in enumerate(param_names, start=1):
        raw = true_params.get(name)
        if raw is None:
            continue
        tv = float(raw)
        if not math.isfinite(tv):
            continue
        unit_label = param_units.get(name, "")
        fig.add_vline(
            x=tv,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text=f"true: {tv:.6g}{unit_label}",
            annotation_position="top right",
            row=i,
            col=1,
        )


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


def _collect_param_keys(parameter_history: list[dict[str, float]]) -> list[str]:
    keys: list[str] = []
    for step in parameter_history:
        for k in step:
            if k not in keys:
                keys.append(k)
    return keys


def _build_param_series(
    keys: list[str],
    history: list[dict[str, float]] | None,
) -> tuple[list[list[float]], list[list[float]]]:
    if not history:
        return [], []
    raw_y: list[list[float]] = []
    norm_y: list[list[float]] = []
    for key in keys:
        values = [step.get(key, math.nan) for step in history]
        raw_y.append(values)
        v0 = next((v for v in values if not math.isnan(v)), math.nan)
        if not math.isnan(v0) and v0 != 0:
            norm_y.append([v / v0 if not math.isnan(v) else math.nan for v in values])
        else:
            norm_y.append([math.nan] * len(values))
    return raw_y, norm_y


class BayesianMixin:
    """Mixin for Bayesian visualization."""

    # Typing for mixin dependency
    out_dir: Path

    def _inject_timeline_bridge(self, html_path: Path, step_labels: list[str] | None = None) -> None:
        """Injects timeline bridge JS to expose showFrame, totalFrames, and stepValues for global parent sync."""
        if not html_path.exists():
            return
        import json

        content = html_path.read_text(encoding="utf-8")

        hide_css = """
        <style>
          @media screen {
            body.in-iframe .updatemenu-container { display: none !important; }
            body.in-iframe .slider-container { display: none !important; }
          }
        </style>
        """

        labels_json = "[]" if step_labels is None else json.dumps(step_labels)

        bridge_script = f"""
        {hide_css}
        <script>
        (function() {{
            if (window.self !== window.top) {{
                document.body.classList.add('in-iframe');
            }}

            function getPlotlyDiv() {{
                return document.querySelector('.plotly-graph-div');
            }}

            window.showFrame = function(idx) {{
                var gd = getPlotlyDiv();
                if (!gd || !window.Plotly) return;

                window.Plotly.animate(gd, [String(idx)], {{
                    mode: 'immediate',
                    transition: {{duration: 0}},
                    frame: {{duration: 0, redraw: true}}
                }});
            }};

            var checkInterval = setInterval(function() {{
                var gd = getPlotlyDiv();
                if (gd && gd._transitionData && gd._transitionData._frames) {{
                    window.totalFrames = gd._transitionData._frames.length;
                    window.stepValues = {labels_json};
                    if (window.stepValues.length === 0) {{
                        try {{
                            var steps = gd.layout.sliders[0].steps;
                            window.stepValues = steps.map(function(s) {{ return s.label; }});
                        }} catch (e) {{}}
                    }}
                    clearInterval(checkInterval);
                }}
            }}, 50);
        }})();
        </script>
        """

        if "</body>" in content:
            content = content.replace("</body>", f"{bridge_script}\n</body>")
        else:
            content += f"\n{bridge_script}"

        html_path.write_text(content, encoding="utf-8")

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
        fig.write_html(out_path, include_mathjax="cdn")

    def plot_posterior_animation_all_params(  # noqa: C901
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
        param_descriptions: dict[str, str] | None = None,
        signal_formula: str | None = None,
        resampled_steps: list[int] | None = None,
        weight_history: list[np.ndarray] | None = None,
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

        ``param_descriptions`` maps parameter names to rich HTML descriptions that will be
        displayed as subplot titles alongside the parameter names.
        """
        if not posterior_inputs_by_param:
            return

        # Define scaling factors for physical parameters to more natural units
        param_scales = {
            "frequency": 1e9,
            "linewidth": 1e6,
            "split": 1e6,
            "fwhm_total": 1e6,
            "fwhm_lorentz": 1e6,
            "fwhm_gauss": 1e6,
        }

        # Scale posterior_inputs_by_param
        posterior_inputs_by_param_scaled = {}
        for p, (history, grid) in posterior_inputs_by_param.items():
            scale = param_scales.get(p, 1.0)
            if scale == 1.0:
                posterior_inputs_by_param_scaled[p] = (history, grid)
                continue
            grid_scaled = grid / scale
            history_scaled = []
            for post in history:
                if post.ndim == 2 and post.shape[1] in (1, 2):
                    # Particle filter format: [particle_vals, weights]
                    post_scaled = post.copy()
                    post_scaled[:, 0] /= scale
                    history_scaled.append(post_scaled)
                else:
                    # Grid / Mixture PDF format: [pdfs]
                    history_scaled.append(post * scale)
            posterior_inputs_by_param_scaled[p] = (history_scaled, grid_scaled)
        posterior_inputs_by_param = posterior_inputs_by_param_scaled

        # Scale true_params
        if true_params is not None:
            true_params_scaled = {}
            for p, val in true_params.items():
                scale = param_scales.get(p, 1.0)
                true_params_scaled[p] = val / scale
            true_params = true_params_scaled

        # Scale acquisition_window
        if acquisition_window is not None and acquisition_param is not None:
            scale = param_scales.get(acquisition_param, 1.0)
            acquisition_window = (acquisition_window[0] / scale, acquisition_window[1] / scale)

        # Scale experiment_domain
        if experiment_domain is not None and acquisition_param is not None:
            scale = param_scales.get(acquisition_param, 1.0)
            experiment_domain = (experiment_domain[0] / scale, experiment_domain[1] / scale)

        # Scale narrowed_param_bounds
        if narrowed_param_bounds is not None:
            narrowed_param_bounds_scaled = {}
            for p, bounds in narrowed_param_bounds.items():
                scale = param_scales.get(p, 1.0)
                narrowed_param_bounds_scaled[p] = (bounds[0] / scale, bounds[1] / scale)
            narrowed_param_bounds = narrowed_param_bounds_scaled

        # Scale per_step_narrowed_bounds
        if per_step_narrowed_bounds is not None:
            per_step_narrowed_bounds_scaled = []
            for bounds_dict in per_step_narrowed_bounds:
                scaled_dict = {}
                for p, bounds in bounds_dict.items():
                    scale = param_scales.get(p, 1.0)
                    scaled_dict[p] = (bounds[0] / scale, bounds[1] / scale)
                per_step_narrowed_bounds_scaled.append(scaled_dict)
            per_step_narrowed_bounds = per_step_narrowed_bounds_scaled

        param_names = list(posterior_inputs_by_param.keys())

        first_param = param_names[0]
        total_steps = len(posterior_inputs_by_param[first_param][0])
        if total_steps == 0:
            return

        step_indices = list(range(total_steps))
        if total_steps > 100:
            step_indices = [int(x) for x in np.linspace(0, total_steps - 1, 100)]

        n = len(param_names)
        has_weights = weight_history is not None and len(weight_history) > 0

        if has_weights:
            subplot_titles = (
                *tuple(_build_subplot_title(p, param_descriptions) for p in param_names),
                "<b>Parameter Correlation Matrix Heatmap</b>",
                "<b>Sorted Particle Weights (Concentration Profile)</b>",
                "<b>Effective Sample Size (ESS) Timeline</b>",
                "<b>Timeline (Resampling & Progress)</b>",
            )
            total_rows = n + 4
            corr_row = n + 1
            weights_row = n + 2
            ess_row = n + 3
            row_heights = [1.0] * n + [1.8] + [0.8, 0.8] + [0.2]

            ess_history = []
            for w in weight_history:
                sum_sq = np.sum(w**2)
                ess_history.append(1.0 / sum_sq if sum_sq > 0.0 else 0.0)
            num_particles = len(weight_history[0])
            from nvision.belief.smc_marginal import NVISION_SMC_ESS_THRESHOLD

            ess_threshold_val = NVISION_SMC_ESS_THRESHOLD * num_particles
        else:
            subplot_titles = (
                *tuple(_build_subplot_title(p, param_descriptions) for p in param_names),
                "<b>Timeline (Resampling & Progress)</b>",
            )
            total_rows = n + 1
            row_heights = [1.0] * n + [0.2]

        fig = make_subplots(
            rows=total_rows,
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=min(0.08, 0.5 / max(total_rows, 1)),  # slightly more spacing for rich titles
            row_heights=row_heights,
        )
        timeline_row = total_rows

        def traces_for_step(step_idx: int) -> list[object]:  # noqa: C901
            traces: list[object] = []
            for param_idx, param in enumerate(param_names, start=1):
                posterior_history, grid = posterior_inputs_by_param[param]
                posterior = posterior_history[-1] if step_idx >= len(posterior_history) else posterior_history[step_idx]
                for tr in _trace_one_marginal_posterior(posterior, grid, param, color_idx=param_idx - 1):
                    if param_idx > 1:
                        tr.update(xaxis=f"x{param_idx}", yaxis=f"y{param_idx}")
                    else:
                        tr.update(xaxis="x", yaxis="y")
                    traces.append(tr)

            if has_weights:
                # Calculate weighted correlation matrix of the particles at the current step
                step_particles = []
                for param in param_names:
                    posterior_history, _ = posterior_inputs_by_param[param]
                    post = posterior_history[-1] if step_idx >= len(posterior_history) else posterior_history[step_idx]
                    step_particles.append(post[:, 0])

                particles_matrix = np.vstack(step_particles)  # (n_params, n_particles)
                w_curr = weight_history[step_idx]
                w_sum = np.sum(w_curr)
                w_norm = w_curr / w_sum if w_sum > 0.0 else np.ones_like(w_curr) / len(w_curr)

                # Compute weighted covariance and correlation matrix
                cov = np.cov(particles_matrix, aweights=w_norm)
                std = np.sqrt(np.diag(cov))
                std[std == 0.0] = 1e-8
                corr = cov / np.outer(std, std)
                corr = np.clip(corr, -1.0, 1.0)

                # Add Heatmap trace
                traces.append(
                    go.Heatmap(
                        z=corr,
                        x=param_names,
                        y=param_names,
                        zmin=-1,
                        zmax=1,
                        colorscale="RdBu",
                        showscale=True,
                        xaxis=f"x{corr_row}",
                        yaxis=f"y{corr_row}",
                        text=np.round(corr, 2),
                        texttemplate="%{text:.2f}",
                        textfont=dict(size=10),
                        colorbar=dict(
                            title="Corr",
                            thickness=15,
                            len=0.3,
                            y=0.3,
                            yanchor="middle",
                        ),
                    )
                )

                # Sorted Particle Weights trace
                w_curr_sorted = np.sort(weight_history[step_idx])[::-1]
                traces.append(
                    go.Scatter(
                        x=list(range(len(w_curr_sorted))),
                        y=w_curr_sorted,
                        mode="lines",
                        fill="tozeroy",
                        fillcolor="rgba(30, 144, 255, 0.25)",
                        line=dict(color="rgba(30, 144, 255, 0.85)", width=2),
                        name="Weight Profile",
                        showlegend=False,
                        xaxis=f"x{weights_row}",
                        yaxis=f"y{weights_row}",
                    )
                )

                # ESS Timeline trace
                ess_x = list(range(1, step_idx + 2))
                ess_y = [ess_history[i] for i in range(step_idx + 1)]
                traces.append(
                    go.Scatter(
                        x=ess_x,
                        y=ess_y,
                        mode="lines+markers",
                        line=dict(color="rgba(155, 89, 182, 0.85)", width=2.5),
                        marker=dict(size=4, color="rgba(155, 89, 182, 1.0)"),
                        name="ESS",
                        showlegend=False,
                        xaxis=f"x{ess_row}",
                        yaxis=f"y{ess_row}",
                    )
                )

                # Resampling Threshold dashed line
                traces.append(
                    go.Scatter(
                        x=[1, total_steps],
                        y=[ess_threshold_val, ess_threshold_val],
                        mode="lines",
                        line=dict(color="rgba(231, 76, 60, 0.5)", width=1.5, dash="dash"),
                        name=f"Resampling Threshold ({int(NVISION_SMC_ESS_THRESHOLD * 100)}%)",
                        showlegend=False,
                        xaxis=f"x{ess_row}",
                        yaxis=f"y{ess_row}",
                    )
                )

                # Resampling markers on ESS (always present)
                if resampled_steps:
                    rs_up_to_now = [rs for rs in resampled_steps if rs <= step_idx]
                    traces.append(
                        go.Scatter(
                            x=[rs + 1 for rs in rs_up_to_now] if rs_up_to_now else [],
                            y=[ess_history[rs] for rs in rs_up_to_now] if rs_up_to_now else [],
                            mode="markers",
                            marker=dict(symbol="x", size=10, color="rgba(231, 76, 60, 0.9)", line=dict(width=2)),
                            name="Resample Event",
                            showlegend=False,
                            xaxis=f"x{ess_row}",
                            yaxis=f"y{ess_row}",
                        )
                    )

            # Add timeline traces
            # Base line
            traces.append(
                go.Scatter(
                    x=[1, total_steps],
                    y=[0, 0],
                    mode="lines",
                    line=dict(color="#eee", width=2),
                    showlegend=False,
                    xaxis=f"x{timeline_row}",
                    yaxis=f"y{timeline_row}",
                )
            )

            # Resampling markers (Red vertical bars)
            if resampled_steps:
                rs_x = [rs + 1 for rs in resampled_steps]
                traces.append(
                    go.Scatter(
                        x=rs_x,
                        y=[0] * len(rs_x),
                        mode="markers",
                        marker=dict(symbol="line-ns-open", size=20, line=dict(color="rgba(231, 76, 60, 0.8)", width=3)),
                        name="Resampling",
                        showlegend=False,
                        xaxis=f"x{timeline_row}",
                        yaxis=f"y{timeline_row}",
                        hoverinfo="text",
                        text=[f"Resampled at step {x}" for x in rs_x],
                    )
                )

            # Current position cursor (Blue vertical line)
            traces.append(
                go.Scatter(
                    x=[step_idx + 1, step_idx + 1],
                    y=[-1, 1],
                    mode="lines",
                    line=dict(color="#1e90ff", width=4),
                    name="Current Step",
                    showlegend=False,
                    xaxis=f"x{timeline_row}",
                    yaxis=f"y{timeline_row}",
                    hoverinfo="skip",
                )
            )

            # Always add exactly one trace for the acquisition window/refocusing overlay to keep trace count constant.
            acq_x = []
            acq_y = []
            acq_name = "post-sweep acquisition"
            row_nb = 1
            if acquisition_param and acquisition_param in param_names:
                row_nb = param_names.index(acquisition_param) + 1

                # Check for step-specific bounds
                step_bounds = None
                if per_step_narrowed_bounds is not None and step_idx < len(per_step_narrowed_bounds):
                    step_bounds = per_step_narrowed_bounds[step_idx]
                elif narrowed_param_bounds is not None:
                    step_bounds = narrowed_param_bounds

                window = None
                if step_bounds and acquisition_param in step_bounds:
                    window = step_bounds[acquisition_param]
                    if step_idx >= 32:
                        acq_name = "posterior refocusing"
                elif (not step_bounds) and acquisition_window is not None and step_idx >= 32:
                    window = acquisition_window

                if window is not None:
                    x0, x1 = float(window[0]), float(window[1])
                    if math.isfinite(x0) and math.isfinite(x1) and x1 > x0:
                        acq_x = [x0, x1, x1, x0, x0]
                        y_limit_val = y_max_by_param.get(acquisition_param)
                        y_val = float(y_limit_val * 2.0) if y_limit_val is not None else 1000.0
                        acq_y = [0.0, 0.0, y_val, y_val, 0.0]

            traces.append(
                go.Scatter(
                    x=acq_x,
                    y=acq_y,
                    fill="toself",
                    fillcolor="rgba(46, 204, 113, 0.18)",
                    line=dict(color="rgba(46, 204, 113, 0.75)" if acq_x else "rgba(0,0,0,0)", width=1 if acq_x else 0),
                    hoverinfo="skip" if acq_x else "none",
                    showlegend=False,
                    xaxis=f"x{row_nb}" if row_nb > 1 else "x",
                    yaxis=f"y{row_nb}" if row_nb > 1 else "y",
                    name=acq_name,
                )
            )

            return traces

        # Pre-calculate global maximum density (y-axis) and active support range (x-axis) across all steps
        y_max_by_param = {}
        x_range_by_param = {}
        for param in param_names:
            posterior_history, grid = posterior_inputs_by_param[param]

            # SMC posteriors in posterior_history are of shape (num_particles, 2)
            is_smc = (
                len(posterior_history) > 0
                and posterior_history[0].ndim == 2
                and posterior_history[0].shape[1] in (1, 2)
            )

            if is_smc:
                # 1. SMC Particle belief
                # The active support is the union of the particle ranges across all steps
                x_min_val = float("inf")
                x_max_val = float("-inf")
                max_density = 0.0
                for post in posterior_history:
                    vals = post[:, 0]
                    x_min_val = min(x_min_val, float(np.min(vals)))
                    x_max_val = max(x_max_val, float(np.max(vals)))

                    weights = post[:, 1] if post.shape[1] == 2 else None
                    counts, _ = _safe_histogram(vals, bins=80, weights=weights, density=True)
                    m = np.max(counts)
                    if m > max_density:
                        max_density = m

                span = x_max_val - x_min_val
                if span <= 0:
                    span = 1e-6
                x_range_by_param[param] = [float(x_min_val - 0.05 * span), float(x_max_val + 0.05 * span)]
                y_max_by_param[param] = float(max_density) if max_density > 0.0 else 1.0
            else:
                # 2. Grid or Mixture belief
                max_density = 0.0
                min_active_idx = len(grid) - 1
                max_active_idx = 0

                for post in posterior_history:
                    post_1d = post[-1] if post.ndim == 2 else post
                    m = np.max(post_1d)
                    if m > max_density:
                        max_density = m

                    if m > 0:
                        thresh = 1e-4 * m
                        active_indices = np.where(post_1d > thresh)[0]
                        if len(active_indices) > 0:
                            min_active_idx = min(min_active_idx, active_indices[0])
                            max_active_idx = max(max_active_idx, active_indices[-1])

                y_max_by_param[param] = max_density if max_density > 0 else 1.0

                if min_active_idx <= max_active_idx:
                    x0 = grid[min_active_idx]
                    x1 = grid[max_active_idx]
                    span = x1 - x0
                    if span == 0:
                        span = 1e-6
                    x_range_by_param[param] = [float(x0 - 0.05 * span), float(x1 + 0.05 * span)]
                else:
                    x_range_by_param[param] = [float(grid[0]), float(grid[-1])]

        # Add initial traces in the exact same structure as frames
        init_idx = step_indices[0]
        for tr in traces_for_step(init_idx):
            fig.add_trace(tr)

        for i, name in enumerate(param_names, start=1):
            x_limit = x_range_by_param[name]
            y_limit_val = y_max_by_param[name]
            is_smc_subplot = (
                len(posterior_inputs_by_param[name][0]) > 0
                and posterior_inputs_by_param[name][0][0].ndim == 2
                and posterior_inputs_by_param[name][0][0].shape[1] in (1, 2)
            )
            if y_limit_val is not None:
                y_limit = float(y_limit_val * 1.15)
                y_min = float(-0.08 * y_limit_val) if is_smc_subplot else 0.0
                fig.update_yaxes(title_text="density", range=[y_min, y_limit], automargin=True, row=i, col=1)
            else:
                # SMC Particle Histogram auto-scaling
                fig.update_yaxes(title_text="density", automargin=True, row=i, col=1)
            fig.update_xaxes(title_text=name, range=x_limit, row=i, col=1)

        if has_weights:
            fig.update_yaxes(title_text="weight", automargin=True, row=weights_row, col=1)
            fig.update_xaxes(title_text="Sorted Particle Index", row=weights_row, col=1)

            fig.update_yaxes(title_text="ESS", automargin=True, range=[0, num_particles * 1.05], row=ess_row, col=1)
            fig.update_xaxes(title_text="Measurement Step", range=[0.5, total_steps + 0.5], row=ess_row, col=1)

            fig.update_yaxes(autorange="reversed", row=corr_row, col=1)

        # Format Timeline subplot
        fig.update_yaxes(visible=False, range=[-1, 1], row=timeline_row, col=1)
        fig.update_xaxes(title_text="Measurement Step", range=[0.5, total_steps + 0.5], row=timeline_row, col=1)

        frames = []
        refocusing_steps = set()  # Track steps where refocusing occurs
        resampling_indices = set()  # Track indices in frames where resampling happened
        resampled_set = set(resampled_steps) if resampled_steps else set()

        for si, step_idx in enumerate(step_indices):
            frame_data = traces_for_step(step_idx)

            # Track refocusing events (after initial sweep, every 20 steps)
            if step_idx >= 32 and (step_idx - 32) % 20 == 0:
                refocusing_steps.add(si)

            # Add indicators to title
            _formula_suffix = f"  {signal_formula}" if signal_formula else ""
            title_text = f"Posterior evolution (all parameters){_formula_suffix}<br>step {step_idx + 1}/{total_steps}"

            is_resampled = step_idx in resampled_set
            if is_resampled:
                resampling_indices.add(si)
                title_text = title_text.replace("<br>", " [RESAMPLED] ↺<br>", 1)

            if si in refocusing_steps:
                title_text = title_text.replace("<br>", " [REFOCUSING]<br>", 1)

            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(si),
                    layout=go.Layout(
                        title_text=title_text,
                        annotations=[
                            dict(
                                text="RESAMPLED ↺",
                                xref="paper",
                                yref="paper",
                                x=1.0,
                                y=1.02,
                                xanchor="right",
                                yanchor="bottom",
                                showarrow=False,
                                font=dict(size=10, color="#ffffff"),
                                bgcolor="rgba(231, 76, 60, 0.7)",
                                bordercolor="rgba(231, 76, 60, 0.9)",
                                borderwidth=0,
                                borderpad=4,
                            )
                        ]
                        if is_resampled
                        else [],
                    ),
                )
            )

        slider_steps = []
        for si, frame in enumerate(frames):
            step_num = step_indices[si] + 1
            # Highlight refocusing steps with brackets and resampling with a circular arrow
            base_label = str(step_num)
            if si in refocusing_steps:
                base_label = f"[{base_label}]"
            if si in resampling_indices:
                base_label = f"{base_label} ↺"

            label = base_label
            if si in resampling_indices:
                # Use bold red for resampling labels
                label = f"<b><span style='color:red;'>🔴{base_label}</span></b>"

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
            opts = SubplotOptions(
                row=row_ap,
                col=1,
                annotation_text="post-sweep acquisition",
                zoom_to_window=False,
            )
            _add_acquisition_window_subplot(
                fig,
                window=acquisition_window,
                full_domain=experiment_domain,
                opts=opts,
            )

        _add_true_vline_subplots(fig, param_names, true_params)

        base_title = "Posterior evolution (all parameters)"
        if signal_formula:
            base_title = f"{base_title}  {signal_formula}"

        # Speed multipliers and their corresponding frame durations (ms)
        # Default is 100ms per frame, so speeds are inversely proportional
        speed_options = [
            ("0.5×", 200),
            ("1×", 100),
            ("1.5×", 67),
            ("2×", 50),
        ]

        # Build play/pause button with toggle capability
        def play_pause_button(is_play: bool) -> dict:
            if is_play:
                return dict(
                    label="<b>▶ Play</b>",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                            "mode": "immediate",
                        },
                    ],
                )
            else:
                return dict(
                    label="<b>⏸ Pause</b>",
                    method="animate",
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                )

        # Build speed buttons
        [
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
            for label, duration in speed_options
        ]

        # Update layout properties for clean display (excluding sliders/updatemenus)
        fig.update_layout(
            title_text=f"{base_title}<br>step {step_indices[0] + 1}/{total_steps}",
            title_font_size=14,
            title_y=0.98,
            title_yanchor="top",
            template="plotly_white",
            height=max(400, int(200 * (n + 2.5 if has_weights else n))),
            margin=dict(
                l=90, r=40, t=145, b=50
            ),  # Increased t (top margin) for header spacing, decreased b (bottom margin)
            updatemenus=[],
            sliders=[],
        )
        fig.frames = frames

        out_path.parent.mkdir(parents=True, exist_ok=True)

        import json

        html_content = fig.to_html(include_plotlyjs="cdn", include_mathjax="cdn", full_html=True)

        # Prepare custom HTML header injection for premium sticky controls
        max_idx = len(frames) - 1
        total_frames = len(frames)
        step_values_json = json.dumps([step["label"] for step in slider_steps])

        custom_controls = f"""  # noqa: E501
        <div id="custom-timeline-controls">
          <button id="timeline-play-btn" class="timeline-btn">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" "  # noqa: E501
            "style="display:inline-block; vertical-align:middle; margin-right:4px;">"
            "<path d="M8 5v14l11-7z"/></svg><span>Play</span>
          </button>
          <div class="timeline-slider-container">
            <input type="range" id="timeline-range-slider" class="timeline-slider" min="0" max="{max_idx}" value="0">
            <span id="timeline-step-label" class="timeline-label">Step: 0 / 0</span>
          </div>
          <select id="timeline-speed-select" class="speed-select">
            <option value="500">Speed: 0.5x</option>
            <option value="200" selected>Speed: 1x</option>
            <option value="100">Speed: 2x</option>
            <option value="50">Speed: 4x</option>
          </select>
        </div>

        <style>
          @media screen {{
            body.in-iframe #custom-timeline-controls {{
              display: none !important;
            }}
            body.in-iframe .plotly-graph-div {{
              padding-top: 0 !important;
            }}
          }}
          #custom-timeline-controls {{
            position: -webkit-sticky;
            position: sticky;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 999999;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-bottom: 1.5px solid #e2e8f0;
            padding: 10px 20px;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            gap: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
          }}
          .plotly-graph-div {{
            padding-top: 10px;
          }}
          .timeline-btn {{
            background: #e6f2ff;
            border: 2px solid #1e90ff;
            color: #005b96;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            outline: none;
            min-width: 95px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
          }}
          .timeline-btn:hover {{
            background: #1e90ff;
            color: white;
            box-shadow: 0 4px 6px rgba(30, 144, 255, 0.2);
          }}
          .timeline-btn:active {{
            transform: scale(0.96);
          }}
          .timeline-slider-container {{
            flex-grow: 1;
            display: flex;
            align-items: center;
            gap: 12px;
          }}
          .timeline-slider {{
            flex-grow: 1;
            height: 6px;
            -webkit-appearance: none;
            appearance: none;
            background: #cbd5e1;
            outline: none;
            border-radius: 999px;
            cursor: pointer;
            transition: background 0.15s ease;
          }}
          .timeline-slider:hover {{
            background: #94a3b8;
          }}
          .timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #1e90ff;
            border: 2.5px solid white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.25);
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
          }}
          .timeline-slider::-webkit-slider-thumb:hover {{
            transform: scale(1.25);
            background: #0070e0;
          }}
          .timeline-label {{
            font-size: 14px;
            font-weight: 700;
            color: #334155;
            min-width: 150px;
            white-space: nowrap;
          }}
          .speed-select {{
            padding: 8px 12px;
            border-radius: 8px;
            border: 1.5px solid #cbd5e1;
            font-size: 13px;
            font-weight: 600;
            background: white;
            color: #334155;
            cursor: pointer;
            outline: none;
            transition: all 0.2s ease;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
          }}
          .speed-select:hover {{
            border-color: #94a3b8;
          }}
        </style>

        <script>
          window.addEventListener('DOMContentLoaded', () => {{
            const gd = document.getElementsByClassName('plotly-graph-div')[0];
            if (!gd) return;

            const playBtn = document.getElementById('timeline-play-btn');
            const playBtnText = playBtn.querySelector('span');
            const playBtnIcon = playBtn.querySelector('svg');
            const slider = document.getElementById('timeline-range-slider');
            const label = document.getElementById('timeline-step-label');
            const speedSelect = document.getElementById('timeline-speed-select');

            let isPlaying = false;
            let playInterval = null;
            let totalFrames = {total_frames};
            let stepValues = {step_values_json};

            slider.max = totalFrames - 1;
            updateLabel(0);

            slider.addEventListener('input', (e) => {{
              const frameIdx = parseInt(e.target.value, 10);
              showFrame(frameIdx);
            }});

            playBtn.addEventListener('click', () => {{
              if (isPlaying) {{
                pause();
              }} else {{
                play();
              }}
            }});

            speedSelect.addEventListener('change', () => {{
              if (isPlaying) {{
                pause();
                play();
              }}
            }});

            function updateLabel(idx) {{
              const val = stepValues[idx] !== undefined ? stepValues[idx] : idx;
              label.textContent = `Step: ${{val}} (${{idx + 1}} / ${{totalFrames}})`;
            }}

            function showFrame(idx) {{
              updateLabel(idx);
              Plotly.animate(gd, [String(idx)], {{
                mode: 'immediate',
                transition: {{duration: 0}},
                frame: {{duration: 0, redraw: true}}
              }});
            }}

            function play() {{
              isPlaying = true;
              playBtnText.textContent = 'Pause';
              playBtnIcon.innerHTML = '<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>';
              playBtn.style.background = '#ffebe6';
              playBtn.style.borderColor = '#ff4d4d';
              playBtn.style.color = '#960000';

              const intervalMs = parseInt(speedSelect.value, 10);
              playInterval = setInterval(() => {{
                let nextIdx = parseInt(slider.value, 10) + 1;
                if (nextIdx >= totalFrames) {{
                  nextIdx = 0;
                }}
                slider.value = nextIdx;
                showFrame(nextIdx);
              }}, intervalMs);
            }}

            function pause() {{
              isPlaying = false;
              playBtnText.textContent = 'Play';
              playBtnIcon.innerHTML = '<path d="M8 5v14l11-7z"/>';
              playBtn.style.background = '#e6f2ff';
              playBtn.style.borderColor = '#1e90ff';
              playBtn.style.color = '#005b96';

              clearInterval(playInterval);
            }}

            // Sync bridge interface for parent global timeline sync
            if (window.self !== window.top) {{
              document.body.classList.add('in-iframe');
            }}
            window.showFrame = function(idx) {{
              slider.value = idx;
              showFrame(idx);
            }};
            window.totalFrames = totalFrames;
            window.stepValues = stepValues;
          }});
        </script>
        """

        # Inject custom timeline header immediately at the start of <body>
        if "<body>" in html_content:
            html_content = html_content.replace("<body>", f"<body>{custom_controls}", 1)
        elif "<body >" in html_content:
            html_content = html_content.replace("<body >", f"<body>{custom_controls}", 1)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def plot_parameter_convergence(  # noqa: C901
        self,
        parameter_history: list[dict[str, float]],
        out_path: Path,
        *,
        estimates_history: list[dict[str, float]] | None = None,
        true_params: Mapping[str, float] | None = None,
    ) -> None:
        """Plot uncertainty trajectories over steps for Bayesian parameters.

        When ``estimates_history`` is provided, dashed estimate traces are drawn
        on a secondary y-axis so parameter values and their uncertainties can be
        viewed together.  ``true_params`` adds horizontal dotted reference lines
        for each known ground-truth value.
        """
        if not parameter_history:
            return

        # Define scaling factors for physical parameters to more natural units
        param_scales = {
            "frequency": 1e9,
            "linewidth": 1e6,
            "split": 1e6,
            "fwhm_total": 1e6,
            "fwhm_lorentz": 1e6,
            "fwhm_gauss": 1e6,
        }
        param_units = {
            "frequency": " (GHz)",
            "linewidth": " (MHz)",
            "split": " (MHz)",
            "fwhm_total": " (MHz)",
            "fwhm_lorentz": " (MHz)",
            "fwhm_gauss": " (MHz)",
        }

        # Scale and rename keys in histories
        parameter_history_scaled = []
        for d in parameter_history:
            d_scaled = {}
            for k, val in d.items():
                scale = param_scales.get(k, 1.0)
                unit = param_units.get(k, "")
                d_scaled[f"{k}{unit}"] = val / scale
            parameter_history_scaled.append(d_scaled)
        parameter_history = parameter_history_scaled

        if estimates_history is not None:
            estimates_history_scaled = []
            for d in estimates_history:
                d_scaled = {}
                for k, val in d.items():
                    scale = param_scales.get(k, 1.0)
                    unit = param_units.get(k, "")
                    d_scaled[f"{k}{unit}"] = val / scale
                estimates_history_scaled.append(d_scaled)
            estimates_history = estimates_history_scaled

        if true_params is not None:
            true_params_scaled = {}
            for k, val in true_params.items():
                scale = param_scales.get(k, 1.0)
                unit = param_units.get(k, "")
                true_params_scaled[f"{k}{unit}"] = val / scale
            true_params = true_params_scaled

        keys = _collect_param_keys(parameter_history)

        if not keys:
            return

        steps = list(range(len(parameter_history)))
        raw_y, norm_y = _build_param_series(keys, parameter_history)
        est_raw_y, est_norm_y = _build_param_series(keys, estimates_history)

        def _is_noise_param(name: str) -> bool:
            name_lc = name.lower()
            return any(p in name_lc for p in ("noise", "poisson", "drift", "sigma", "scale"))

        fig = go.Figure()

        for key, y in zip(keys, raw_y, strict=True):
            color = "red" if _is_noise_param(key) else None
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=y,
                    mode="lines",
                    name=f"{key} (unc)",
                    line=dict(color=color),
                    legendgroup=key,
                )
            )

        # Estimate traces on secondary y-axis
        for key, y in zip(keys, est_raw_y, strict=True):
            color = "red" if _is_noise_param(key) else None
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=y,
                    mode="lines",
                    name=f"{key} (est)",
                    line=dict(dash="dash", color=color),
                    yaxis="y2",
                    legendgroup=key,
                    visible="legendonly",
                )
            )

        # True-value horizontal lines
        true_raw_y = []
        true_norm_y = []
        has_true = False
        if true_params:
            for key in keys:
                tv = true_params.get(key)
                if tv is not None and math.isfinite(float(tv)):
                    has_true = True
                    init_est = estimates_history[0].get(key, 1.0) if estimates_history else 1.0
                    if init_est == 0:
                        init_est = 1.0
                    true_raw_y.append([float(tv), float(tv)])
                    true_norm_y.append([float(tv) / init_est, float(tv) / init_est])
                else:
                    true_raw_y.append([0.0, 0.0])
                    true_norm_y.append([0.0, 0.0])

        if has_true:
            for key, y_val in zip(keys, true_raw_y, strict=False):
                fig.add_trace(
                    go.Scatter(
                        x=[steps[0], steps[-1]],
                        y=y_val,
                        mode="lines",
                        name=f"{key} (true)",
                        line=dict(dash="dot", color="green"),
                        yaxis="y2" if estimates_history else "y",
                        legendgroup=key,
                        visible="legendonly",
                    )
                )

        fig.update_layout(
            title="Parameter Convergence",
            xaxis_title="Step",
            yaxis_title="Uncertainty (std dev)",
            yaxis2=dict(
                title="Parameter estimate",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            template="plotly_white",
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    pad={"r": 8, "t": 8},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.18,
                    yanchor="bottom",
                    buttons=[
                        dict(
                            label="Absolute",
                            method="update",
                            args=[
                                {"y": raw_y + est_raw_y + (true_raw_y if has_true else [])},
                                {
                                    "yaxis.title.text": "Uncertainty (std dev)",
                                    "yaxis2.title.text": "Parameter estimate",
                                },
                            ],
                        ),
                        dict(
                            label="Normalized (÷ initial)",
                            method="update",
                            args=[
                                {"y": norm_y + est_norm_y + (true_norm_y if has_true else [])},
                                {
                                    "yaxis.title.text": "Relative uncertainty (u / u₀)",
                                    "yaxis2.title.text": "Relative estimate (val / val₀)",
                                },
                            ],
                        ),
                    ],
                )
            ],
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path, include_mathjax="cdn")

    def plot_correlation_heatmap_animation(
        self,
        correlation_history: list[np.ndarray],
        param_names: list[str],
        out_path: Path,
        *,
        true_params: Mapping[str, float] | None = None,
    ) -> None:
        """Animate correlation matrix evolution as a heatmap (Option A).

        Shows how parameter correlations develop and stabilize over inference steps.
        Values range from -1 (red, negative correlation) to +1 (blue, positive correlation).
        """
        if not correlation_history or len(param_names) < 2:
            return

        n_steps = len(correlation_history)
        len(param_names)

        # Subsample if too many steps
        step_indices = list(range(n_steps))
        if n_steps > 100:
            step_indices = [int(x) for x in np.linspace(0, n_steps - 1, 100)]
            correlation_history = [correlation_history[i] for i in step_indices]
            n_steps = len(step_indices)

        # Initial frame
        initial_corr = correlation_history[0]

        fig = go.Figure(
            data=go.Heatmap(
                z=initial_corr,
                x=param_names,
                y=param_names,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                colorbar=dict(title="Correlation"),
                text=np.round(initial_corr, 2),
                texttemplate="%{text:.2f}",
                textfont=dict(size=10),
            ),
            layout=go.Layout(
                title="Parameter Correlation Evolution",
                xaxis=dict(title="Parameter", tickangle=45),
                yaxis=dict(title="Parameter", autorange="reversed"),
                template="plotly_white",
            ),
        )

        # Build frames
        frames = []
        for step_idx, corr in enumerate(correlation_history):
            frames.append(
                go.Frame(
                    data=[
                        go.Heatmap(
                            z=corr,
                            x=param_names,
                            y=param_names,
                            zmin=-1,
                            zmax=1,
                            colorscale="RdBu",
                            text=np.round(corr, 2),
                            texttemplate="%{text:.2f}",
                        )
                    ],
                    name=str(step_idx),
                    layout=dict(title=f"Parameter Correlation Evolution — Step {step_indices[step_idx]}"),
                )
            )

        # Animation controls
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0.1,
                    y=-0.15,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 50},
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
                )
            ],
            sliders=[
                dict(
                    active=0,
                    pad={"t": 30, "b": 10},
                    currentvalue={"prefix": "Step: "},
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(i)],
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 50},
                                },
                            ],
                            label=str(step_indices[i]),
                        )
                        for i in range(n_steps)
                    ],
                )
            ],
        )

        # Set frames directly on figure (not via update_layout - frames is a figure property)
        fig.frames = frames

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path, include_mathjax="cdn")

    def plot_covariance_ellipses(  # noqa: C901
        self,
        covariance_history: list[np.ndarray],
        param_names: list[str],
        pairs: list[tuple[int, int]],
        out_path: Path,
        *,
        means_history: list[dict[str, float]] | None = None,
        true_params: Mapping[str, float] | None = None,
    ) -> None:
        """Animate 2D covariance ellipses for parameter pairs (Option C).

        Shows how uncertainty regions shrink toward true parameter values.
        Each subplot shows a 2σ ellipse. If means_history is provided,
        ellipses are centered at the estimated means (showing jitter).
        """
        if not covariance_history or not pairs:
            return

        # Define scaling factors for physical parameters to more natural units
        param_scales = {
            "frequency": 1e9,
            "linewidth": 1e6,
            "split": 1e6,
            "fwhm_total": 1e6,
            "fwhm_lorentz": 1e6,
            "fwhm_gauss": 1e6,
        }
        param_units = {
            "frequency": " (GHz)",
            "linewidth": " (MHz)",
            "split": " (MHz)",
            "fwhm_total": " (MHz)",
            "fwhm_lorentz": " (MHz)",
            "fwhm_gauss": " (MHz)",
        }

        # Create scale matrix
        scales = np.array([param_scales.get(p, 1.0) for p in param_names], dtype=float)
        D = np.diag(1.0 / scales)  # noqa: N806
        covariance_history = [D @ cov @ D for cov in covariance_history]

        param_names_scaled = [f"{p}{param_units.get(p, '')}" for p in param_names]

        if means_history is not None:
            means_history_scaled = []
            for d in means_history:
                d_scaled = {}
                for idx, p in enumerate(param_names):
                    d_scaled[param_names_scaled[idx]] = d.get(p, 0.0) / scales[idx]
                means_history_scaled.append(d_scaled)
            means_history = means_history_scaled

        if true_params is not None:
            true_params_scaled = {}
            for idx, p in enumerate(param_names):
                val = true_params.get(p)
                if val is not None:
                    true_params_scaled[param_names_scaled[idx]] = val / scales[idx]
            true_params = true_params_scaled

        param_names = param_names_scaled

        n_steps = len(covariance_history)

        # Subsample if too many steps
        step_indices = list(range(n_steps))
        if n_steps > 150:
            step_indices = [int(x) for x in np.linspace(0, n_steps - 1, 150)]
            covariance_history = [covariance_history[i] for i in step_indices]
            if means_history:
                means_history = [means_history[i] for i in step_indices]
            n_steps = len(step_indices)

        n_pairs = len(pairs)
        fig = make_subplots(
            rows=1,
            cols=n_pairs,
            subplot_titles=[f"{param_names[i]} vs {param_names[j]}" for i, j in pairs],
        )

        # Colors for progression (early = light, late = dark)
        colors = [f"rgba(0, 100, 200, {0.3 + 0.7 * i / max(1, n_steps - 1)})" for i in range(n_steps)]

        # Build all frames upfront
        frames = []
        for step_idx in range(n_steps):
            frame_data = []

            for pair_idx, (i, j) in enumerate(pairs):
                # Get 2x2 covariance for this pair at this step
                cov_full = covariance_history[step_idx]
                cov_2d = np.array(
                    [
                        [cov_full[i, i], cov_full[i, j]],
                        [cov_full[j, i], cov_full[j, j]],
                    ]
                )

                # Extremely robust eigensystem via Normalized Correlation Matrix
                # Scale physical variance values
                std_i = np.sqrt(max(cov_2d[0, 0], 1e-30))
                std_j = np.sqrt(max(cov_2d[1, 1], 1e-30))

                # Correlation coefficient rho
                rho = cov_2d[0, 1] / (std_i * std_j)
                if not np.isfinite(rho):
                    rho = 0.0
                rho = np.clip(rho, -0.9999, 0.9999)  # Avoid singularity

                # Stable Correlation Matrix
                corr_2d = np.array([[1.0, rho], [rho, 1.0]])

                # Eigensystem on normalized correlation matrix
                eigvals, eigvecs = np.linalg.eigh(corr_2d)

                # Generate 2-sigma ellipse points on normalized scale
                theta = np.linspace(0, 2 * np.pi, 100)
                a, b = 2 * np.sqrt(np.maximum(eigvals, 0.0))

                ellipse_x = a * np.cos(theta)
                ellipse_y = b * np.sin(theta)
                rot = eigvecs
                points_corr = np.column_stack([ellipse_x, ellipse_y]) @ rot.T

                # Map back to physical units
                points = np.zeros_like(points_corr)
                points[:, 0] = points_corr[:, 0] * std_i
                points[:, 1] = points_corr[:, 1] * std_j

                # Center at means if available
                if means_history:
                    step_means = means_history[step_idx]
                    m_i = step_means.get(param_names[i], 0.0)
                    m_j = step_means.get(param_names[j], 0.0)
                    points[:, 0] += m_i
                    points[:, 1] += m_j

                frame_data.append(
                    go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode="lines",
                        line=dict(color=colors[step_idx], width=3),
                        fill="toself",
                        fillcolor=colors[step_idx],
                        opacity=0.6,
                        name=f"Step {step_indices[step_idx]}",
                        showlegend=(pair_idx == 0),
                    )
                )

                # Add true value cross
                if true_params:
                    t_i = true_params.get(param_names[i])
                    t_j = true_params.get(param_names[j])
                    if t_i is not None and t_j is not None:
                        frame_data.append(
                            go.Scatter(
                                x=[t_i],
                                y=[t_j],
                                mode="markers",
                                marker=dict(symbol="x", color="red", size=10),
                                name="True",
                                showlegend=(pair_idx == 0 and step_idx == 0),
                            )
                        )

            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(step_idx),
                    layout=dict(
                        title=f"Covariance Ellipse Evolution — Step {step_indices[step_idx]}",
                    ),
                )
            )

        # Initial frame
        fig.add_traces(frames[0].data)

        # Layout
        fig.update_layout(
            template="plotly_white",
            title="Covariance Ellipse Evolution",
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0.1,
                    y=-0.2,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 50},
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
                )
            ],
            sliders=[
                dict(
                    active=0,
                    pad={"t": 30, "b": 30},
                    currentvalue={"prefix": "Step: "},
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(i)],
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 50},
                                },
                            ],
                            label=str(step_indices[i]),
                        )
                        for i in range(n_steps)
                    ],
                )
            ],
        )

        fig.frames = frames

        # Update all subplots with proper axis labels
        for pair_idx, (i, j) in enumerate(pairs):
            col = pair_idx + 1
            fig.update_xaxes(title_text=param_names[i], row=1, col=col)
            fig.update_yaxes(title_text=param_names[j], row=1, col=col)
            # Use data-driven ranges if centered at means
            if not means_history:
                fig.update_yaxes(scaleanchor=f"x{col if col > 1 else ''}", scaleratio=1, row=1, col=col)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path, include_mathjax="cdn")
        self._inject_timeline_bridge(out_path, step_labels=[str(idx) for idx in step_indices])

    def plot_fisher_vs_crlb(
        self,
        fisher_bounds_hist: list[np.ndarray],
        actual_uncertainty_hist: list[Mapping[str, float]],
        param_names: list[str],
        out_path: Path,
        *,
        true_params: Mapping[str, float] | None = None,
    ) -> None:
        """Plot Fisher Information against Cramér-Rao Lower Bound vs actual uncertainty over time.

        Shows how close the Bayesian inference gets to the Cramér-Rao Lower Bound (CRLB).
        The CRLB (green) is the theoretical minimum uncertainty from Fisher Information.
        The actual uncertainty (blue) should approach this bound as measurements accumulate.
        """
        n_steps = len(fisher_bounds_hist)
        n_params = len(param_names)

        # Guard against degenerate FIM (all NaN or all zero -> nothing to plot)
        if n_steps == 0 or n_params == 0:
            return

        # Define scaling factors for physical parameters to more natural units
        param_scales = {
            "frequency": 1e9,
            "linewidth": 1e6,
            "split": 1e6,
            "fwhm_total": 1e6,
            "fwhm_lorentz": 1e6,
            "fwhm_gauss": 1e6,
        }
        param_units = {
            "frequency": " (GHz)",
            "linewidth": " (MHz)",
            "split": " (MHz)",
            "fwhm_total": " (MHz)",
            "fwhm_lorentz": " (MHz)",
            "fwhm_gauss": " (MHz)",
        }

        scales = np.array([param_scales.get(p, 1.0) for p in param_names], dtype=float)
        fisher_bounds_hist = [fb / scales for fb in fisher_bounds_hist]

        param_names_scaled = [f"{p}{param_units.get(p, '')}" for p in param_names]

        actual_uncertainty_hist_scaled = []
        for d in actual_uncertainty_hist:
            d_scaled = {}
            for idx, p in enumerate(param_names):
                d_scaled[param_names_scaled[idx]] = d.get(p, 0.0) / scales[idx]
            actual_uncertainty_hist_scaled.append(d_scaled)
        actual_uncertainty_hist = actual_uncertainty_hist_scaled

        if true_params is not None:
            true_params_scaled = {}
            for idx, p in enumerate(param_names):
                val = true_params.get(p)
                if val is not None:
                    true_params_scaled[param_names_scaled[idx]] = val / scales[idx]
            true_params = true_params_scaled

        param_names = param_names_scaled

        bounds_stack = np.vstack(fisher_bounds_hist)

        if np.all(np.isnan(bounds_stack)) or np.all(bounds_stack == 0):
            return

        # Create subplots - one per parameter
        fig = make_subplots(
            rows=(n_params + 1) // 2,
            cols=2,
            subplot_titles=[f"{p}" for p in param_names],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        steps = list(range(n_steps))

        # Add traces for each parameter
        for i, param in enumerate(param_names):
            row = i // 2 + 1
            col = i % 2 + 1

            # Cramér-Rao Lower Bound (theoretical minimum from Fisher Information)
            fisher_vals = [float(fb[i]) for fb in fisher_bounds_hist]
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=fisher_vals,
                    name="CRLB (from Fisher Info)",
                    line=dict(color="green", width=2, dash="dash"),
                    legendgroup="fisher",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

            # Actual uncertainty from SMC
            actual_vals = [float(u.get(param, np.nan)) for u in actual_uncertainty_hist]
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=actual_vals,
                    name="Actual Uncertainty",
                    line=dict(color="blue", width=2),
                    legendgroup="actual",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            title="Fisher Information / CRLB vs Actual Uncertainty",
            xaxis_title="Step",
            yaxis_title="Standard Deviation",
            height=300 * ((n_params + 1) // 2),
            width=1000,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
            ),
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path, include_mathjax="cdn")

    def plot_fisher_crlb_pairs(  # noqa: C901
        self,
        fisher_hist: list[np.ndarray],
        param_names: list[str],
        out_path: Path,
        *,
        steps_to_show: list[int] | None = None,
        confidence: float = 0.95,
        true_params: Mapping[str, float] | None = None,
    ) -> None:
        """Plot Fisher Information CRLB confidence ellipses for parameter pairs.

        Shows 2D joint uncertainty bounds revealing parameter correlations.
        Each ellipse represents the Cramér-Rao Lower Bound covariance for a
        parameter pair. Tight ellipses (circular) indicate uncorrelated, well-
        constrained parameters; elongated ellipses reveal strong correlations.

        Parameters
        ----------
        fisher_hist : list[np.ndarray]
            List of full Fisher Information Matrices (n_params × n_params) at each step.
        param_names : list[str]
            Names of parameters (order matches FIM dimensions).
        out_path : Path
            Output HTML file path.
        steps_to_show : list[int] | None
            Specific steps to visualize. If None, shows start, middle, and end.
        confidence : float
            Confidence level for ellipses (default 0.95 for 95%).
        true_params : Mapping[str, float] | None
            Ground truth values to mark with red crosses.
        """
        n_params = len(param_names)
        if n_params < 2:
            return

        n_steps = len(fisher_hist)

        # Guard against degenerate FIM (all zero matrices -> nothing useful to plot)
        if n_steps == 0 or all(np.all(f == 0) for f in fisher_hist):
            return

        # Define scaling factors for physical parameters to more natural units
        param_scales = {
            "frequency": 1e9,
            "linewidth": 1e6,
            "split": 1e6,
            "fwhm_total": 1e6,
            "fwhm_lorentz": 1e6,
            "fwhm_gauss": 1e6,
        }
        param_units = {
            "frequency": " (GHz)",
            "linewidth": " (MHz)",
            "split": " (MHz)",
            "fwhm_total": " (MHz)",
            "fwhm_lorentz": " (MHz)",
            "fwhm_gauss": " (MHz)",
        }

        scales = np.array([param_scales.get(p, 1.0) for p in param_names], dtype=float)
        D_inv = np.diag(scales)  # noqa: N806
        fisher_hist = [D_inv @ fim @ D_inv for fim in fisher_hist]

        param_names_scaled = [f"{p}{param_units.get(p, '')}" for p in param_names]

        if true_params is not None:
            true_params_scaled = {}
            for idx, p in enumerate(param_names):
                val = true_params.get(p)
                if val is not None:
                    true_params_scaled[param_names_scaled[idx]] = val / scales[idx]
            true_params = true_params_scaled

        param_names = param_names_scaled

        # Subsample if too many steps for performance
        step_indices = list(range(n_steps))
        if n_steps > 100:
            step_indices = [int(x) for x in np.linspace(0, n_steps - 1, 100)]
            fisher_hist = [fisher_hist[i] for i in step_indices]
            n_steps = len(step_indices)

        # Generate all unique pairs
        pairs = [(i, j) for i in range(n_params) for j in range(i + 1, n_params)]
        n_pairs = len(pairs)
        if n_pairs == 0:
            return

        # Create a compact grid layout: n_rows x n_cols
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols

        subplot_titles = [f"{param_names[j]} vs {param_names[i]}" for i, j in pairs]

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        # Chi-squared scale factor for confidence ellipse
        # For 2D: r^2 = chi2.ppf(confidence, 2)
        import scipy.stats as stats

        chi2_scale = np.sqrt(stats.chi2.ppf(confidence, 2))

        # Build animation frames showing ellipse evolution over time
        # Each subplot needs axis references: xaxis="x", xaxis2, etc.
        def get_axis_refs(row, col):
            plot_idx = (row - 1) * n_cols + col
            xaxis = f"x{plot_idx}" if plot_idx > 1 else "x"
            yaxis = f"y{plot_idx}" if plot_idx > 1 else "y"
            return xaxis, yaxis

        frames = []
        for step_idx in range(n_steps):
            frame_data = []

            for pair_idx, (i, j) in enumerate(pairs):
                row = (pair_idx // n_cols) + 1
                col = (pair_idx % n_cols) + 1
                xaxis, yaxis = get_axis_refs(row, col)

                fim = fisher_hist[step_idx]
                if fim is None or fim.shape != (n_params, n_params):
                    continue

                # Extract 2x2 submatrix and invert to get CRLB covariance
                sub_indices = [i, j]
                sub_fim = fim[np.ix_(sub_indices, sub_indices)]

                try:
                    # Add small ridge for numerical stability
                    sub_cov = np.linalg.inv(sub_fim + np.eye(2) * 1e-8)
                except np.linalg.LinAlgError:
                    continue

                # Generate ellipse points
                theta = np.linspace(0, 2 * np.pi, 100)
                eigvals, eigvecs = np.linalg.eigh(sub_cov)
                eigvals = np.maximum(eigvals, 0)

                # Scale by chi-squared factor for confidence level
                a = chi2_scale * np.sqrt(eigvals[0])
                b = chi2_scale * np.sqrt(eigvals[1])

                x_std = a * np.cos(theta)
                y_std = b * np.sin(theta)
                ellipse_points = eigvecs @ np.vstack([x_std, y_std])
                x_ellipse = ellipse_points[0, :]
                y_ellipse = ellipse_points[1, :]

                frame_data.append(
                    go.Scatter(
                        x=x_ellipse,
                        y=y_ellipse,
                        xaxis=xaxis,
                        yaxis=yaxis,
                        mode="lines",
                        line=dict(color="#2d8f2d", width=2),
                        name="CRLB Ellipse",
                        showlegend=(i == 0 and j == 1),
                        legendgroup="ellipse",
                    )
                )

                # Mark true parameters if provided (static across frames)
                if true_params is not None:
                    true_i = true_params.get(param_names[i])
                    true_j = true_params.get(param_names[j])
                    if true_i is not None and true_j is not None:
                        frame_data.append(
                            go.Scatter(
                                x=[true_i],
                                y=[true_j],
                                xaxis=xaxis,
                                yaxis=yaxis,
                                mode="markers",
                                marker=dict(color="red", size=10, symbol="x"),
                                name="True Value",
                                showlegend=(i == 0 and j == 1),
                                legendgroup="true",
                            )
                        )

            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(step_idx),
                )
            )

        # Add initial frame data to proper subplots
        if frames:
            trace_idx = 0
            for pair_idx, (i, j) in enumerate(pairs):
                row = (pair_idx // n_cols) + 1
                col = (pair_idx % n_cols) + 1
                # Add ellipse trace
                if trace_idx < len(frames[0].data):
                    fig.add_trace(frames[0].data[trace_idx], row=row, col=col)
                    trace_idx += 1
                    # Add true value marker if present
                    if true_params is not None:
                        true_i = true_params.get(param_names[i])
                        true_j = true_params.get(param_names[j])
                        if true_i is not None and true_j is not None and trace_idx < len(frames[0].data):
                            fig.add_trace(frames[0].data[trace_idx], row=row, col=col)
                            trace_idx += 1

        # Set axis titles per subplot
        for pair_idx, (i, j) in enumerate(pairs):
            row = (pair_idx // n_cols) + 1
            col = (pair_idx % n_cols) + 1
            fig.update_xaxes(title_text=param_names[j], autorange=True, row=row, col=col)
            fig.update_yaxes(title_text=param_names[i], autorange=True, row=row, col=col)

        # Layout with animation controls - autosize to avoid scrollbars
        fig.update_layout(
            height=max(400, 300 * n_rows),
            autosize=True,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.02,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(l=60, r=30, t=40, b=60),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0.1,
                    y=-0.15,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 150, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 50},
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
                )
            ],
            sliders=[
                dict(
                    active=0,
                    pad={"t": 30, "b": 10},
                    currentvalue={"prefix": "Step: "},
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(i)],
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 50},
                                },
                            ],
                            label=str(step_indices[i]),
                        )
                        for i in range(n_steps)
                    ],
                )
            ],
        )

        fig.frames = frames

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path, include_mathjax="cdn")

    def _add_param_convergence_trace(
        self,
        fig: go.Figure,
        param: str,
        row: int,
        n_steps: int,
        conv_metrics: list[dict],
        steps: list[int],
        convergence_threshold: float,
    ) -> None:
        """Helper method to add convergence traces and annotations for a single parameter."""
        # Uncertainty values
        uncertainties = [float(cm["uncertainties"].get(param, np.nan)) for cm in conv_metrics]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=uncertainties,
                name=f"{param} σ",
                line=dict(width=2),
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # Threshold line
        absolute_threshold = PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS.get(param)
        if absolute_threshold is not None:
            if param == "frequency":
                y_thresh = absolute_threshold / 1000.0
                thresh_label = f"{y_thresh:.0f} KHz threshold"
            else:
                y_thresh = absolute_threshold
                thresh_label = f"threshold ({absolute_threshold:g})"
            fig.add_hline(
                y=y_thresh,
                line=dict(color="red", dash="dash", width=1.5),
                annotation_text=thresh_label,
                row=row,
                col=1,
            )

        else:
            fig.add_hline(
                y=convergence_threshold,
                line=dict(color="red", dash="dash", width=1.5),
                annotation_text=f"threshold ({convergence_threshold})",
                row=row,
                col=1,
            )

            # Reference line at y=1.0: uncertainty equal to full bound width
            fig.add_hline(
                y=1.0,
                line=dict(color="gray", dash="dot", width=1),
                annotation_text="100 % of bound",
                annotation_font_color="gray",
                row=row,
                col=1,
            )

        # Highlight converged regions
        converged_regions = []
        in_converged = False
        start_idx = 0
        for idx, cm in enumerate(conv_metrics):
            is_conv = cm["converged_params"].get(param, False)
            if is_conv and not in_converged:
                in_converged = True
                start_idx = idx
            elif not is_conv and in_converged:
                in_converged = False
                converged_regions.append((start_idx, idx - 1))
        if in_converged:
            converged_regions.append((start_idx, n_steps - 1))

        for start, end in converged_regions:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="green",
                opacity=0.1,
                layer="below",
                line_width=0,
                row=row,
                col=1,
            )

        # Status badge on the right
        is_conv_end = conv_metrics[-1]["converged_params"].get(param, False)
        status_text = "CONVERGED" if is_conv_end else "NOT CONVERGED"
        status_color = "#2ecc71" if is_conv_end else "#e74c3c"

        fig.add_annotation(
            text=f"<b>{status_text}</b>",
            xref=f"x{row if row > 1 else ''} domain",
            yref=f"y{row if row > 1 else ''} domain",
            x=0.98,
            y=0.95,
            showarrow=False,
            font=dict(color="white", size=9),
            xanchor="right",
            yanchor="top",
            bgcolor=status_color,
            bordercolor=status_color,
            borderwidth=1,
            borderpad=4,
        )

    def plot_convergence_metrics(
        self,
        conv_metrics: list[dict],
        param_names: list[str],
        convergence_threshold: float,
        convergence_patience: int,
        out_path: Path,
        param_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Plot convergence metrics showing per-parameter uncertainty vs threshold.

        Creates an interactive visualization with:
        - Per-parameter uncertainty evolution with threshold line
        - Convergence streak counter
        - Convergence achieved indicator
        - Parameter bounds (if provided) shown in subplot titles
        """
        n_steps = len(conv_metrics)
        n_params = len(param_names)

        # Define scaling factors for physical parameters to more natural units
        param_scales = {
            "frequency": 1e9,
            "linewidth": 1e6,
            "split": 1e6,
            "fwhm_total": 1e6,
            "fwhm_lorentz": 1e6,
            "fwhm_gauss": 1e6,
        }
        param_units = {
            "frequency": " GHz",
            "linewidth": " MHz",
            "split": " MHz",
            "fwhm_total": " MHz",
            "fwhm_lorentz": " MHz",
            "fwhm_gauss": " MHz",
        }

        # Scale frequency absolute uncertainty from Hz to KHz for conv_metrics
        conv_metrics_scaled = []
        for cm in conv_metrics:
            cm_scaled = cm.copy()
            cm_scaled["uncertainties"] = cm["uncertainties"].copy()
            if "frequency" in cm["uncertainties"]:
                cm_scaled["uncertainties"]["frequency"] = cm["uncertainties"]["frequency"] / 1000.0
            conv_metrics_scaled.append(cm_scaled)
        conv_metrics = conv_metrics_scaled

        def _subplot_title(p: str) -> str:
            base = "frequency (absolute uncertainty, KHz)" if p == "frequency" else f"{p} (relative uncertainty)"
            if param_bounds and p in param_bounds:
                lo, hi = param_bounds[p]
                scale = param_scales.get(p, 1.0)
                unit = param_units.get(p, "")
                bounds_str = f"bounds: [{lo / scale:.4g}, {hi / scale:.4g}]{unit}"
                width_str = f"width={(hi - lo) / scale:.4g}{unit}"
                base += f"<br><sup>{bounds_str} ({width_str})</sup>"
            return base

        # Create subplots - one row per parameter, plus one for convergence streak
        fig = make_subplots(
            rows=n_params + 1,
            cols=1,
            subplot_titles=[_subplot_title(p) for p in param_names] + ["Convergence Streak"],
            vertical_spacing=0.08,
            shared_xaxes=True,
        )

        steps = list(range(n_steps))
        achieved = [bool(cm["convergence_achieved"]) for cm in conv_metrics]
        global_conv_idx = next((i for i, a in enumerate(achieved) if a), None)

        # Plot per-parameter uncertainties with threshold line
        for i, param in enumerate(param_names):
            row = i + 1
            self._add_param_convergence_trace(
                fig,
                param,
                row,
                n_steps,
                conv_metrics,
                steps,
                convergence_threshold,
            )

        # Convergence streak plot (bottom subplot)
        streaks = [int(cm["convergence_streak"]) for cm in conv_metrics]
        achieved = [bool(cm["convergence_achieved"]) for cm in conv_metrics]

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=streaks,
                name="Convergence Streak",
                line=dict(color="blue", width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 0, 255, 0.1)",
                showlegend=False,
            ),
            row=n_params + 1,
            col=1,
        )

        # Patience threshold line
        fig.add_hline(
            y=convergence_patience,
            line=dict(color="green", dash="dash", width=2),
            annotation_text=f"patience ({convergence_patience})",
            row=n_params + 1,
            col=1,
        )

        # Highlight where convergence is achieved
        for idx, is_achieved in enumerate(achieved):
            if is_achieved:
                fig.add_vrect(
                    x0=idx,
                    x1=idx + 0.9,
                    fillcolor="green",
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    row=n_params + 1,
                    col=1,
                )

        # Mark global convergence step across all subplots
        if global_conv_idx is not None:
            for r in range(1, n_params + 2):
                fig.add_vline(
                    x=global_conv_idx,
                    line=dict(color="#2ecc71", dash="dash", width=2),
                    annotation_text="Converged" if r == 1 else None,
                    annotation_position="top left",
                    row=r,
                    col=1,
                )

        fig.update_layout(
            title="Bayesian Convergence Metrics",
            xaxis_title="Step",
            height=250 * (n_params + 1),
            width=1000,
            margin=dict(r=150),
            template="plotly_white",
            showlegend=False,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path, include_mathjax="cdn")
