from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class SubplotOptions:
    row: int
    col: int
    annotation_text: str = "acquisition window"
    zoom_to_window: bool = False


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
    if descriptions is None:
        return param
    desc = descriptions.get(param, "")
    if desc:
        return f"<b>{param}</b><br><span style='font-size:10px;color:#666;'>{desc}</span>"
    return param


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
            nbinsx=80,
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
            yaxis_layout = dict(title="Probability Density", automargin=True)
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
            yaxis_layout = dict(title="Probability Density", automargin=True, range=[0, max_prob * 1.1])

        # Speed multipliers for single parameter animation
        speed_options_single = [
            ("0.5×", 200),
            ("1×", 100),
            ("1.5×", 67),
            ("2×", 50),
        ]

        speed_buttons_single = [
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

        fig = go.Figure(
            data=initial_data,
            layout=go.Layout(
                xaxis=dict(title="Frequency / Parameter"),
                yaxis=yaxis_layout,
                margin=dict(l=80, r=40, t=80, b=80),
                title="Posterior Evolution",
                updatemenus=[
                    # Play/Pause toggle - positioned below slider
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
                    # Speed controls - positioned below slider, right of play/pause
                    dict(
                        type="buttons",
                        direction="left",
                        x=0.18,
                        y=-0.12,
                        xanchor="left",
                        yanchor="top",
                        showactive=True,
                        active=1,  # Default to 1×
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

        subplot_titles = tuple(_build_subplot_title(p, param_descriptions) for p in param_names)
        fig = make_subplots(
            rows=n,
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=min(0.15, 0.6 / max(n, 1)),  # slightly more spacing for rich titles
        )
        for i, tr in enumerate(traces_for_step(step_indices[0]), start=1):
            fig.add_trace(tr, row=i, col=1)
        for i, name in enumerate(param_names, start=1):
            fig.update_yaxes(title_text="density", automargin=True, row=i, col=1)
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
                        annotation = "post-sweep acquisition" if step_idx < 32 else "posterior refocusing"

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
            elif (
                not step_bounds
                and acquisition_window is not None
                and acquisition_param
                and acquisition_param in posterior_inputs_by_param
            ):
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

            # Add refocusing indicator to title (LaTeX formulas render via $...$)
            _formula_suffix = f"  {signal_formula}" if signal_formula else ""
            title_text = f"Posterior evolution (all parameters){_formula_suffix}<br>step {step_idx + 1}/{total_steps}"
            if si in refocusing_steps:
                title_text = title_text.replace("<br>", " [REFOCUSING]<br>", 1)

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
            label = f"[{step_num}]" if si in refocusing_steps else str(step_num)  # Brackets mark refocusing steps

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
                zoom_to_window=True,
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
        speed_buttons = [
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

        fig.update_layout(
            title_text=f"{base_title}<br>step {step_indices[0] + 1}/{total_steps}",
            title_font_size=14,
            title_y=0.98,
            title_yanchor="top",
            template="plotly_white",
            height=max(400, 200 * n),
            margin=dict(l=90, r=40, t=110, b=110),
            updatemenus=[
                # Play/Pause toggle button - positioned below slider, left side
                dict(
                    type="buttons",
                    direction="left",
                    x=0.02,
                    y=-0.12,
                    xanchor="left",
                    yanchor="top",
                    showactive=False,
                    pad={"r": 10, "t": 0},
                    bgcolor="#e6f2ff",
                    bordercolor="#1e90ff",
                    borderwidth=2,
                    font=dict(size=14, color="#005b96"),
                    buttons=[
                        play_pause_button(True),
                        play_pause_button(False),
                    ],
                ),
                # Speed control buttons - positioned below slider, right of play/pause
                dict(
                    type="buttons",
                    direction="left",
                    x=0.18,
                    y=-0.12,
                    xanchor="left",
                    yanchor="top",
                    showactive=True,
                    active=1,  # Default to 1× speed
                    pad={"r": 10, "t": 0},
                    buttons=speed_buttons,
                ),
            ],
            sliders=[
                dict(
                    active=0,
                    pad={"t": 30, "b": 10},
                    currentvalue={"prefix": "Measurement step: "},
                    steps=slider_steps,
                )
            ],
        )
        fig.frames = frames

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path, include_mathjax="cdn")

    def plot_parameter_convergence(
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
        if true_params:
            for key in keys:
                tv = true_params.get(key)
                if tv is not None and math.isfinite(float(tv)):
                    fig.add_hline(
                        y=float(tv),
                        line_dash="dot",
                        line_color="green",
                        annotation_text=f"true {key}",
                        annotation_position="right",
                        yref="y2" if estimates_history else "y",
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
                    y=1.12,
                    yanchor="bottom",
                    buttons=[
                        dict(
                            label="Absolute",
                            method="update",
                            args=[
                                {"y": raw_y + est_raw_y},
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
                                {"y": norm_y + est_norm_y},
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

    def plot_covariance_ellipses(
        self,
        covariance_history: list[np.ndarray],
        param_names: list[str],
        pairs: list[tuple[int, int]],
        out_path: Path,
        *,
        true_params: Mapping[str, float] | None = None,
    ) -> None:
        """Animate 2D covariance ellipses for parameter pairs (Option C).

        Shows how uncertainty regions shrink toward true parameter values.
        Each subplot shows a 2σ ellipse shrinking as inference progresses.
        """
        if not covariance_history or not pairs:
            return

        n_steps = len(covariance_history)

        # Subsample if too many steps
        step_indices = list(range(n_steps))
        if n_steps > 100:
            step_indices = [int(x) for x in np.linspace(0, n_steps - 1, 100)]
            covariance_history = [covariance_history[i] for i in step_indices]
            n_steps = len(step_indices)

        from plotly.subplots import make_subplots

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

                # Generate 2σ ellipse points
                theta = np.linspace(0, 2 * np.pi, 100)
                eigvals, eigvecs = np.linalg.eigh(cov_2d)
                a, b = 2 * np.sqrt(np.maximum(eigvals, 0))

                ellipse_x = a * np.cos(theta)
                ellipse_y = b * np.sin(theta)
                rot = eigvecs
                points = np.column_stack([ellipse_x, ellipse_y]) @ rot.T

                frame_data.append(
                    go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode="lines",
                        line=dict(color=colors[step_idx], width=2),
                        fill="toself",
                        fillcolor=colors[step_idx],
                        opacity=0.7,
                        name=f"Step {step_indices[step_idx]}",
                        showlegend=(pair_idx == 0),
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
            fig.update_xaxes(title_text=f"Δ{param_names[i]} (deviation)", row=1, col=col)
            fig.update_yaxes(title_text=f"Δ{param_names[j]} (deviation)", row=1, col=col)
            fig.update_yaxes(scaleanchor=f"x{col if col > 1 else ''}", scaleratio=1, row=1, col=col)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path, include_mathjax="cdn")

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

        from plotly.subplots import make_subplots

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

    def plot_convergence_metrics(  # noqa: C901
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

        def _subplot_title(p: str) -> str:
            base = f"{p} (relative uncertainty)"
            if param_bounds and p in param_bounds:
                lo, hi = param_bounds[p]
                base += f"<br><sup>bounds: [{lo:.4g}, {hi:.4g}] (width={hi - lo:.4g})</sup>"
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

        # Plot per-parameter uncertainties with threshold line
        for i, param in enumerate(param_names):
            row = i + 1

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

        fig.update_layout(
            title="Bayesian Convergence Metrics",
            xaxis_title="Step",
            height=250 * (n_params + 1),
            width=900,
            template="plotly_white",
            showlegend=False,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path, include_mathjax="cdn")
