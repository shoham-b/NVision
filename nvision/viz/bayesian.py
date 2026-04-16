from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _get_signal_formula(model: Any) -> str:
    """Return a LaTeX string with the full signal formula for the given model.

    Plotly renders LaTeX in titles when wrapped in $...$.
    """
    # Unwrap UnitCubeSignalModel to get the inner physical model
    inner = getattr(model, "inner", model)
    inner_name = type(inner).__name__

    if "OnePeakLorentzian" in inner_name:
        return (
            r"$S(f) = B - A \frac{\omega^2}{(f - f_0)^2 + \omega^2}$"
        )
    if "NVCenterLorentzian" in inner_name:
        return (
            r"$S(f) = B - \frac{A}{k} \frac{\omega^2}{(f-f_0+\Delta)^2 + \omega^2} - A \frac{\omega^2}{(f-f_0)^2 + \omega^2} - A \cdot k \cdot \frac{\omega^2}{(f-f_0-\Delta)^2 + \omega^2}$"
        )
    if "VoigtZeeman" in inner_name:
        return (
            r"$S(f) = B - \frac{A}{k}(L*G)(f\!-\!f_0\!+\!\Delta) - A(L*G)(f\!-\!f_0) - Ak(L*G)(f\!-\!f_0\!-\!\Delta)$, "
            r"$(L*G)(x) = \int_{-\infty}^{\infty}\!\frac{\eta W/2\pi}{t^2+(\eta W/2)^2} \cdot "
            r"\frac{\exp\!\left[-\frac{(x-t)^2}{2((1-\eta)W/2\sqrt{2\ln 2})^2}\right]}{\frac{(1-\eta)W}{2\sqrt{2\ln 2}}\sqrt{2\pi}}\,dt$"
        )
    if "NVCenterVoigt" in inner_name:
        return (
            r"$S(f) = B - \frac{A}{k}(L*G)(f\!-\!f_0\!+\!\Delta) - A(L*G)(f\!-\!f_0) - Ak(L*G)(f\!-\!f_0\!-\!\Delta)$, "
            r"$(L*G)(x) = \int_{-\infty}^{\infty}\!\frac{\eta W/2\pi}{t^2+(\eta W/2)^2} \cdot "
            r"\frac{\exp\!\left[-\frac{(x-t)^2}{2((1-\eta)W/2\sqrt{2\ln 2})^2}\right]}{\frac{(1-\eta)W}{2\sqrt{2\ln 2}}\sqrt{2\pi}}\,dt$"
        )
    if "OnePeakVoigt" in inner_name:
        return (
            r"$S(f) = B - A(L*G)(f\!-\!f_0)$, $(L*G)(x) = \int_{-\infty}^{\infty}\!"
            r"\frac{\gamma_L/2\pi}{t^2+(\gamma_L/2)^2} \cdot "
            r"\frac{\exp\!\left[-\frac{(x-t)^2}{2(\gamma_G/2\sqrt{2\ln 2})^2}\right]}{\frac{\gamma_G}{2\sqrt{2\ln 2}}\sqrt{2\pi}}\,dt$"
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
    row: int,
    col: int,
    window: tuple[float, float],
    full_domain: tuple[float, float] | None,
    annotation_text: str = "acquisition window",
    zoom_to_window: bool = False,
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
    if zoom_to_window:
        # Focus only on the acquisition window with 10% padding
        pad = (x1 - x0) * 0.1
        fig.update_xaxes(range=[x0 - pad, x1 + pad], row=row, col=col)
    elif full_domain is not None:
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
                zoom_to_window=True,
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
                    label="▶ Play",
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
            height=max(300, 180 * n),
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
        n_params = len(param_names)

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
            subplot_titles=[
                f"{param_names[i]} vs {param_names[j]}" for i, j in pairs
            ],
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
                cov_2d = np.array([
                    [cov_full[i, i], cov_full[i, j]],
                    [cov_full[j, i], cov_full[j, j]],
                ])

                # Generate 2σ ellipse points
                theta = np.linspace(0, 2 * np.pi, 100)
                eigvals, eigvecs = np.linalg.eigh(cov_2d)
                a, b = 2 * np.sqrt(np.maximum(eigvals, 0))

                ellipse_x = a * np.cos(theta)
                ellipse_y = b * np.sin(theta)
                R = eigvecs
                points = np.column_stack([ellipse_x, ellipse_y]) @ R.T

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

    def plot_fisher_bounds_comparison(
        self,
        fisher_bounds_hist: list[np.ndarray],
        actual_uncertainty_hist: list[Mapping[str, float]],
        param_names: list[str],
        out_path: Path,
        *,
        true_params: Mapping[str, float] | None = None,
    ) -> None:
        """Plot Fisher information bounds vs actual SMC uncertainty over time.

        Shows how close the Bayesian inference gets to the Cramér-Rao bound.
        The Fisher bound (green) is the theoretical minimum uncertainty.
        The actual uncertainty (blue) should approach this bound as measurements accumulate.
        """
        n_steps = len(fisher_bounds_hist)
        n_params = len(param_names)

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

            # Fisher bound (theoretical minimum)
            fisher_vals = [float(fb[i]) for fb in fisher_bounds_hist]
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=fisher_vals,
                    name="Fisher Bound (CRB)",
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
            title="Fisher Information Bounds vs Actual Uncertainty",
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
