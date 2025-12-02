from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from nvision.pathutils import ensure_out_dir
from nvision.sim.core import CompositeOverFrequencyNoise, DataBatch


class Viz:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        ensure_out_dir(self.out_dir)

    def plot_experiment_summary(self, df: pl.DataFrame) -> Sequence[Path]:
        """Plot RMSE by (noise, strategy) for each generator in experiment results.

        Returns list of saved image paths.
        """
        paths: list[Path] = []
        if df.height == 0:
            return paths

        # Not yet implemented, return empty list to satisfy types and callers.
        return paths

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

        if has_metrics:
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.15,
                subplot_titles=("Scan Measurements", "Convergence Metrics"),
                row_heights=[0.7, 0.3],
            )
        else:
            fig = go.Figure()

        # True signal
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", name="true signal", line=dict(color="blue")),
            row=1 if has_metrics else None,
            col=1 if has_metrics else None,
        )

        # Simulated noisy curve (e.g., over-frequency noise)
        if over_frequency_noise is not None:
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
                row=1 if has_metrics else None,
                col=1 if has_metrics else None,
            )

        # Overlay samples
        if history.height > 0:
            steps = list(range(history.height))
            xs_s = history.get_column("x").to_list() if "x" in history.columns else []
            ys_s = (
                history.get_column("signal_values").to_list()
                if "signal_values" in history.columns
                else []
            )
            # Noisy measurements
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
                        colorbar=dict(title="step", len=0.6, y=0.8)
                        if has_metrics
                        else dict(title="step"),
                        line=dict(width=0.5, color="black"),
                    ),
                ),
                row=1 if has_metrics else None,
                col=1 if has_metrics else None,
            )

            # Plot metrics if available
            if has_metrics:
                steps_arr = np.array(steps)
                if "entropy" in history.columns:
                    entropy = history.get_column("entropy").to_list()
                    # Filter out None/NaN
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

        layout_args = dict(
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

        fig.write_html(out_path.as_posix(), include_plotlyjs="inline")
        return out_path

    def _plot_pivot_from_polars(
        self, pivot_pl: pl.DataFrame, title: str, ylabel: str, out_path: Path
    ) -> Path:
        """Plot a bar chart from a polars pivoted dataframe as an interactive plot.

        Expected format: first column is the index (noise), remaining columns are strategies.
        """
        cols = pivot_pl.columns
        if not cols:
            return out_path
        index_col = cols[0]
        strategies = [c for c in cols if c != index_col]
        noises = pivot_pl.get_column(index_col).to_list()
        if not strategies:
            return out_path

        fig = go.Figure()
        for s in strategies:
            fig.add_trace(go.Bar(name=s, x=noises, y=pivot_pl.get_column(s).to_list()))

        fig.update_layout(
            barmode="group",
            title=title,
            xaxis_title="Noise",
            yaxis_title=ylabel,
            template="plotly_white",
        )
        fig.write_html(out_path.as_posix(), include_plotlyjs="cdn")
        return out_path

    def _plot_single_peak(self, sub: pl.DataFrame, gen: str, paths: list[dict]) -> None:
        """Handle plotting for a single-peak generator subset."""
        for col, title in zip(["abs_err_x", "uncert"], ["Abs Error", "Uncertainty"], strict=False):
            piv_pl = (
                sub.select(["noise", "strategy", col])
                .group_by(["noise", "strategy"])
                .agg(pl.col(col).mean())
                .pivot(values=col, index="noise", columns="strategy")
                .sort("noise")
            )
            path = self.out_dir / f"locator_summary_single_{gen}_{col}.html"
            self._plot_pivot_from_polars(piv_pl, f"{title} — {gen}", col, path)
            paths.append(
                {
                    "type": "summary",
                    "kind": "single_peak",
                    "generator": gen,
                    "metric": col,
                    "path": path,
                }
            )

    def _plot_double_peak(self, sub: pl.DataFrame, gen: str, paths: list[dict]) -> None:
        """Handle plotting for a two-peak generator subset."""
        metric_pairs = zip(
            ["pair_rmse", "uncert_sep"],
            ["Pair RMSE", "Uncertainty (sep)"],
            strict=False,
        )
        for col, title in metric_pairs:
            piv_pl = (
                sub.select(["noise", "strategy", col])
                .group_by(["noise", "strategy"])
                .agg(pl.col(col).mean())
                .pivot(values=col, index="noise", columns="strategy")
                .sort("noise")
            )
            path = self.out_dir / f"locator_summary_double_{gen}_{col}.html"
            self._plot_pivot_from_polars(piv_pl, f"{title} — {gen}", col, path)
            paths.append(
                {
                    "type": "summary",
                    "kind": "double_peak",
                    "generator": gen,
                    "metric": col,
                    "path": path,
                }
            )

    def _plot_nv_center(self, sub: pl.DataFrame, gen: str, paths: list[dict]) -> None:
        """Handle plotting for NV center generator subset."""
        # Plot errors for each peak position if available
        for col, title in zip(
            [
                "abs_err_x1",
                "abs_err_x2",
                "abs_err_x3",
                "uncert_pos",
                "final_entropy",
                "final_max_prob",
            ],
            [
                "Abs Error X1",
                "Abs Error X2",
                "Abs Error X3",
                "Uncertainty (pos)",
                "Final Entropy",
                "Final Max Prob",
            ],
            strict=False,
        ):
            if col not in sub.columns:
                continue
            piv_pl = (
                sub.select(["noise", "strategy", col])
                .group_by(["noise", "strategy"])
                .agg(pl.col(col).mean())
                .pivot(values=col, index="noise", columns="strategy")
                .sort("noise")
            )
            path = self.out_dir / f"locator_summary_nvcenter_{gen}_{col}.html"
            self._plot_pivot_from_polars(piv_pl, f"{title} — {gen}", col, path)
            paths.append(
                {
                    "type": "summary",
                    "kind": "nv_center",
                    "generator": gen,
                    "metric": col,
                    "path": path,
                }
            )

    def plot_locator_summary(self, df: pl.DataFrame) -> Sequence[dict]:
        """Create comparison plots for locator sweeps."""
        paths: list[dict] = []
        if df.height == 0:
            return paths

        plot_configs = [
            ("OnePeak", ["abs_err_x", "uncert"], self._plot_single_peak),
            ("TwoPeak", ["pair_rmse", "uncert_sep"], self._plot_double_peak),
            ("NVCenter", [], self._plot_nv_center),
        ]

        for name, required_cols, plot_func in plot_configs:
            subset = df.filter(pl.col("generator").str.contains(name))
            if subset.height > 0 and (
                not required_cols or all(col in subset.columns for col in required_cols)
            ):
                for gen in sorted(set(subset.get_column("generator").to_list())):
                    sub = subset.filter(pl.col("generator") == gen)
                    if sub.height > 0:
                        plot_func(sub, gen, paths)

        return paths

    def plot_posterior_animation(
        self,
        posterior_history: list[np.ndarray],
        freq_grid: np.ndarray,
        out_path: Path,
        model_history: list[np.ndarray] | None = None,
    ) -> Path:
        """Create an interactive Plotly animation of the posterior distribution evolution."""
        ensure_out_dir(out_path.parent)

        if not posterior_history:
            return out_path

        # Create figure
        fig = go.Figure()

        # Add initial trace (step 0)
        fig.add_trace(
            go.Scatter(
                x=freq_grid,
                y=posterior_history[0],
                mode="lines",
                name="Posterior",
                line=dict(color="blue"),
            )
        )

        if model_history:
            # Normalize model for visualization scale if needed, or plot on secondary y-axis?
            # Posterior is a probability density (integral=1). Model is signal intensity (e.g. 0.99 to 1.01).
            # They have vastly different scales. We should use a secondary y-axis.
            fig.add_trace(
                go.Scatter(
                    x=freq_grid,
                    y=model_history[0],
                    mode="lines",
                    name="Best Fit Model",
                    line=dict(color="red", dash="dash"),
                    yaxis="y2",
                )
            )

        # Create frames for animation
        frames = []
        for i, posterior in enumerate(posterior_history):
            frame_data = [
                go.Scatter(
                    x=freq_grid,
                    y=posterior,
                    mode="lines",
                    line=dict(color="blue"),
                )
            ]
            if model_history and i < len(model_history):
                frame_data.append(
                    go.Scatter(
                        x=freq_grid,
                        y=model_history[i],
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        yaxis="y2",
                    )
                )

            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(i),
                )
            )

        fig.frames = frames

        # Add slider and buttons
        fig.update_layout(
            title="Posterior Distribution Evolution",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Probability Density",
            yaxis2=dict(
                title="Signal Intensity",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            template="plotly_white",
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.05,
                    x=1.15,
                    xanchor="right",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=100, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue=dict(
                        font=dict(size=16),
                        prefix="Step: ",
                        visible=True,
                        xanchor="right",
                    ),
                    transition=dict(duration=0, easing="cubic-in-out"),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            args=[
                                [str(k)],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=str(k),
                            method="animate",
                        )
                        for k in range(len(posterior_history))
                    ],
                )
            ],
        )

        fig.write_html(out_path.as_posix(), include_plotlyjs="inline")
        return out_path

    def plot_parameter_convergence(
        self,
        parameter_history: list[dict[str, float]],
        out_path: Path,
    ) -> Path:
        """Plot the convergence of parameters with uncertainty bands."""
        ensure_out_dir(out_path.parent)

        if not parameter_history:
            return out_path

        # Identify parameters to plot (exclude frequency, entropy, max_prob, uncertainty)
        # We want things like linewidth, amplitude, background, split, etc.
        all_keys = set().union(*(d.keys() for d in parameter_history))
        exclude = {"frequency", "entropy", "max_prob", "uncertainty", "uncert", "uncert_pos"}
        param_keys = [k for k in all_keys if k not in exclude and not k.endswith("_uncertainty")]

        if not param_keys:
            return out_path

        from plotly.subplots import make_subplots

        n_params = len(param_keys)
        # Calculate rows/cols
        cols = 2
        rows = (n_params + 1) // 2

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=param_keys, vertical_spacing=0.1)

        steps = list(range(len(parameter_history)))

        for i, key in enumerate(sorted(param_keys)):
            row = i // cols + 1
            col = i % cols + 1

            values = [d.get(key, float("nan")) for d in parameter_history]
            uncerts = [d.get(f"{key}_uncertainty", 0.0) for d in parameter_history]

            # Upper and lower bounds
            upper = [v + u for v, u in zip(values, uncerts, strict=False)]
            lower = [v - u for v, u in zip(values, uncerts, strict=False)]

            # Plot value
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=values,
                    mode="lines+markers",
                    name=key,
                    line=dict(color="blue"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Plot uncertainty band if any uncertainty exists
            if any(u > 0 for u in uncerts):
                fig.add_trace(
                    go.Scatter(
                        x=steps + steps[::-1],
                        y=upper + lower[::-1],
                        fill="toself",
                        fillcolor="rgba(0,0,255,0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            title="Parameter Convergence",
            height=300 * rows,
            template="plotly_white",
        )

        fig.write_html(out_path.as_posix(), include_plotlyjs="inline")
        return out_path
