from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from nvision.core.paths import ensure_out_dir
from nvision.sim.core import CompositeOverFrequencyNoise, DataBatch


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
