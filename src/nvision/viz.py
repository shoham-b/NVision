from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from nvision.pathutils import ensure_out_dir
from nvision.sim.core import CompositeOverVoltageNoise, DataBatch


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
        over_voltage_noise: CompositeOverVoltageNoise | None = None,
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

        fig = go.Figure()

        # True signal
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", name="true signal", line=dict(color="blue"))
        )

        # Simulated noisy curve (e.g., over-voltage noise)
        if over_voltage_noise is not None:
            dense_batch = DataBatch.from_arrays(xs, ys, meta={})
            noisy_batch = over_voltage_noise.apply(dense_batch, random.Random(0))
            noisy_values = [float(v) for v in noisy_batch.signal_values]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=noisy_values,
                    mode="lines",
                    name="simulated noisy signal (over-voltage)",
                    line=dict(color="orange", dash="dot"),
                )
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
                        colorbar=dict(title="step"),
                        line=dict(width=0.5, color="black"),
                    ),
                )
            )

        fig.update_layout(
            title="Scan with sampled measurements",
            xaxis_title="frequency",
            yaxis_title="intensity (photon count)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template="plotly_white",
        )

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

    def plot_locator_summary(self, df: pl.DataFrame) -> Sequence[dict]:
        """Create comparison plots for locator sweeps.

        - For single-peak generators (name contains 'OnePeak'): plot abs_err_x
          and uncert by (noise, strategy).
        - For two-peak generators (name contains 'TwoPeak'): plot pair_rmse and
          uncert_sep by (noise, strategy).
        """
        paths: list[dict] = []
        if df.height == 0:
            return paths

        # Single-peak
        singles = df.filter(pl.col("generator").str.contains("OnePeak"))
        if singles.height > 0 and all(col in singles.columns for col in ["abs_err_x", "uncert"]):
            for gen in sorted(set(singles.get_column("generator").to_list())):
                sub = singles.filter(pl.col("generator") == gen)
                if sub.height == 0:
                    continue
                self._plot_single_peak(sub, gen, paths)

        # Two-peak
        doubles = df.filter(pl.col("generator").str.contains("TwoPeak"))
        if doubles.height > 0 and all(
            col in doubles.columns for col in ["pair_rmse", "uncert_sep"]
        ):
            for gen in sorted(set(doubles.get_column("generator").to_list())):
                sub = doubles.filter(pl.col("generator") == gen)
                if sub.height == 0:
                    continue
                self._plot_double_peak(sub, gen, paths)

        return paths
