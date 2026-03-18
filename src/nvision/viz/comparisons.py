from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import polars as pl


class ComparisonsMixin:
    """Mixin for generating comparison plots between signal/strategies."""

    out_dir: Path

    def plot_model_comparisons(self, df: pl.DataFrame) -> list[dict[str, Any]]:
        """
        Generate comparison box plots for each (generator, noise) combination.

        Groups data by strategy and shows distribution of metrics:
        - Absolute Error (abs_err_x)
        - Measurements
        - Duration (ms)

        Returns list of manifest entries for the generated plots.
        """
        if df.is_empty():
            return []

        # Ensure we have necessary columns
        required_cols = ["generator", "noise", "strategy"]
        if not all(col in df.columns for col in required_cols):
            return []

        # Get unique combinations of Generator and Noise
        # unique() on struct to get unique rows
        combos = df.select(["generator", "noise"]).unique()

        manifest_entries = []

        for row in combos.iter_rows(named=True):
            gen = row["generator"]
            noise = row["noise"]

            # Filter data for this combination
            sub_df = df.filter((pl.col("generator") == gen) & (pl.col("noise") == noise))

            if sub_df.is_empty():
                continue

            # Generate stats/plots for this combo
            # We want to compare strategies side-by-side

            # Metric 1: Absolute Error
            if "abs_err_x" in sub_df.columns:
                self._create_comparison_plot(
                    sub_df,
                    gen,
                    noise,
                    metric="abs_err_x",
                    title_metric="Absolute Frequency Error",
                    y_axis_title="Error (Hz)",
                    manifest_entries=manifest_entries,
                )

            # Metric 2: Measurements
            if "measurements" in sub_df.columns:
                self._create_comparison_plot(
                    sub_df,
                    gen,
                    noise,
                    metric="measurements",
                    title_metric="Measurements Count",
                    y_axis_title="Count",
                    manifest_entries=manifest_entries,
                )

            # Metric 3: Duration
            if "duration_ms" in sub_df.columns:
                self._create_comparison_plot(
                    sub_df,
                    gen,
                    noise,
                    metric="duration_ms",
                    title_metric="Duration",
                    y_axis_title="Time (ms)",
                    manifest_entries=manifest_entries,
                )

        return manifest_entries

    def _create_comparison_plot(
        self,
        df: pl.DataFrame,
        gen: str,
        noise: str,
        metric: str,
        title_metric: str,
        y_axis_title: str,
        manifest_entries: list[dict[str, Any]],
    ) -> None:
        """Create and save a box plot comparing strategies for a specific metric."""
        fig = go.Figure()

        strategies = df.select("strategy").unique().to_series().to_list()
        # Sort strategies for consistent order
        strategies.sort()

        for strat in strategies:
            strat_data = df.filter(pl.col("strategy") == strat)
            vals = strat_data.get_column(metric).to_list()

            fig.add_trace(
                go.Box(
                    y=vals,
                    name=strat,
                    boxpoints="all",  # Show all points
                    jitter=0.3,  # Spread them out
                    pointpos=-1.8,  # Move points to the side
                )
            )

        title = f"Model Comparison: {title_metric}<br>Generator: {gen} | Noise: {noise}"
        fig.update_layout(
            title=title,
            yaxis_title=y_axis_title,
            template="plotly_white",
            showlegend=False,  # Box names are on X axis
            margin=dict(t=80, b=50, l=50, r=50),
        )

        filename = f"comparison_{gen}_{noise}_{metric}.html"
        # Sanitize filename
        safe_filename = filename.replace(" ", "_").replace("/", "-").replace(":", "")
        out_path = self.out_dir / safe_filename

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path)

        # Add entry to manifest list
        manifest_entries.append(
            {
                "type": "model_comparison",
                "metric": metric,
                "generator": gen,
                "noise": noise,
                "path": out_path.as_posix(),
                "title": f"Comparison: {title_metric}",
            }
        )
