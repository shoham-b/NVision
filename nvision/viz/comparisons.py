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

            # Metric 1: Absolute Error (with fallbacks for multi-peak runs).
            # Keep manifest metric key as `abs_err_x` because the static UI expects it.
            error_source_col = self._resolve_error_metric_column(sub_df)
            if error_source_col is not None:
                metric_df = sub_df
                if error_source_col != "abs_err_x":
                    metric_df = sub_df.with_columns(pl.col(error_source_col).alias("abs_err_x"))
                self._create_comparison_plot(
                    metric_df,
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

    @staticmethod
    def _resolve_error_metric_column(df: pl.DataFrame) -> str | None:
        """Pick the best available error column for strategy comparisons."""
        for col in ("abs_err_x", "pair_rmse", "abs_err_x1", "abs_err_x2"):
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _strategy_display_name(strategy: str) -> str:
        """Compact strategy names for readable x-axis labels."""
        short = strategy
        short = short.replace("Bayesian-", "Bayes-")
        short = short.replace("MaximumLikelihood", "MaxLike")
        short = short.replace("UtilitySampling", "Utility")
        short = short.replace("SimpleSweep", "Sweep")
        short = short.replace("MaxVariance", "MaxVar")
        return short

    @staticmethod
    def _strategy_tick_label(display_name: str) -> str:
        """Wrap strategy labels to multiple lines for crowded x-axes."""
        if "-" not in display_name:
            return display_name
        parts = [p for p in display_name.split("-") if p]
        # Plotly supports <br> in tick/category labels.
        return "<br>".join(parts)

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
            display_name = self._strategy_display_name(strat)
            tick_label = self._strategy_tick_label(display_name)

            fig.add_trace(
                go.Box(
                    y=vals,
                    name=tick_label,
                    boxpoints="all",  # Show all points
                    jitter=0.3,  # Spread them out
                    pointpos=-1.8,  # Move points to the side
                )
            )

        title = f"Model Comparison: {title_metric}<br>Generator: {gen} | Noise: {noise}"
        fig.update_layout(
            title=title,
            yaxis_title=y_axis_title,
            xaxis=dict(
                tickangle=-25,
                automargin=True,
            ),
            template="plotly_white",
            showlegend=False,  # Box names are on X axis
            margin=dict(t=80, b=120, l=50, r=50),
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
