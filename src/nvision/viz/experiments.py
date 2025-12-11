from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl


class ExperimentsMixin:
    """Mixin for experiment summary plots."""

    out_dir: Path

    def plot_experiment_summary(self, df: pl.DataFrame) -> list[Path]:
        """Plot RMSE by (noise, strategy) for each generator in experiment results."""
        if df.is_empty():
            return []

        generators = df.get_column("generator").unique().to_list()
        plots = []

        for gen in generators:
            sub = df.filter(pl.col("generator") == gen)
            # Create a pivot table: Index=Noise, Columns=Strategy, Value=RMSE (mean) or similar metric
            # Using metric 'pair_rmse' or 'abs_err_x' depending on availability

            metric = "pair_rmse" if "pair_rmse" in sub.columns else "abs_err_x"
            if metric not in sub.columns:
                continue

            # Aggregate
            agg = sub.group_by(["noise", "strategy"]).agg(pl.col(metric).mean())

            # Pivot for heatmap/bar chart
            try:
                pivot = agg.pivot(on="strategy", index="noise", values=metric)
            except Exception:
                # Polars pivot syntax might vary or fail if types mismatch
                continue

            out_path = self.out_dir / f"summary_{gen}_{metric}.html"
            self._plot_pivot_from_polars(pivot, f"Summary: {gen} ({metric})", "Noise Level", out_path)
            plots.append(out_path)

        return plots

    def _plot_pivot_from_polars(self, pivot_pl: pl.DataFrame, title: str, ylabel: str, out_path: Path) -> None:
        """Plot a bar chart from a polars pivoted dataframe as an interactive plot."""
        fig = go.Figure()

        index_col = pivot_pl.columns[0]
        strategies = pivot_pl.columns[1:]

        indices = pivot_pl.get_column(index_col).to_list()

        for strat in strategies:
            values = pivot_pl.get_column(strat).to_list()
            fig.add_trace(go.Bar(name=strat, x=indices, y=values))

        fig.update_layout(
            title=title,
            xaxis_title=ylabel,
            yaxis_title="Error / Metric",
            barmode="group",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path)

    def plot_locator_summary(self, df: pl.DataFrame) -> None:
        """Create comparison plots for locator sweeps."""
        # Implementation of locator specific summaries if needed
        pass
