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

        plots = []

        # partition_by is O(N) compared to O(M*N) multiple .filter passes
        partitions = df.partition_by("generator", as_dict=True)

        for gen_tuple, sub in partitions.items():
            gen = gen_tuple[0]
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

    def plot_locator_summary(self, df: pl.DataFrame) -> list[dict]:
        """Create comparison plots for locator sweeps."""
        if df.is_empty():
            return []

        entries = []

        # Per-generator summary: error by noise level
        experiment_plots = self.plot_experiment_summary(df)
        for p in experiment_plots:
            entries.append({"type": "summary", "path": str(p)})

        # Cross-strategy comparison plots
        comparison_plots = self.plot_model_comparisons(df)
        entries.extend(comparison_plots)

        # Milestone Analysis
        milestone_plots = self.plot_milestone_analysis(df)
        entries.extend(milestone_plots)

        return entries

    def plot_milestone_analysis(self, df: pl.DataFrame) -> list[dict]:  # noqa: C901
        """Create plots for milestone-based convergence analysis."""
        milestone_cols = [
            "steps_to_fb",
            "err_fb_at_milestone",
            "err_fc_at_milestone",
            "final_err_fb",
            "final_err_fc",
            "err_fb_diff",
            "err_fc_diff",
        ]

        # Check if any milestone metrics exist
        if not any(col in df.columns for col in milestone_cols):
            return []

        entries = []

        # 1. Distribution of steps to fb convergence
        if "steps_to_fb" in df.columns:
            out_path = self.out_dir / "milestone_steps_to_fb.html"
            fig = go.Figure()
            # Histogram of steps per strategy
            for strat in df.get_column("strategy").unique():
                sub = df.filter(pl.col("strategy") == strat).get_column("steps_to_fb").drop_nans().drop_nulls()
                if not sub.is_empty():
                    fig.add_trace(go.Histogram(x=sub.to_list(), name=strat, opacity=0.75))

            fig.update_layout(
                title="Steps to Center Frequency (fb) Convergence",
                xaxis_title="Steps",
                yaxis_title="Count",
                barmode="overlay",
                template="plotly_white",
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(out_path)
            entries.append({"type": "milestone", "path": str(out_path), "title": "Steps to fb Convergence"})

        # 2. Error comparison (Milestone vs Final)
        if "err_fc_at_milestone" in df.columns and "final_err_fc" in df.columns:
            out_path = self.out_dir / "milestone_error_comparison_fc.html"
            fig = go.Figure()
            for strat in df.get_column("strategy").unique():
                sub = df.filter(pl.col("strategy") == strat)
                m_err = sub.get_column("err_fc_at_milestone").drop_nans().drop_nulls()
                f_err = sub.get_column("final_err_fc").drop_nans().drop_nulls()

                if not m_err.is_empty():
                    fig.add_trace(go.Box(y=m_err.to_list(), name=f"{strat} (Milestone)"))
                if not f_err.is_empty():
                    fig.add_trace(go.Box(y=f_err.to_list(), name=f"{strat} (Final)"))

            fig.update_layout(
                title="Splitting (fc) Absolute Error: Milestone vs Final",
                yaxis_title="Absolute Error (Hz)",
                template="plotly_white",
            )
            fig.write_html(out_path)
            entries.append({"type": "milestone", "path": str(out_path), "title": "Splitting Error Comparison"})

        # 3. Zeeman resolution sufficiency (Error Delta)
        if "err_fc_diff" in df.columns:
            out_path = self.out_dir / "milestone_error_delta_fc.html"
            fig = go.Figure()
            for strat in df.get_column("strategy").unique():
                sub = df.filter(pl.col("strategy") == strat).get_column("err_fc_diff").drop_nans().drop_nulls()
                if not sub.is_empty():
                    fig.add_trace(go.Box(y=sub.to_list(), name=strat))

            fig.update_layout(
                title="Error Reduction after fb Convergence (fc)",
                yaxis_title="Error Reduction (Hz)",
                template="plotly_white",
            )
            fig.write_html(out_path)
            entries.append({"type": "milestone", "path": str(out_path), "title": "Zeeman Resolution Gain"})

        return entries
