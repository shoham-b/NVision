from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl

from nvision.viz._f32_json import write_plotly_gz


class ExperimentsMixin:
    """Mixin for experiment summary plots."""

    out_dir: Path

    def plot_experiment_summary(self, df: pl.DataFrame) -> list[dict]:
        """Plot RMSE and Measurements by (noise, strategy) for each generator in experiment results."""
        if df.is_empty():
            return []

        plots = []

        # partition_by is O(N) compared to O(M*N) multiple .filter passes
        partitions = df.partition_by("generator", as_dict=True)

        for gen_tuple, sub in partitions.items():
            gen = gen_tuple[0]

            metrics_to_plot = []
            error_metric = "pair_rmse" if "pair_rmse" in sub.columns else "abs_err_x"
            if error_metric in sub.columns:
                metrics_to_plot.append((error_metric, "Error / Metric"))

            if "measurements" in sub.columns:
                metrics_to_plot.append(("measurements", "Average Steps to Converge"))

            for metric, ylabel in metrics_to_plot:
                agg = sub.group_by(["noise", "strategy"]).agg(pl.col(metric).mean())
                try:
                    pivot = agg.pivot(on="strategy", index="noise", values=metric)
                except Exception:
                    continue

                try:
                    pivot = (
                        pivot.with_columns(pl.col("noise").str.extract(r"([\d\.]+)").cast(pl.Float64).alias("_n"))
                        .sort("_n")
                        .drop("_n")
                    )
                except Exception:
                    pivot = pivot.sort("noise")

                out_path = self.out_dir / f"summary_{gen}_{metric}.json.gz"
                self._plot_pivot_from_polars(pivot, f"Summary: {gen} ({metric})", "Noise Level", ylabel, out_path)
                plots.append({"type": "summary", "path": str(out_path), "generator": gen, "metric": metric})

            # Compute Absolute Step Savings relative to Sweeps
            if "measurements" in sub.columns:
                agg = sub.group_by(["noise", "strategy"]).agg(pl.col("measurements").mean())
                try:
                    pivot = agg.pivot(on="strategy", index="noise", values="measurements")
                    sweep_col = None
                    for col in pivot.columns:
                        if "sweep" in col.lower() or "sobol" in col.lower():
                            sweep_col = col
                            break

                    if sweep_col:
                        try:
                            pivot = (
                                pivot.with_columns(
                                    pl.col("noise").str.extract(r"([\d\.]+)").cast(pl.Float64).alias("_n")
                                )
                                .sort("_n")
                                .drop("_n")
                            )
                        except Exception:
                            pivot = pivot.sort("noise")

                        fig = go.Figure()
                        indices = pivot.get_column("noise").to_list()
                        baseline_vals = pivot.get_column(sweep_col).to_list()

                        has_savings = False
                        for strat in pivot.columns:
                            if strat == "noise" or strat == sweep_col:
                                continue
                            strat_vals = pivot.get_column(strat).to_list()
                            savings = [
                                b - s if b is not None and s is not None else None
                                for b, s in zip(baseline_vals, strat_vals)
                            ]
                            fig.add_trace(go.Scatter(name=strat, x=indices, y=savings, mode="lines+markers"))
                            has_savings = True

                        if has_savings:
                            fig.update_layout(
                                title=f"Measurement Savings vs {sweep_col} <br>Generator: {gen}",
                                xaxis_title="Noise Level",
                                yaxis_title="Absolute Steps Saved",
                                template="plotly_white",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            )
                            out_path = self.out_dir / f"summary_{gen}_savings.json.gz"
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            write_plotly_gz(fig, out_path)
                            plots.append(
                                {"type": "summary", "path": str(out_path), "generator": gen, "metric": "savings"}
                            )
                except Exception:
                    pass

        return plots

    def _plot_pivot_from_polars(
        self, pivot_pl: pl.DataFrame, title: str, xlabel: str, ylabel: str, out_path: Path
    ) -> None:
        """Plot a chart from a polars pivoted dataframe as an interactive plot."""
        fig = go.Figure()

        index_col = pivot_pl.columns[0]
        strategies = pivot_pl.columns[1:]

        indices = pivot_pl.get_column(index_col).to_list()

        is_line_chart = "measurements" in title.lower()

        for strat in strategies:
            values = pivot_pl.get_column(strat).to_list()
            if is_line_chart:
                fig.add_trace(go.Scatter(name=strat, x=indices, y=values, mode="lines+markers"))
            else:
                fig.add_trace(go.Bar(name=strat, x=indices, y=values))

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            barmode="group" if not is_line_chart else None,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_plotly_gz(fig, out_path)

    def plot_locator_summary(self, df: pl.DataFrame) -> list[dict]:
        """Create comparison plots for locator sweeps."""
        if df.is_empty():
            return []

        entries = []

        # Per-generator summary: error by noise level
        experiment_plots = self.plot_experiment_summary(df)
        entries.extend(experiment_plots)

        # Cross-strategy comparison plots
        comparison_plots = self.plot_model_comparisons(df)
        entries.extend(comparison_plots)

        # Milestone Analysis
        milestone_plots = self.plot_milestone_analysis(df)
        entries.extend(milestone_plots)

        # Savings vs Span Analysis (Over Noise)
        span_plots = self.plot_savings_vs_span(df)
        entries.extend(span_plots)

        # Savings vs Span Analysis (Per Noise)
        span_per_noise_plots = self.plot_savings_vs_span_per_noise(df)
        entries.extend(span_per_noise_plots)

        return entries

    def plot_savings_vs_span_per_noise(self, df: pl.DataFrame) -> list[dict]:
        plots = []
        if "measurements" not in df.columns or "attempt" not in df.columns:
            return plots

        partitions = df.partition_by(["generator", "noise"], as_dict=True)
        for (gen, noise), sub in partitions.items():
            try:
                # Pivot on strategy to get measurements per strategy per repeat
                pivot_m = sub.pivot(on="strategy", index="attempt", values="measurements", aggregate_function="mean")
                sweep_col = None
                for col in pivot_m.columns:
                    if "sweep" in col.lower() or "sobol" in col.lower():
                        sweep_col = col
                        break
                if not sweep_col:
                    continue

                # get parameters per repeat for this group
                params_df = sub.group_by("attempt").first()
                joined = pivot_m.join(params_df, on="attempt")
                for _c in [
                    "final_est_linewidth",
                    "final_est_split",
                    "acquisition_hi",
                    "acquisition_lo",
                    "expected_uniform_points",
                ]:
                    if _c in joined.columns:
                        joined = joined.with_columns(pl.col(_c).cast(pl.Float64, strict=False))

                fig = go.Figure()
                f_spans = []
                baseline_vals = joined.get_column(sweep_col).to_list()

                has_valid = False
                for row in joined.iter_rows(named=True):
                    lw = row.get("final_est_linewidth")
                    split = row.get("final_est_split")
                    hi = row.get("acquisition_hi")
                    lo = row.get("acquisition_lo")
                    exp_pts = row.get("expected_uniform_points")

                    f_span = None
                    if lw is not None and hi is not None and lo is not None and hi > lo:
                        split_val = split if split is not None else 0
                        f_span = max(2 * lw, split_val + lw) / (hi - lo)
                    elif exp_pts is not None and exp_pts > 0:
                        f_span = 1.0 / exp_pts

                    f_spans.append(f_span)

                for strat in pivot_m.columns:
                    if strat == "attempt" or strat == sweep_col:
                        continue

                    strat_vals = joined.get_column(strat).to_list()
                    strat_savings = []
                    valid_f_spans = []

                    for b, s, f in zip(baseline_vals, strat_vals, f_spans):
                        if b is not None and s is not None and f is not None:
                            strat_savings.append(b - s)
                            valid_f_spans.append(f)
                            has_valid = True

                    if strat_savings:
                        fig.add_trace(go.Scatter(name=strat, x=valid_f_spans, y=strat_savings, mode="markers"))

                if has_valid:
                    fig.update_layout(
                        title=f"Savings vs Span <br>Gen: {gen} | Noise: {noise}",
                        xaxis_title="Fractional Signal Span (f_span)",
                        yaxis_title=f"Absolute Steps Saved vs {sweep_col}",
                        template="plotly_white",
                        xaxis_type="log",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    safe_noise = str(noise).replace(".", "_")
                    out_path = self.out_dir / f"model_comp_{gen}_{safe_noise}_savings_span.json.gz"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    write_plotly_gz(fig, out_path)
                    plots.append(
                        {
                            "type": "model_comparison",
                            "path": str(out_path),
                            "generator": gen,
                            "noise": noise,
                            "metric": "savings_vs_span_per_noise",
                        }
                    )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Could not plot span per noise for {gen} {noise}: {e}")

        return plots

    def plot_savings_vs_span(self, df: pl.DataFrame) -> list[dict]:
        plots = []
        if "measurements" not in df.columns:
            return plots

        partitions = df.partition_by("generator", as_dict=True)
        for gen_tuple, sub in partitions.items():
            gen = gen_tuple[0]

            _numeric_cols = [
                "final_est_linewidth",
                "final_est_split",
                "acquisition_hi",
                "acquisition_lo",
                "expected_uniform_points",
                "measurements",
            ]
            for _c in _numeric_cols:
                if _c in sub.columns:
                    sub = sub.with_columns(pl.col(_c).cast(pl.Float64, strict=False))

            agg_exprs = [pl.col("measurements").mean()]
            agg = sub.group_by(["noise", "strategy"]).agg(agg_exprs)

            try:
                pivot_m = agg.pivot(on="strategy", index="noise", values="measurements")
                sweep_col = None
                for col in pivot_m.columns:
                    if "sweep" in col.lower() or "sobol" in col.lower():
                        sweep_col = col
                        break
                if not sweep_col:
                    continue

                # Get the max linewidth, split, etc per noise level (best estimate)
                noise_agg_cols = []
                for c in [
                    "final_est_linewidth",
                    "final_est_split",
                    "acquisition_hi",
                    "acquisition_lo",
                    "expected_uniform_points",
                ]:
                    if c in sub.columns:
                        noise_agg_cols.append(pl.col(c).max().alias(c))

                if not noise_agg_cols:
                    continue

                noise_df = sub.group_by("noise").agg(noise_agg_cols)
                joined = pivot_m.join(noise_df, on="noise")

                fig = go.Figure()
                f_spans = []
                baseline_vals = joined.get_column(sweep_col).to_list()

                has_valid = False
                for row in joined.iter_rows(named=True):
                    lw = row.get("final_est_linewidth")
                    split = row.get("final_est_split")
                    hi = row.get("acquisition_hi")
                    lo = row.get("acquisition_lo")
                    exp_pts = row.get("expected_uniform_points")

                    f_span = None
                    if lw is not None and hi is not None and lo is not None and hi > lo:
                        split_val = split if split is not None else 0
                        f_span = max(2 * lw, split_val + lw) / (hi - lo)
                    elif exp_pts is not None and exp_pts > 0:
                        f_span = 1.0 / exp_pts

                    f_spans.append(f_span)

                for strat in pivot_m.columns:
                    if strat == "noise" or strat == sweep_col:
                        continue

                    strat_vals = joined.get_column(strat).to_list()
                    strat_savings = []
                    valid_f_spans = []

                    for b, s, f in zip(baseline_vals, strat_vals, f_spans):
                        if b is not None and s is not None and f is not None:
                            strat_savings.append(b - s)
                            valid_f_spans.append(f)
                            has_valid = True

                    if strat_savings:
                        pts = sorted(zip(valid_f_spans, strat_savings))
                        x_vals = [p[0] for p in pts]
                        y_vals = [p[1] for p in pts]
                        fig.add_trace(go.Scatter(name=strat, x=x_vals, y=y_vals, mode="lines+markers"))

                if has_valid:
                    fig.update_layout(
                        title=f"Measurement Savings vs Fractional Signal Span <br>Generator: {gen}",
                        xaxis_title="Fractional Signal Span (f_span)",
                        yaxis_title=f"Absolute Steps Saved vs {sweep_col}",
                        template="plotly_white",
                        xaxis_type="log",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    out_path = self.out_dir / f"summary_{gen}_savings_vs_span.json.gz"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    write_plotly_gz(fig, out_path)
                    plots.append(
                        {
                            "type": "summary",
                            "path": str(out_path),
                            "generator": gen,
                            "metric": "savings_vs_span",
                            "title": "Savings vs Span",
                        }
                    )

            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Failed to plot savings vs span: {e}")

        return plots

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
        partitions = df.partition_by("strategy", as_dict=True)

        # 1. Distribution of steps to fb convergence
        if "steps_to_fb" in df.columns:
            out_path = self.out_dir / "milestone_steps_to_fb.json.gz"
            fig = go.Figure()
            # Histogram of steps per strategy
            for (strat,), sub in partitions.items():
                steps = sub.get_column("steps_to_fb").drop_nans().drop_nulls()
                if not steps.is_empty():
                    fig.add_trace(go.Histogram(x=steps.to_list(), name=strat, opacity=0.75))

            fig.update_layout(
                title="Steps to Center Frequency (fb) Convergence",
                xaxis_title="Steps",
                yaxis_title="Count",
                barmode="overlay",
                template="plotly_white",
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_plotly_gz(fig, out_path)
            entries.append({"type": "milestone", "path": str(out_path), "title": "Steps to fb Convergence"})

        # 2. Error comparison (Milestone vs Final)
        if "err_fc_at_milestone" in df.columns and "final_err_fc" in df.columns:
            out_path = self.out_dir / "milestone_error_comparison_fc.json.gz"
            fig = go.Figure()
            for (strat,), sub in partitions.items():
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
            write_plotly_gz(fig, out_path)
            entries.append({"type": "milestone", "path": str(out_path), "title": "Splitting Error Comparison"})

        # 3. Zeeman resolution sufficiency (Error Delta)
        if "err_fc_diff" in df.columns:
            out_path = self.out_dir / "milestone_error_delta_fc.json.gz"
            fig = go.Figure()
            for (strat,), sub in partitions.items():
                err_diff = sub.get_column("err_fc_diff").drop_nans().drop_nulls()
                if not err_diff.is_empty():
                    fig.add_trace(go.Box(y=err_diff.to_list(), name=strat))

            fig.update_layout(
                title="Error Reduction after fb Convergence (fc)",
                yaxis_title="Error Reduction (Hz)",
                template="plotly_white",
            )
            write_plotly_gz(fig, out_path)
            entries.append({"type": "milestone", "path": str(out_path), "title": "Zeeman Resolution Gain"})

        return entries
