from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import polars as pl

from nvision.spectra.nv_center import DEFAULT_NV_CENTER_FREQ_X_MAX, DEFAULT_NV_CENTER_FREQ_X_MIN


class ExperimentsMixin:
    """Mixin for experiment summary plots."""

    out_dir: Path

    @staticmethod
    def _find_sweep_baseline(columns: list[str]) -> str | None:
        """Find the sweep baseline strategy. Prioritizes 'sweep', then falls back to 'sobol'."""
        for col in columns:
            if "sweep" in col.lower():
                return col
        for col in columns:
            if "sobol" in col.lower():
                return col
        return None

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

            if "splitting_converged_step" in sub.columns and sub["splitting_converged_step"].drop_nulls().len() > 0:
                metrics_to_plot.append(("splitting_converged_step", "Splitting Converged @ Step"))

            if "all_converged_step" in sub.columns and sub["all_converged_step"].drop_nulls().len() > 0:
                metrics_to_plot.append(("all_converged_step", "All Converged @ Step"))

            if "err_fb_at_milestone" in sub.columns and sub["err_fb_at_milestone"].drop_nulls().len() > 0:
                metrics_to_plot.append(("err_fb_at_milestone", "Error @ Freq. Convergence"))

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

                out_path = self._plot_pivot_from_polars(
                    pivot, f"Summary: {gen} ({metric})", "Noise Level", ylabel, f"summary_{gen}_{metric}.json.gz"
                )
                plots.append({"type": "summary", "path": out_path.as_posix(), "generator": gen, "metric": metric})

            # Compute Absolute Step Savings relative to Sweeps
            if "measurements" in sub.columns:
                agg = sub.group_by(["noise", "strategy"]).agg(pl.col("measurements").mean())
                try:
                    pivot = agg.pivot(on="strategy", index="noise", values="measurements")
                    sweep_col = self._find_sweep_baseline(pivot.columns)

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

                        indices = pivot.get_column("noise").to_list()
                        baseline_vals = pivot.get_column(sweep_col).to_list()

                        savings_series: list[dict[str, Any]] = []
                        for strat in pivot.columns:
                            if strat == "noise" or strat == sweep_col:
                                continue
                            strat_vals = pivot.get_column(strat).to_list()
                            savings = [
                                b - s if b is not None and s is not None else None
                                for b, s in zip(baseline_vals, strat_vals, strict=False)
                            ]
                            savings_series.append({"name": strat, "x": indices, "y": savings})

                        if savings_series:
                            savings_data: dict[str, Any] = {
                                "_graph_type": "chart",
                                "title": f"Measurement Savings vs {sweep_col} <br>Generator: {gen}",
                                "xaxis_title": "Noise Level",
                                "yaxis_title": "Absolute Steps Saved",
                                "mode": "lines+markers",
                                "series": savings_series,
                            }
                            out_path = self._emit(savings_data, f"summary_{gen}_savings.json.gz")
                            plots.append(
                                {"type": "summary", "path": out_path.as_posix(), "generator": gen, "metric": "savings"}
                            )
                except Exception:
                    pass

            # Savings for convergence-step metrics vs sweep measurements
            for conv_metric, sav_suffix, sav_label in [
                ("splitting_converged_step", "savings_freq_converged", "Steps Saved to Splitting Convergence"),
                ("all_converged_step", "savings_all_converged", "Steps Saved to Full Convergence"),
            ]:
                if conv_metric not in sub.columns or "measurements" not in sub.columns:
                    continue
                if sub[conv_metric].drop_nulls().len() == 0:
                    continue
                try:
                    meas_agg = sub.group_by(["noise", "strategy"]).agg(pl.col("measurements").mean())
                    meas_pivot = meas_agg.pivot(on="strategy", index="noise", values="measurements")
                    sweep_col = self._find_sweep_baseline(meas_pivot.columns)
                    if not sweep_col:
                        continue

                    conv_agg = (
                        sub.filter(pl.col(conv_metric).is_not_null())
                        .group_by(["noise", "strategy"])
                        .agg(pl.col(conv_metric).mean())
                    )
                    if conv_agg.is_empty():
                        continue

                    sweep_baseline = meas_pivot.select(["noise", sweep_col]).rename({sweep_col: "_baseline"})
                    try:
                        sweep_baseline = (
                            sweep_baseline.with_columns(
                                pl.col("noise").str.extract(r"([\d\.]+)").cast(pl.Float64).alias("_n")
                            )
                            .sort("_n")
                            .drop("_n")
                        )
                        conv_agg = (
                            conv_agg.with_columns(
                                pl.col("noise").str.extract(r"([\d\.]+)").cast(pl.Float64).alias("_n")
                            )
                            .sort("_n")
                            .drop("_n")
                        )
                    except Exception:
                        sweep_baseline = sweep_baseline.sort("noise")
                        conv_agg = conv_agg.sort("noise")

                    joined = conv_agg.join(sweep_baseline, on="noise", how="inner")

                    conv_savings_series: list[dict[str, Any]] = []
                    for strat in joined.get_column("strategy").unique().to_list():
                        if strat == sweep_col:
                            continue
                        strat_data = joined.filter(pl.col("strategy") == strat)
                        noises = strat_data.get_column("noise").to_list()
                        baseline = strat_data.get_column("_baseline").to_list()
                        conv_steps = strat_data.get_column(conv_metric).to_list()
                        savings = [
                            b - s if b is not None and s is not None else None
                            for b, s in zip(baseline, conv_steps, strict=False)
                        ]
                        conv_savings_series.append({"name": strat, "x": noises, "y": savings})

                    if conv_savings_series:
                        conv_sav_data: dict[str, Any] = {
                            "_graph_type": "chart",
                            "title": f"{sav_label} vs {sweep_col} <br>Generator: {gen}",
                            "xaxis_title": "Noise Level",
                            "yaxis_title": f"Absolute Steps Saved vs {sweep_col}",
                            "mode": "lines+markers",
                            "series": conv_savings_series,
                        }
                        out_path = self._emit(conv_sav_data, f"summary_{gen}_{sav_suffix}.json.gz")
                        plots.append(
                            {
                                "type": "summary",
                                "path": out_path.as_posix(),
                                "generator": gen,
                                "metric": sav_suffix,
                            }
                        )
                except Exception:
                    pass

        return plots

    def _plot_pivot_from_polars(
        self, pivot_pl: pl.DataFrame, title: str, xlabel: str, ylabel: str, rel_name: str
    ) -> Path:
        """Serialize pivot chart data (definition lives in static/graphs/chart.json)."""
        index_col = pivot_pl.columns[0]
        strategies = pivot_pl.columns[1:]
        indices = pivot_pl.get_column(index_col).to_list()
        is_line_chart = "measurements" in title.lower()

        series: list[dict[str, Any]] = []
        for strat in strategies:
            values = pivot_pl.get_column(strat).to_list()
            series.append({"name": strat, "x": indices, "y": values})

        data: dict[str, Any] = {
            "_graph_type": "chart",
            "title": title,
            "xaxis_title": xlabel,
            "yaxis_title": ylabel,
            "mode": "lines+markers" if is_line_chart else "bar",
            "series": series,
        }
        return self._emit(data, rel_name)

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

        # Parameter-grid study summaries (no-op unless grid_ columns are present)
        grid_plots = self.plot_grid_study(df)
        entries.extend(grid_plots)

        return entries

    def _add_correct_f_span(self, sub: pl.DataFrame) -> pl.DataFrame:
        """Add correct row_f_span column to a generator-partitioned dataframe."""
        sweep_rows = sub.filter(pl.col("strategy") == "SimpleSweep")
        if not sweep_rows.is_empty():
            hi_val = sweep_rows.get_column("acquisition_hi").drop_nulls().mean()
            lo_val = sweep_rows.get_column("acquisition_lo").drop_nulls().mean()
        else:
            hi_val, lo_val = DEFAULT_NV_CENTER_FREQ_X_MAX, DEFAULT_NV_CENTER_FREQ_X_MIN
        domain_width = (
            hi_val - lo_val
            if hi_val is not None and lo_val is not None and hi_val > lo_val
            else DEFAULT_NV_CENTER_FREQ_X_MAX - DEFAULT_NV_CENTER_FREQ_X_MIN
        )

        # Lineshape-agnostic width source: prefer the derived effective-HWHM column
        # (saturation-Voigt runs), fall back to the raw Lorentzian/Voigt columns.
        width_candidates: list[pl.Expr] = []
        if "final_est_effective_hwhm" in sub.columns:
            width_candidates.append(pl.col("final_est_effective_hwhm"))
        if "final_est_linewidth" in sub.columns:
            width_candidates.append(pl.col("final_est_linewidth"))
        if "final_est_homogeneous_linewidth" in sub.columns:
            width_candidates.append(pl.col("final_est_homogeneous_linewidth"))
        if "final_est_fwhm_total" in sub.columns:
            width_candidates.append(pl.col("final_est_fwhm_total") / 2.0)
        width_expr = pl.coalesce(width_candidates) if width_candidates else pl.lit(None, dtype=pl.Float64)

        # Total splitting extent: two Zeeman groups at +/- zeeman_split, plus hyperfine split.
        zee_expr = (
            pl.col("final_est_zeeman_split").fill_null(0.0) * 2.0
            if "final_est_zeeman_split" in sub.columns
            else pl.lit(0.0)
        )
        hf_expr = pl.col("final_est_split").fill_null(0.0) if "final_est_split" in sub.columns else pl.lit(0.0)

        sub = sub.with_columns(width_expr.alias("_width_src"), (zee_expr + hf_expr).alias("_split_src"))

        best_est = (
            sub.filter(~pl.col("strategy").str.contains("Sweep"))
            .group_by(["noise", "attempt"])
            .agg(
                [
                    pl.col("_width_src").mean().alias("ref_linewidth"),
                    pl.col("_split_src").mean().alias("ref_split"),
                ]
            )
        )

        # Always join (even when best_est is empty -- a left join against an empty
        # table just yields null ref_linewidth/ref_split) rather than branching on
        # emptiness: the two branches previously added different column counts,
        # which crashed the pl.concat() below whenever some generators had only
        # Sweep-strategy rows (best_est empty) and others didn't (best_est non-empty)
        # in the same summary pass.
        sub = sub.join(best_est, on=["noise", "attempt"], how="left")
        sub = sub.with_columns(
            pl.col("ref_linewidth").fill_null(pl.col("_width_src")).alias("effective_lw"),
            pl.col("ref_split").fill_null(pl.col("_split_src")).alias("effective_split"),
        )

        def calc_row_f_span(row):
            lw = row["effective_lw"]
            split = row["effective_split"]
            exp_pts = row["expected_uniform_points"]
            if lw is not None and lw > 0:
                split_val = split if split is not None else 0.0
                return max(2.0 * lw, split_val + lw) / domain_width
            elif exp_pts is not None and exp_pts > 0:
                return 1.0 / exp_pts
            return None

        return sub.with_columns(
            pl.struct(["effective_lw", "effective_split", "expected_uniform_points"])
            .map_elements(calc_row_f_span, return_dtype=pl.Float64)
            .alias("row_f_span")
        )

    def plot_savings_vs_span_per_noise(self, df: pl.DataFrame) -> list[dict]:
        plots = []
        if "measurements" not in df.columns or "attempt" not in df.columns:
            return plots

        # Add correct f_span column to df grouped by generator first
        df_list = []
        for _gen_tuple, sub in df.partition_by("generator", as_dict=True).items():
            for _c in [
                "final_est_linewidth",
                "final_est_homogeneous_linewidth",
                "final_est_split",
                "final_est_effective_hwhm",
                "final_est_zeeman_split",
                "final_est_fwhm_total",
                "acquisition_hi",
                "acquisition_lo",
                "expected_uniform_points",
                "measurements",
            ]:
                if _c in sub.columns:
                    sub = sub.with_columns(pl.col(_c).cast(pl.Float64, strict=False))
            df_list.append(self._add_correct_f_span(sub))
        df = pl.concat(df_list) if df_list else df

        partitions = df.partition_by(["generator", "noise"], as_dict=True)
        for (gen, noise), sub in partitions.items():
            try:
                # Pivot on strategy to get measurements per strategy per repeat
                pivot_m = sub.pivot(on="strategy", index="attempt", values="measurements", aggregate_function="mean")
                sweep_col = self._find_sweep_baseline(pivot_m.columns)
                if not sweep_col:
                    continue

                # get parameters per repeat for this group
                params_df = sub.group_by("attempt").agg(pl.col("row_f_span").first().alias("f_span"))
                joined = pivot_m.join(params_df, on="attempt")

                f_spans = joined.get_column("f_span").to_list()
                baseline_vals = joined.get_column(sweep_col).to_list()

                has_valid = False
                span_series: list[dict[str, Any]] = []
                for strat in pivot_m.columns:
                    if strat == "attempt" or strat == sweep_col:
                        continue

                    strat_vals = joined.get_column(strat).to_list()
                    strat_savings = []
                    valid_f_spans = []

                    for b, s, f in zip(baseline_vals, strat_vals, f_spans, strict=False):
                        if b is not None and s is not None and f is not None:
                            strat_savings.append(b - s)
                            valid_f_spans.append(f)
                            has_valid = True

                    if strat_savings:
                        span_series.append({"name": strat, "x": valid_f_spans, "y": strat_savings})

                if has_valid:
                    span_data: dict[str, Any] = {
                        "_graph_type": "chart",
                        "title": f"Savings vs Span <br>Gen: {gen} | Noise: {noise}",
                        "xaxis_title": "Fractional Signal Span (f_span)",
                        "yaxis_title": f"Absolute Steps Saved vs {sweep_col}",
                        "xaxis_type": "log",
                        "mode": "markers",
                        "series": span_series,
                    }
                    safe_noise = str(noise).replace(".", "_")
                    out_path = self._emit(span_data, f"model_comp_{gen}_{safe_noise}_savings_span.json.gz")
                    plots.append(
                        {
                            "type": "model_comparison",
                            "path": out_path.as_posix(),
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
        if "measurements" not in df.columns or "attempt" not in df.columns:
            return plots

        partitions = df.partition_by("generator", as_dict=True)
        for gen_tuple, sub in partitions.items():
            gen = gen_tuple[0]

            _numeric_cols = [
                "final_est_linewidth",
                "final_est_split",
                "final_est_effective_hwhm",
                "final_est_zeeman_split",
                "final_est_fwhm_total",
                "acquisition_hi",
                "acquisition_lo",
                "expected_uniform_points",
                "measurements",
            ]
            for _c in _numeric_cols:
                if _c in sub.columns:
                    sub = sub.with_columns(pl.col(_c).cast(pl.Float64, strict=False))

            sub = self._add_correct_f_span(sub)

            try:
                # Identify the sweep baseline column
                strategies = sub.get_column("strategy").unique().to_list()
                sweep_col = self._find_sweep_baseline(strategies)
                if not sweep_col:
                    continue

                sweep_df = (
                    sub.filter(pl.col("strategy") == sweep_col)
                    .group_by(["noise", "attempt"])
                    .agg(pl.col("measurements").mean().alias("sweep_meas"))
                )

                has_valid = False
                vs_span_series: list[dict[str, Any]] = []
                for strat in strategies:
                    if strat == sweep_col:
                        continue

                    strat_df = (
                        sub.filter(pl.col("strategy") == strat)
                        .group_by(["noise", "attempt"])
                        .agg(
                            pl.col("measurements").mean().alias("strat_meas"),
                            pl.col("row_f_span").first().alias("f_span"),
                        )
                        .join(sweep_df, on=["noise", "attempt"], how="inner")
                    )

                    pts = []
                    for row in strat_df.iter_rows(named=True):
                        b = row["sweep_meas"]
                        s = row["strat_meas"]
                        f = row["f_span"]
                        if b is not None and s is not None and f is not None:
                            pts.append((f, b - s))
                            has_valid = True

                    if pts:
                        pts.sort()
                        vs_span_series.append(
                            {
                                "name": strat,
                                "x": [p[0] for p in pts],
                                "y": [p[1] for p in pts],
                            }
                        )

                # Only emit the chart when there is genuine x-variation
                all_x = [x for s in vs_span_series for x in s["x"]]
                if not has_valid or len(set(all_x)) < 2:
                    continue

                vs_span_data: dict[str, Any] = {
                    "_graph_type": "chart",
                    "title": f"Measurement Savings vs Fractional Signal Span <br>Generator: {gen}",
                    "xaxis_title": "Fractional Signal Span (f_span)",
                    "yaxis_title": f"Absolute Steps Saved vs {sweep_col}",
                    "xaxis_type": "log",
                    "mode": "markers",
                    "series": vs_span_series,
                }
                out_path = self._emit(vs_span_data, f"summary_{gen}_savings_vs_span.json.gz")
                plots.append(
                    {
                        "type": "summary",
                        "path": out_path.as_posix(),
                        "generator": gen,
                        "metric": "savings_vs_span",
                        "title": "Savings vs Span",
                    }
                )

            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Failed to plot savings vs span: {e}")

        return plots

    def plot_milestone_analysis(self, df: pl.DataFrame) -> list[dict]:
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
            out_path = self._emit(fig, "milestone_steps_to_fb.json.gz", is_figure=True)
            entries.append({"type": "milestone", "path": out_path.as_posix(), "title": "Steps to fb Convergence"})

        # 2. Error comparison (Milestone vs Final)
        if "err_fc_at_milestone" in df.columns and "final_err_fc" in df.columns:
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
            out_path = self._emit(fig, "milestone_error_comparison_fc.json.gz", is_figure=True)
            entries.append({"type": "milestone", "path": out_path.as_posix(), "title": "Splitting Error Comparison"})

        # 3. Zeeman resolution sufficiency (Error Delta)
        if "err_fc_diff" in df.columns:
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
            out_path = self._emit(fig, "milestone_error_delta_fc.json.gz", is_figure=True)
            entries.append({"type": "milestone", "path": out_path.as_posix(), "title": "Zeeman Resolution Gain"})

        return entries
