from __future__ import annotations

import polars as pl

from nvision.viz.base import VizBase
from nvision.viz.bayesian import BayesianMixin
from nvision.viz.comparisons import ComparisonsMixin
from nvision.viz.experiments import ExperimentsMixin
from nvision.viz.grid_study import GridStudyMixin
from nvision.viz.measurements import MeasurementsMixin
from nvision.viz.metrics import MetricsVizMixin

# Removed duplicate plot_all_metrics implementation


class Viz(
    VizBase, ExperimentsMixin, MeasurementsMixin, BayesianMixin, ComparisonsMixin, MetricsVizMixin, GridStudyMixin
):
    """Visualization facade combining all plotting capabilities."""

    def plot_all_metrics(self, df_loc: pl.DataFrame) -> list[dict[str, object]]:
        """Generate all metric plots for each unique strategy/generator.

        Returns a list of manifest entries produced by the individual metric plot methods.
        """
        if df_loc.is_empty():
            return []
        entries: list[dict[str, object]] = []
        # Iterate over unique combinations of generator, noise, and strategy
        unique = df_loc.select(["generator", "noise", "strategy"]).unique()
        for row in unique.iter_rows(named=True):
            gen = row["generator"]
            noise = row["noise"]
            strat = row["strategy"]
            # Retrieve metrics for this combo from the dataframe rows
            subset = df_loc.filter(
                (pl.col("generator") == gen) & (pl.col("noise") == noise) & (pl.col("strategy") == strat)
            )
            # Metrics are stored in the "metrics" column as a struct; extract first if any
            if "metrics" not in subset.columns:
                continue
            metrics_struct = subset.get_column("metrics")[0] if subset.height > 0 else None
            if metrics_struct is None:
                continue
            # The StrategyMetrics object is stored; we assume it can be used directly
            metrics_obj = metrics_struct  # type: ignore[assignment]
            # Plot error distribution per parameter
            for param in getattr(metrics_obj, "absolute_errors", {}):
                entry = self.plot_error_distribution(metrics_obj, param, gen, noise)
                if entry:
                    entries.append(entry)
            # Plot convergence steps per parameter
            for param in getattr(metrics_obj, "convergence_steps", {}):
                entry = self.plot_convergence_steps(metrics_obj, param, gen, noise)
                if entry:
                    entries.append(entry)
            # Plot Sobol difference (single per strategy)
            entry = self.plot_sobol_difference(metrics_obj, gen, noise)
            if entry:
                entries.append(entry)
        return entries

    pass


__all__ = ["Viz"]
