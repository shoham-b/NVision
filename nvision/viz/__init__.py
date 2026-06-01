from __future__ import annotations

import polars as pl

from nvision.viz.base import VizBase
from nvision.viz.bayesian import BayesianMixin
from nvision.viz.comparisons import ComparisonsMixin
from nvision.viz.experiments import ExperimentsMixin
from nvision.viz.measurements import MeasurementsMixin
from nvision.viz.metrics import MetricsVizMixin

# Removed duplicate plot_all_metrics implementation



class Viz(VizBase, ExperimentsMixin, MeasurementsMixin, BayesianMixin, ComparisonsMixin, MetricsVizMixin):
    """Visualization facade combining all plotting capabilities."""

    def plot_all_metrics(self, df_loc: pl.DataFrame) -> list[dict[str, object]]:
        """Generate all metric plots for each unique strategy/generator.

        Returns a list of manifest entries produced by the individual metric plot methods.
        """
        if df_loc.is_empty():
            return []
        entries: list[dict[str, object]] = []
        # Iterate over unique combinations of generator, noise, and strategy
        # ⚡ Bolt Optimization: Using partition_by instead of repeated .filter() inside a loop
        # reduces time complexity from O(N^2) to O(N) by dividing the DataFrame in a single pass.
        partitions = df_loc.partition_by(["generator", "noise", "strategy"], as_dict=True)
        for (gen, noise, _strat), subset in partitions.items():
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
