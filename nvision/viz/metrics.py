"""Visualization mixin for strategy metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from nvision.metrics.types import StrategyMetrics
from nvision.viz._f32_json import write_plotly_gz


class MetricsVizMixin:
    """Mixin for generating advanced metric plots."""

    out_dir: Path

    def plot_error_distribution(
        self, metrics: StrategyMetrics, param_name: str, gen_name: str, noise_name: str
    ) -> dict[str, Any]:
        """Generate a violin plot showing the density of absolute errors."""
        errors = metrics.absolute_errors.get(param_name, [])
        if not errors:
            return {}

        fig = go.Figure()
        fig.add_trace(
            go.Violin(
                y=errors,
                name=metrics.strategy_name,
                box_visible=True,
                meanline_visible=True,
                fillcolor="lightseagreen",
                line_color="darkslategray",
                opacity=0.6,
            )
        )

        title = (
            f"Absolute Error Distribution: {param_name}<br>"
            f"Strategy: {metrics.strategy_name} | Generator: {gen_name} | Noise: {noise_name}"
        )
        fig.update_layout(title=title, yaxis_title="Absolute Error", template="plotly_white")

        filename = f"error_dist_{metrics.strategy_name}_{param_name}.json.gz"
        safe_filename = filename.replace(" ", "_").replace("/", "-").replace(":", "")
        out_path = self.out_dir / safe_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_plotly_gz(fig, out_path)

        return {
            "type": "error_distribution",
            "parameter": param_name,
            "strategy": metrics.strategy_name,
            "path": out_path.as_posix(),
            "title": f"Error Distribution: {param_name}",
        }

    def plot_convergence_steps(
        self, metrics: StrategyMetrics, param_name: str, gen_name: str, noise_name: str
    ) -> dict[str, Any]:
        """Generate a histogram of steps to convergence."""
        steps = metrics.convergence_steps.get(param_name, [])
        if not steps:
            return {}

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=steps, name="Steps to Convergence", marker_color="royalblue", opacity=0.75, nbinsx=15)
        )

        title = (
            f"Steps to Convergence: {param_name}<br>"
            f"Strategy: {metrics.strategy_name} | Generator: {gen_name} | Noise: {noise_name}"
        )
        fig.update_layout(title=title, xaxis_title="Steps", yaxis_title="Count", template="plotly_white")

        filename = f"conv_steps_{metrics.strategy_name}_{param_name}.json.gz"
        safe_filename = filename.replace(" ", "_").replace("/", "-").replace(":", "")
        out_path = self.out_dir / safe_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_plotly_gz(fig, out_path)

        return {
            "type": "convergence_steps",
            "parameter": param_name,
            "strategy": metrics.strategy_name,
            "path": out_path.as_posix(),
            "title": f"Convergence Steps: {param_name}",
        }

    def plot_sobol_difference(self, metrics: StrategyMetrics, gen_name: str, noise_name: str) -> dict[str, Any]:
        """Generate a histogram plot showing the difference in measurements vs simple Sobol sweep."""
        diffs = getattr(metrics, "sobol_differences", [])
        if not diffs:
            return {}

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=diffs, name="Sweep Savings", marker_color="orange", opacity=0.75, nbinsx=15))

        title = (
            f"Sobol Sweep Measurement Savings<br>"
            f"Strategy: {metrics.strategy_name} | Generator: {gen_name} | Noise: {noise_name}"
        )
        fig.update_layout(title=title, xaxis_title="Measurements Saved", yaxis_title="Count", template="plotly_white")

        filename = f"sobol_diff_{metrics.strategy_name}.json.gz"
        safe_filename = filename.replace(" ", "_").replace("/", "-").replace(":", "")
        out_path = self.out_dir / safe_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_plotly_gz(fig, out_path)

        return {
            "type": "sobol_difference",
            "strategy": metrics.strategy_name,
            "path": out_path.as_posix(),
            "title": "Sobol Sweep Measurement Savings",
        }
