"""Visualization mixin for strategy metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvision.metrics.types import StrategyMetrics
from nvision.viz._f32_json import dump_gz


class MetricsVizMixin:
    """Mixin for generating advanced metric plots."""

    out_dir: Path

    def plot_error_distribution(
        self, metrics: StrategyMetrics, param_name: str, gen_name: str, noise_name: str
    ) -> dict[str, Any]:
        """Serialize violin data (definition lives in static/graphs/violin.json)."""
        errors = metrics.absolute_errors.get(param_name, [])
        if not errors:
            return {}

        data: dict[str, Any] = {
            "_graph_type": "violin",
            "title": (
                f"Absolute Error Distribution: {param_name}<br>"
                f"Strategy: {metrics.strategy_name} | Generator: {gen_name} | Noise: {noise_name}"
            ),
            "yaxis_title": "Absolute Error",
            "name": metrics.strategy_name,
            "values": [v for v in errors if v is not None],
        }

        filename = f"error_dist_{metrics.strategy_name}_{param_name}.json.gz"
        safe_filename = filename.replace(" ", "_").replace("/", "-").replace(":", "")
        out_path = self.out_dir / safe_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_gz(data, out_path)

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
        """Serialize histogram data (definition lives in static/graphs/histogram.json)."""
        steps = metrics.convergence_steps.get(param_name, [])
        if not steps:
            return {}

        data: dict[str, Any] = {
            "_graph_type": "histogram",
            "title": (
                f"Steps to Convergence: {param_name}<br>"
                f"Strategy: {metrics.strategy_name} | Generator: {gen_name} | Noise: {noise_name}"
            ),
            "xaxis_title": "Steps",
            "yaxis_title": "Count",
            "color": "royalblue",
            "name": "Steps to Convergence",
            "values": [v for v in steps if v is not None],
        }

        filename = f"conv_steps_{metrics.strategy_name}_{param_name}.json.gz"
        safe_filename = filename.replace(" ", "_").replace("/", "-").replace(":", "")
        out_path = self.out_dir / safe_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_gz(data, out_path)

        return {
            "type": "convergence_steps",
            "parameter": param_name,
            "strategy": metrics.strategy_name,
            "path": out_path.as_posix(),
            "title": f"Convergence Steps: {param_name}",
        }

    def plot_sobol_difference(self, metrics: StrategyMetrics, gen_name: str, noise_name: str) -> dict[str, Any]:
        """Serialize histogram data (definition lives in static/graphs/histogram.json)."""
        diffs = getattr(metrics, "sobol_differences", [])
        if not diffs:
            return {}

        data: dict[str, Any] = {
            "_graph_type": "histogram",
            "title": (
                f"Sobol Sweep Measurement Savings<br>"
                f"Strategy: {metrics.strategy_name} | Generator: {gen_name} | Noise: {noise_name}"
            ),
            "xaxis_title": "Measurements Saved",
            "yaxis_title": "Count",
            "color": "orange",
            "name": "Sweep Savings",
            "values": [v for v in diffs if v is not None],
        }

        filename = f"sobol_diff_{metrics.strategy_name}.json.gz"
        safe_filename = filename.replace(" ", "_").replace("/", "-").replace(":", "")
        out_path = self.out_dir / safe_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_gz(data, out_path)

        return {
            "type": "sobol_difference",
            "strategy": metrics.strategy_name,
            "path": out_path.as_posix(),
            "title": "Sobol Sweep Measurement Savings",
        }
