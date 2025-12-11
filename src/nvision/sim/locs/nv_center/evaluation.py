"""
Evaluation metrics for Bayesian Inference locators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class BayesianMetrics:
    """
    Container and calculator for Bayesian evaluation metrics.

    Attributes:
        uncertainty_history: List of uncertainty values (std dev) over steps.
        entropy_history: List of entropy values over steps.
        parameter_history: DataFrame of parameter estimates over steps.
        error_history: (Optional) List of absolute errors w.r.t ground truth.
        z_score_history: (Optional) List of Z-scores w.r.t ground truth.
        ground_truth: (Optional) Dictionary of true parameter values.
    """

    uncertainty_history: list[float]
    entropy_history: list[float]
    parameter_history: pl.DataFrame
    error_history: list[float] | None = None
    z_score_history: list[float] | None = None
    ground_truth: dict[str, float] | None = None

    @classmethod
    def from_history(
        cls,
        parameter_history: list[dict[str, float]] | pl.DataFrame,
        ground_truth: dict[str, float] | None = None,
        param_name: str = "frequency",
    ) -> "BayesianMetrics":
        """
        Calculate metrics from a history of parameter estimates.

        Args:
            parameter_history: List of estimate dictionaries or DataFrame.
            ground_truth: Dictionary of true parameter values (e.g., {'frequency': 2.87e9}).
            param_name: The main parameter to track error/z-score for (default: "frequency").
        """
        if isinstance(parameter_history, list):
            df = pl.DataFrame(parameter_history)
        else:
            df = parameter_history

        uncertainty = df["uncertainty"].to_list() if "uncertainty" in df.columns else []
        entropy = df["entropy"].to_list() if "entropy" in df.columns else []

        metrics = cls(
            uncertainty_history=uncertainty,
            entropy_history=entropy,
            parameter_history=df,
            ground_truth=ground_truth,
        )

        if ground_truth and param_name in ground_truth and param_name in df.columns:
            true_val = ground_truth[param_name]
            estimates = df[param_name].to_numpy()

            # Calculate Error
            metrics.error_history = np.abs(estimates - true_val).tolist()

            # Calculate Z-Score (Error / Uncertainty)
            if metrics.uncertainty_history:
                uncert_arr = np.array(metrics.uncertainty_history)
                # Avoid division by zero
                uncert_arr[uncert_arr == 0] = 1e-15
                metrics.z_score_history = ((estimates - true_val) / uncert_arr).tolist()

        return metrics

    def to_dataframe(self) -> pl.DataFrame:
        """Export scalar metrics history to a DataFrame."""
        data = {
            "step": list(range(len(self.uncertainty_history))),
            "uncertainty": self.uncertainty_history,
            "entropy": self.entropy_history,
        }
        if self.error_history:
            data["error"] = self.error_history
        if self.z_score_history:
            data["z_score"] = self.z_score_history

        return pl.DataFrame(data)

    def summary(self) -> dict[str, float]:
        """Return a summary of final metrics."""
        if not self.uncertainty_history:
            return {}

        summary = {
            "final_uncertainty": self.uncertainty_history[-1],
            "final_entropy": self.entropy_history[-1],
        }

        if self.error_history:
            summary["final_error"] = self.error_history[-1]
            summary["mean_error"] = np.mean(self.error_history)

        if self.z_score_history:
            summary["final_z_score"] = self.z_score_history[-1]

        return summary
