"""
Evaluation metrics for Bayesian Inference locators.
"""

from __future__ import annotations

import contextlib
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
        crb_std_history: (Optional) List of Cramér-Rao Bound (std) values.
        mse_history: (Optional) List of Mean Squared Error values over time.
        mae_history: (Optional) List of Mean Absolute Error values over time.
        cumulative_info_gain: (Optional) List of cumulative information gain.
        convergence_step: (Optional) Step number where convergence was reached.
        ground_truth: (Optional) Dictionary of true parameter values.
    """

    uncertainty_history: list[float]
    entropy_history: list[float]
    parameter_history: pl.DataFrame
    error_history: list[float] | None = None
    z_score_history: list[float] | None = None
    crb_std_history: list[float] | None = None
    mse_history: list[float] | None = None
    mae_history: list[float] | None = None
    cumulative_info_gain: list[float] | None = None
    convergence_step: int | None = None
    ground_truth: dict[str, float] | None = None

    @classmethod
    def from_history(
        cls,
        parameter_history: list[dict[str, float]] | pl.DataFrame,
        ground_truth: dict[str, float] | None = None,
        measurement_history: list[float] | list[dict[str, float]] | pl.DataFrame | None = None,
        noise_model: str = "gaussian",
        noise_params: dict[str, float] | None = None,
        param_name: str = "frequency",
        convergence_threshold: float = 1e6,  # Example default: 1 MHz
    ) -> BayesianMetrics:
        """
        Calculate metrics from a history of parameter estimates.

        Args:
            parameter_history: List of estimate dictionaries or DataFrame.
            ground_truth: Dictionary of true parameter values (e.g., {'frequency': 2.87e9}).
            measurement_history: Sequence of measurements (x values) for CRB calculation.
            noise_model: Noise model used in simulation ("gaussian" or "poisson").
            noise_params: Parameters for noise model (e.g. {'sigma': 0.05}).
            param_name: The main parameter to track error/z-score/CRB for (default: "frequency").
            convergence_threshold: Uncertainty threshold to define convergence step.
        """
        df = pl.DataFrame(parameter_history) if isinstance(parameter_history, list) else parameter_history

        uncertainty = df["uncertainty"].to_list() if "uncertainty" in df.columns else []
        entropy = df["entropy"].to_list() if "entropy" in df.columns else []

        # Calculate Cumulative Information Gain
        cumulative_info_gain = None
        if entropy:
            initial_entropy = entropy[0]
            # Info Gain = H_initial - H_current
            cumulative_info_gain = [initial_entropy - e for e in entropy]

        # Calculate Convergence Step
        convergence_step = None
        if uncertainty:
            for i, unc in enumerate(uncertainty):
                if unc < convergence_threshold:
                    convergence_step = i
                    break

        metrics = cls(
            uncertainty_history=uncertainty,
            entropy_history=entropy,
            parameter_history=df,
            ground_truth=ground_truth,
            cumulative_info_gain=cumulative_info_gain,
            convergence_step=convergence_step,
        )

        if ground_truth and param_name in ground_truth and param_name in df.columns:
            true_val = ground_truth[param_name]
            estimates = df[param_name].to_numpy()

            # Calculate Error
            error_arr = np.abs(estimates - true_val)
            metrics.error_history = error_arr.tolist()
            metrics.mae_history = [float(np.mean(error_arr[: i + 1])) for i in range(len(error_arr))]
            metrics.mse_history = [float(np.mean((estimates[: i + 1] - true_val) ** 2)) for i in range(len(estimates))]

            # Calculate Z-Score (Error / Uncertainty)
            if metrics.uncertainty_history:
                uncert_arr = np.array(metrics.uncertainty_history)
                # Avoid division by zero
                uncert_arr[uncert_arr == 0] = 1e-15
                metrics.z_score_history = ((estimates - true_val) / uncert_arr).tolist()

            # Calculate CRB if measurement history is provided
            if measurement_history is not None:
                cls._calculate_and_set_crb(metrics, measurement_history, ground_truth, noise_model, noise_params)

        return metrics

    @staticmethod
    def _calculate_and_set_crb(metrics, measurement_history, ground_truth, noise_model, noise_params):
        from nvision.sim.locs.nv_center.fisher_information import calculate_crb

        # Extract x values from measurement history
        if isinstance(measurement_history, pl.DataFrame):
            xs = measurement_history["x"].to_list() if "x" in measurement_history.columns else []
        elif isinstance(measurement_history, list):
            if measurement_history and isinstance(measurement_history[0], dict):
                xs = [m.get("x", m.get("frequency", 0.0)) for m in measurement_history]
            else:
                xs = measurement_history
        else:
            xs = []

        # CRB requires frequency, linewidth, and amplitude
        required_keys = {"frequency", "linewidth", "amplitude"}
        # CRB requires frequency, linewidth, and amplitude
        required_keys = {"frequency", "linewidth", "amplitude"}
        if xs and ground_truth and required_keys.issubset(ground_truth.keys()):
            with contextlib.suppress(Exception):
                metrics.crb_std_history = calculate_crb(
                    xs, ground_truth, noise_model=noise_model, noise_params=noise_params
                )

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
        if self.crb_std_history:
            # Match lengths if CRB history is shorter/longer (should match steps)
            # Generally len(xs) == len(steps)
            length = len(self.uncertainty_history)
            crb = (
                self.crb_std_history[:length] + [None] * (length - len(self.crb_std_history))
                if len(self.crb_std_history) < length
                else self.crb_std_history[:length]
            )
            data["crb_std"] = crb
        if self.cumulative_info_gain:
            data["info_gain"] = self.cumulative_info_gain

        return pl.DataFrame(data)

    def summary(self) -> dict[str, float]:
        """Return a summary of final metrics."""
        if not self.uncertainty_history:
            return {}

        summary = {
            "final_uncertainty": self.uncertainty_history[-1],
            "final_entropy": self.entropy_history[-1],
        }

        if self.convergence_step is not None:
            summary["convergence_step"] = float(self.convergence_step)

        if self.error_history:
            summary["final_error"] = self.error_history[-1]
            summary["mean_error"] = np.mean(self.error_history)

        if self.z_score_history:
            summary["final_z_score"] = self.z_score_history[-1]

        if self.crb_std_history:
            summary["final_crb_std"] = self.crb_std_history[-1]

        return summary
