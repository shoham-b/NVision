"""
Sequential Bayesian Experiment Design Locator for ODMR of NV Centers

This module implements a Sequential Bayesian Experiment Design (SBED) strategy
for Optically Detected Magnetic Resonance (ODMR) measurements of Nitrogen-Vacancy
centers in diamond, based on the methodology described in:

"Sequential Bayesian Experiment Design for Optically Detected Magnetic Resonance
of Nitrogen-Vacancy Centers" by Dushenko et al., Phys. Rev. Applied 14, 054036 (2020)

The implementation provides order-of-magnitude speedup compared to conventional
frequency-swept measurements by using Bayesian inference to adaptively select
optimal measurement frequencies in real-time.

Key Features:
- Bayesian posterior updating with measurement data
- Information-theoretic utility functions for optimal setting selection
- Real-time adaptive frequency selection
- ODMR-specific Lorentzian lineshape modeling
- Support for both Gaussian and Poisson noise models
- Multi-peak detection capabilities
- Convergence criteria based on uncertainty reduction
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from nvision.sim.locs.base import Locator, ScanBatch

try:  # pragma: no cover - import guarded for optional dependency resilience
    from bayes_opt import BayesianOptimization, UtilityFunction
except ImportError:  # pragma: no cover
    BayesianOptimization = None  # type: ignore[assignment]
    UtilityFunction = None  # type: ignore[assignment]


@dataclass
class NVCenterSequentialBayesianLocator(Locator):
    """
    An advanced locator implementing Sequential Bayesian Experiment Design (SBED).

    This locator implements the methodology from Dushenko et al. (2020) for
    optimal Bayesian experimental design in NV center magnetometry, providing
    significant speedup over conventional frequency-swept measurements.

    The strategy involves maintaining a posterior probability distribution over the
    parameter space (e.g., resonance frequency). At each step, it selects the next
    measurement point that is expected to provide the most information (i.e.,
    maximizes the expected reduction in the posterior's entropy), allowing it to
    converge on the true parameters with minimal measurements.
    Attributes:
        max_evals: Maximum number of measurements
        prior_bounds: Prior bounds for frequency parameters (Hz)
        noise_model: Type of noise model ("gaussian", "poisson")
        acquisition_function: Utility function for next measurement selection
        convergence_threshold: Threshold for parameter convergence (Hz)
        min_uncertainty_reduction: Minimum required uncertainty reduction per step
        n_monte_carlo: Number of Monte Carlo samples for utility estimation
        grid_resolution: Resolution for frequency grid discretization
        linewidth_prior: Prior bounds for linewidth parameter (Hz)
    """

    max_evals: int = 50
    prior_bounds: tuple[float, float] = (2.6e9, 3.1e9)
    noise_model: str = "gaussian"
    acquisition_function: str = "expected_information_gain"
    convergence_threshold: float = 1e-8
    min_uncertainty_reduction: float = 1e-9
    n_monte_carlo: int = 100
    grid_resolution: int = 1000
    linewidth_prior: tuple[float, float] = (1e6, 50e6)
    bo_enabled: bool = True
    bo_acq_func: str = "ucb"
    bo_kappa: float = 2.576
    bo_xi: float = 0.0
    bo_random_state: int | None = None
    utility_history_window: int = 9
    n_warmup: int = 10

    def __post_init__(self):
        """Initialize the locator after dataclass creation."""
        if self.n_warmup >= self.max_evals:
            raise ValueError("n_warmup must be smaller than max_evals")
        self._bo: Any = None
        self._bo_utility: Any = None
        self.reset_posterior()
        self.measurement_history: list[dict[str, float]] = []
        self.utility_history: list[float] = []
        self.posterior_history: list[np.ndarray] = []

    def reset_posterior(self):
        """Reset posterior distributions to priors."""
        self.freq_grid = np.linspace(
            self.prior_bounds[0], self.prior_bounds[1], self.grid_resolution
        )
        self.freq_posterior = np.ones(self.grid_resolution) / self.grid_resolution
        self.current_estimates = {
            "frequency": np.mean(self.prior_bounds),
            "linewidth": np.mean(self.linewidth_prior),
            "amplitude": 1.0,
            "background": 0.1,
            "uncertainty": np.inf,
        }
        self._init_bayes_optimizer()

    def reset_run_state(self) -> None:
        """Clear accumulated histories and reinitialize the posterior."""
        if hasattr(self, "measurement_history"):
            self.measurement_history.clear()
        if hasattr(self, "posterior_history"):
            self.posterior_history.clear()
        if hasattr(self, "utility_history"):
            self.utility_history.clear()
        self.reset_posterior()

    def _init_bayes_optimizer(self) -> None:
        if not self.bo_enabled or BayesianOptimization is None or UtilityFunction is None:
            self._bo = None
            self._bo_utility = None
            return

        pbounds = {"freq": (float(self.prior_bounds[0]), float(self.prior_bounds[1]))}
        self._bo = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=0,
            random_state=self.bo_random_state,
            allow_duplicate_points=True,
        )
        self._bo_utility = UtilityFunction(
            kind=self.bo_acq_func,
            kappa=self.bo_kappa,
            xi=self.bo_xi,
        )

    def odmr_model(self, frequency: float, params: dict[str, float]) -> float:
        f0 = params["frequency"]
        gamma = params["linewidth"]
        amplitude = params["amplitude"]
        bg = params["background"]
        lorentzian = amplitude * (gamma / 2) ** 2 / ((frequency - f0) ** 2 + (gamma / 2) ** 2)
        return bg - lorentzian

    def _coerce_measurement(self, measurement: dict[str, float]) -> dict[str, float]:
        if "frequency" in measurement:
            return {
                "x": measurement.get("x", measurement["frequency"]),
                "signal_values": measurement.get("signal_values", measurement["intensity"]),
                "uncertainty": measurement.get("uncertainty", 0.05),
            }
        if "x" in measurement and "signal_values" in measurement:
            return measurement
        raise KeyError("measurement must include either frequency/intensity or x/signal_values")

    def likelihood(self, measurement: dict[str, float], params: dict[str, float]) -> float:
        measurement = self._coerce_measurement(measurement)
        predicted = self.odmr_model(measurement["x"], params)
        observed = measurement["signal_values"]
        sigma = 0.05  # Placeholder

        if self.noise_model == "gaussian":
            return -0.5 * ((observed - predicted) / sigma) ** 2 - 0.5 * np.log(2 * np.pi * sigma**2)
        elif self.noise_model == "poisson":
            if predicted <= 0:
                return -np.inf
            return observed * np.log(predicted) - predicted - math.lgamma(observed + 1)
        else:
            raise ValueError(f"Unknown noise model: {self.noise_model}")

    def update_posterior(self, measurement: dict[str, float]):
        measurement = self._coerce_measurement(measurement)
        log_likelihoods = np.zeros(self.grid_resolution)
        for i, freq in enumerate(self.freq_grid):
            params = {
                "frequency": freq,
                "linewidth": self.current_estimates["linewidth"],
                "amplitude": self.current_estimates["amplitude"],
                "background": self.current_estimates["background"],
            }
            log_likelihoods[i] = self.likelihood(measurement, params)

        log_posterior = np.log(self.freq_posterior + 1e-300) + log_likelihoods
        log_posterior -= logsumexp(log_posterior)
        self.freq_posterior = np.exp(log_posterior)

        self.current_estimates["frequency"] = np.sum(self.freq_grid * self.freq_posterior)
        self.current_estimates["uncertainty"] = np.sqrt(
            np.sum(
                (self.freq_grid - self.current_estimates["frequency"]) ** 2 * self.freq_posterior
            )
        )
        self.measurement_history.append(measurement.copy())
        self.posterior_history.append(self.freq_posterior.copy())
        if self.bo_enabled and self._bo is not None:
            self._bo.register(
                params={"freq": float(measurement["x"])},
                target=-float(measurement["signal_values"]),
            )

    def expected_information_gain(self, test_frequency: float) -> float:
        current_entropy = -np.sum(self.freq_posterior * np.log(self.freq_posterior + 1e-300))
        expected_entropy = 0.0
        n_samples = min(self.n_monte_carlo // 10, 100)

        for _ in range(n_samples):
            freq_idx = np.random.choice(self.grid_resolution, p=self.freq_posterior)
            true_freq = self.freq_grid[freq_idx]
            true_params = {
                "frequency": true_freq,
                "linewidth": self.current_estimates["linewidth"],
                "amplitude": self.current_estimates["amplitude"],
                "background": self.current_estimates["background"],
            }
            expected_signal = self.odmr_model(test_frequency, true_params)

            if self.noise_model == "gaussian":
                noise_std = 0.05 * abs(expected_signal) + 0.01
                simulated_intensity = np.random.normal(expected_signal, noise_std)
            else:
                rate = max(expected_signal, 0.1)
                simulated_intensity = np.random.poisson(rate)

            sim_measurement = {"x": test_frequency, "signal_values": simulated_intensity}

            temp_posterior = self.freq_posterior.copy()
            log_likelihoods = np.zeros(self.grid_resolution)
            for i, freq in enumerate(self.freq_grid):
                params = {
                    "frequency": freq,
                    "linewidth": self.current_estimates["linewidth"],
                    "amplitude": self.current_estimates["amplitude"],
                    "background": self.current_estimates["background"],
                }
                log_likelihoods[i] = self.likelihood(sim_measurement, params)

            log_temp_posterior = np.log(temp_posterior + 1e-300) + log_likelihoods
            log_temp_posterior -= logsumexp(log_temp_posterior)
            temp_posterior = np.exp(log_temp_posterior)
            entropy = -np.sum(temp_posterior * np.log(temp_posterior + 1e-300))
            expected_entropy += entropy

        expected_entropy /= n_samples
        info_gain = current_entropy - expected_entropy
        return max(info_gain, 0.0)

    def plot_bo(
        self,
        output_path: Path,
        fig_size: tuple[int, int] = (12, 8),
        dpi: int = 100,
    ) -> Path | None:
        if (
            not self.bo_enabled
            or self._bo is None
            or self._bo_utility is None
            or output_path is None
        ):
            return None

        if getattr(self._bo.space, "params", None) is None or self._bo.space.params.size == 0:
            return None

        try:
            x = np.linspace(
                self.prior_bounds[0], self.prior_bounds[1], self.grid_resolution
            ).reshape(-1, 1)
            mu, sigma = self._bo._gp.predict(x, return_std=True)
        except Exception:
            return None

        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        ax.plot(x, mu, lw=2, label="Surrogate Model (GP)")
        ax.fill_between(
            x.ravel(),
            mu - 1.96 * sigma,
            mu + 1.96 * sigma,
            alpha=0.2,
            label="95% Confidence Interval",
        )

        params = self._bo.space.params.ravel()
        targets = self._bo.space.target
        if params.size and targets.size:
            ax.scatter(
                params,
                targets,
                c="red",
                s=50,
                zorder=10,
                edgecolors=(0, 0, 0),
                label="Observed Points",
            )

        try:
            utility = self._bo_utility.utility(x, self._bo._gp, 0)
        except Exception:
            utility = None

        if utility is not None:
            ax2 = ax.twinx()
            ax2.plot(x, utility, "g--", label=f"Acquisition Function ({self.bo_acq_func})")
            ax2.set_ylabel("Acquisition Function Value", color="green")
            ax2.tick_params(axis="y", labelcolor="green")
            ax2.legend(loc="upper right")

        ax.set_title(f"Bayesian Optimization after {len(self.measurement_history)} measurements")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Objective (negative signal)")
        ax.legend(loc="upper left")
        ax.grid(True)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path.as_posix(), bbox_inches="tight")
        plt.close(fig)
        return output_path

    def _history_iter(self, history: Sequence | pl.DataFrame) -> Iterable[dict[str, float]]:
        if isinstance(history, pl.DataFrame):
            yield from history.iter_rows(named=True)
        else:
            for entry in history:
                if isinstance(entry, dict):
                    yield entry
                else:
                    entry_map = {
                        "x": getattr(entry, "x", None),
                        "signal_values": getattr(
                            entry, "signal_values", getattr(entry, "intensity", None)
                        ),
                        "uncertainty": getattr(entry, "uncertainty", 0.05),
                    }
                    if entry_map["x"] is None or entry_map["signal_values"] is None:
                        raise KeyError("History entries must provide x and signal_values/intensity")
                    yield entry_map

    def _ingest_history(self, history: Sequence | pl.DataFrame) -> None:
        for row in self._history_iter(history):
            is_new = not any(existing["x"] == row["x"] for existing in self.measurement_history)
            if is_new:
                self.update_posterior(row)

    def _optimize_acquisition(self, domain: tuple[float, float]) -> tuple[float, float]:
        domain_low, domain_high = domain

        def negative_utility(freq: float) -> float:
            try:
                return -self.expected_information_gain(freq)
            except Exception as e:
                warnings.warn(f"Error in utility calculation: {e}", stacklevel=2)
                return 0.0

        try:
            result = minimize_scalar(
                negative_utility,
                bounds=(domain_low, domain_high),
                method="bounded",
                options={"maxiter": 100},
            )
            optimal_freq = result.x
            utility = -result.fun
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}, using fallback strategy", stacklevel=2)
            optimal_freq = self.current_estimates["frequency"]
            utility = 0.0

        return optimal_freq, utility

    def propose_next(self, history: Sequence | pl.DataFrame, scan: ScanBatch) -> float:
        self._ingest_history(history)

        if len(self.measurement_history) < self.n_warmup:
            return np.linspace(scan.x_min, scan.x_max, self.n_warmup)[len(self.measurement_history)]

        if self.bo_enabled and self._bo is not None and self._bo_utility is not None:
            taken = {round(m["x"], 12) for m in self.measurement_history}
            proposal = self.current_estimates["frequency"]
            for _ in range(10):
                suggestion = self._bo.suggest(self._bo_utility)["freq"]
                suggestion = float(np.clip(suggestion, scan.x_min, scan.x_max))
                if round(suggestion, 12) not in taken:
                    proposal = suggestion
                    break
            self.utility_history.append(float("nan"))
            return proposal

        optimal_freq, utility = self._optimize_acquisition((scan.x_min, scan.x_max))
        self.utility_history.append(utility)
        return optimal_freq

    def should_stop(self, history: Sequence | pl.DataFrame, scan: ScanBatch) -> bool:
        hist_len = history.height if isinstance(history, pl.DataFrame) else len(history)
        if hist_len >= self.max_evals:
            return True
        if self.current_estimates["uncertainty"] < self.convergence_threshold:
            return True
        return len(self.utility_history) == self.utility_history_window and all(
            u < self.min_uncertainty_reduction for u in self.utility_history
        )

    def finalize(self, history: Sequence | pl.DataFrame, scan: ScanBatch) -> dict[str, float]:
        self._ingest_history(history)

        posterior_smooth = np.convolve(self.freq_posterior, np.ones(5) / 5, mode="same")
        peaks = []
        threshold = 0.1 * np.max(posterior_smooth)
        for i in range(1, len(posterior_smooth) - 1):
            if (
                posterior_smooth[i] > posterior_smooth[i - 1]
                and posterior_smooth[i] > posterior_smooth[i + 1]
                and posterior_smooth[i] > threshold
            ):
                peaks.append(i)

        n_peaks = len(peaks)
        uncertainty = float(self.current_estimates["uncertainty"])
        if n_peaks == 0:
            max_idx = int(np.argmax(self.freq_posterior))
            x1 = float(self.freq_grid[max_idx])
            return {
                "n_peaks": 1.0,
                "x1_hat": x1,
                "x2_hat": math.nan,
                "x3_hat": math.nan,
                "uncert": uncertainty,
                "uncert_pos": uncertainty,
            }

        peak_strengths = [(idx, self.freq_posterior[idx]) for idx in peaks]
        peak_strengths.sort(key=lambda item: item[1], reverse=True)
        top_peaks = [float(self.freq_grid[idx]) for idx, _strength in peak_strengths[:3]]
        top_peaks.sort()

        result: dict[str, float] = {
            "n_peaks": float(len(top_peaks)),
            "uncert": uncertainty,
            "uncert_pos": uncertainty,
            "x1_hat": top_peaks[0] if len(top_peaks) >= 1 else math.nan,
            "x2_hat": top_peaks[1] if len(top_peaks) >= 2 else math.nan,
            "x3_hat": top_peaks[2] if len(top_peaks) >= 3 else math.nan,
        }
        return result
