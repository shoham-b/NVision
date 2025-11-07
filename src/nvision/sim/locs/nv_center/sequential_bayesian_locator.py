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
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, voigt_profile
from scipy.stats import cauchy

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class NVCenterSequentialBayesianLocatorSingle(Locator):
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
    distribution: str = "lorentzian"
    gaussian_width_prior: tuple[float, float] = (1e6, 50e6)
    split_prior: tuple[float, float] = (1e6, 10e6)
    k_np_prior: tuple[float, float] = (2.0, 4.0)
    amplitude_prior: tuple[float, float] = (0.01, 1.0)
    background_prior: tuple[float, float] = (0.99, 1.01)
    bo_enabled: bool = False
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

        self._unscaled_prior_bounds = self.prior_bounds
        self._unscaled_linewidth_prior = self.linewidth_prior
        self._unscaled_gaussian_width_prior = self.gaussian_width_prior
        self._unscaled_split_prior = self.split_prior

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
            "amplitude": np.mean(self.amplitude_prior),
            "background": np.mean(self.background_prior),
            "uncertainty": np.inf,
            "gaussian_width": np.mean(self.gaussian_width_prior),
            "split": np.mean(self.split_prior),
            "k_np": np.mean(self.k_np_prior),
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
        self._bo = None
        self._bo_utility = None

    def odmr_model(
        self,
        frequency: float | np.ndarray,
        params: dict[str, float | np.ndarray],
    ) -> np.ndarray:
        f0 = params["frequency"]
        gamma = params["linewidth"]
        amplitude = params["amplitude"]
        bg = params["background"]

        if self.distribution == "lorentzian":
            scale = gamma / 2
            lorentzian = amplitude * math.pi * scale * cauchy.pdf(frequency, loc=f0, scale=scale)
            return bg - lorentzian

        if self.distribution == "voigt":
            sigma = params["gaussian_width"]
            peak_val = voigt_profile(0, sigma, gamma)
            if peak_val < 1e-9:
                return np.full_like(f0, bg, dtype=float)
            profile = voigt_profile(frequency - f0, sigma, gamma)
            return bg - amplitude * profile / peak_val

        if self.distribution == "voigt-zeeman":
            split = params["split"]
            k_np = params["k_np"]
            sigma = max(split / 10.0, 1e-9)
            peak_val = voigt_profile(0, sigma, gamma)
            if peak_val < 1e-12:
                return np.full_like(f0, bg, dtype=float)

            f_left = f0 - split
            f_center = f0
            f_right = f0 + split

            w_left = 1.0 / max(k_np, 1e-9)
            w_center = 1.0
            w_right = max(k_np, 1e-9)

            v_left = voigt_profile(frequency - f_left, sigma, gamma) / peak_val
            v_center = voigt_profile(frequency - f_center, sigma, gamma) / peak_val
            v_right = voigt_profile(frequency - f_right, sigma, gamma) / peak_val

            composite = w_left * v_left + w_center * v_center + w_right * v_right
            return bg - amplitude * composite / max(k_np, 1e-9)

        raise ValueError(f"Unknown distribution: {self.distribution}")

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

    def likelihood(
        self,
        measurement: dict[str, float],
        params: dict[str, float | np.ndarray],
    ) -> np.ndarray:
        measurement = self._coerce_measurement(measurement)
        predicted = self.odmr_model(measurement["x"], params)
        observed = np.array(measurement["signal_values"])
        if observed.ndim == 1:
            observed = observed[:, np.newaxis]  # Prepare for broadcasting

        sigma = 0.05  # Placeholder

        if self.noise_model == "gaussian":
            return -0.5 * ((observed - predicted) / sigma) ** 2 - 0.5 * np.log(2 * np.pi * sigma**2)
        if self.noise_model == "poisson":
            safe_predicted = np.maximum(predicted, 1e-9)
            if isinstance(observed, np.ndarray):
                lgamma_values = np.vectorize(math.lgamma)(observed + 1)
            else:
                lgamma_values = math.lgamma(observed + 1)
            return observed * np.log(safe_predicted) - safe_predicted - lgamma_values

        raise ValueError(f"Unknown noise model: {self.noise_model}")

    def _get_optim_config(self):
        if self.distribution == "lorentzian":
            param_keys = ["linewidth", "amplitude", "background"]
            bounds = [self.linewidth_prior, self.amplitude_prior, self.background_prior]
        elif self.distribution == "voigt":
            param_keys = ["linewidth", "gaussian_width", "amplitude", "background"]
            bounds = [
                self.linewidth_prior,
                self.gaussian_width_prior,
                self.amplitude_prior,
                self.background_prior,
            ]
        elif self.distribution == "voigt-zeeman":
            # Match generator: sigma derived from split; include k_np parameter
            param_keys = ["linewidth", "split", "k_np", "amplitude", "background"]
            bounds = [
                self.linewidth_prior,
                self.split_prior,
                self.k_np_prior,
                self.amplitude_prior,
                self.background_prior,
            ]
        else:
            param_keys = []
            bounds = []
        return param_keys, bounds

    def _objective_func(self, param_values, param_keys):
        params = self.current_estimates.copy()
        for key, value in zip(param_keys, param_values, strict=False):
            params[key] = value

        params["frequency"] = self.current_estimates["frequency"]

        if not self.measurement_history:
            return 0.0

        history_df = pl.DataFrame(self.measurement_history)
        measurements_x = history_df["x"].to_numpy()
        measurements_y = history_df["signal_values"].to_numpy()

        predicted = self.odmr_model(measurements_x, params)
        observed = measurements_y
        sigma = 0.05  # Placeholder

        if self.noise_model == "gaussian":
            log_lik_array = -0.5 * ((observed - predicted) / sigma) ** 2 - 0.5 * np.log(
                2 * np.pi * sigma**2
            )
        elif self.noise_model == "poisson":
            predicted[predicted <= 0] = 1e-9
            lgamma_observed = np.vectorize(math.lgamma)(observed + 1)
            log_lik_array = observed * np.log(predicted) - predicted - lgamma_observed
        else:
            raise ValueError(f"Unknown noise model: {self.noise_model}")

        total_log_likelihood = np.sum(log_lik_array)
        if np.isneginf(total_log_likelihood):
            return np.inf

        return -total_log_likelihood

    def _optimize_lineshape_params(self):
        if len(self.measurement_history) < 5:
            return

        param_keys, bounds = self._get_optim_config()
        if not param_keys:
            return

        initial_guess = [self.current_estimates[key] for key in param_keys]

        sanitized_bounds = []
        for b in bounds:
            low, high = b
            if low == high:
                high = low + 1e-9
            sanitized_bounds.append((low, high))

        result = minimize(
            lambda p: self._objective_func(p, param_keys),
            initial_guess,
            bounds=sanitized_bounds,
            method="L-BFGS-B",
        )

        if result.success:
            for key, value in zip(param_keys, result.x, strict=False):
                self.current_estimates[key] = value

    def _calculate_log_likelihoods(
        self,
        measurement: dict[str, float],
        base_params: dict[str, float],
    ) -> np.ndarray:
        """Calculate log-likelihoods over the frequency grid using vectorized operations."""
        params = base_params.copy()
        params["frequency"] = self.freq_grid
        return self.likelihood(measurement, params)

    def update_posterior(self, measurement: dict[str, float]):
        measurement = self._coerce_measurement(measurement)

        base_params = {k: v for k, v in self.current_estimates.items() if k != "frequency"}
        log_likelihoods = self._calculate_log_likelihoods(measurement, base_params)

        log_posterior = np.log(self.freq_posterior + 1e-300) + log_likelihoods
        log_posterior -= logsumexp(log_posterior)
        self.freq_posterior = np.exp(log_posterior)

        # Vectorized calculation of estimates using Polars
        df = pl.DataFrame({"freq": self.freq_grid, "posterior": self.freq_posterior})
        est_freq = (df["freq"] * df["posterior"]).sum()
        self.current_estimates["frequency"] = est_freq
        self.current_estimates["uncertainty"] = (
            ((df["freq"] - est_freq).pow(2) * df["posterior"]).sum()
        ) ** 0.5

        self.measurement_history.append(measurement.copy())
        self.posterior_history.append(self.freq_posterior.copy())
        self._optimize_lineshape_params()

    def expected_information_gain(self, test_frequency: float) -> float:
        current_entropy = -np.sum(self.freq_posterior * np.log(self.freq_posterior + 1e-300))
        n_samples = min(self.n_monte_carlo // 10, 100)

        # Vectorized Monte Carlo Sampling
        freq_indices = np.random.choice(self.grid_resolution, size=n_samples, p=self.freq_posterior)
        true_freqs = self.freq_grid[freq_indices]

        true_params = {
            "frequency": true_freqs,
            "linewidth": self.current_estimates["linewidth"],
            "amplitude": self.current_estimates["amplitude"],
            "background": self.current_estimates["background"],
            "gaussian_width": self.current_estimates["gaussian_width"],
            "split": self.current_estimates["split"],
        }
        expected_signals = self.odmr_model(test_frequency, true_params)

        if self.noise_model == "gaussian":
            noise_stds = 0.05 * np.abs(expected_signals) + 0.01
            simulated_intensities = np.random.normal(expected_signals, noise_stds)
        else:  # poisson
            rates = np.maximum(expected_signals, 0.1)
            simulated_intensities = np.random.poisson(rates)

        sim_measurement = {"x": test_frequency, "signal_values": simulated_intensities}

        base_params = {k: v for k, v in self.current_estimates.items() if k != "frequency"}
        log_likelihoods = self._calculate_log_likelihoods(
            sim_measurement, base_params
        )  # Shape: (n_samples, grid_resolution)

        # Vectorized posterior update and entropy calculation
        log_posteriors = np.log(self.freq_posterior + 1e-300) + log_likelihoods
        log_posteriors -= logsumexp(log_posteriors, axis=1, keepdims=True)
        temp_posteriors = np.exp(log_posteriors)

        entropies = -np.sum(temp_posteriors * np.log(temp_posteriors + 1e-300), axis=1)
        expected_entropy = np.mean(entropies)

        info_gain = current_entropy - expected_entropy
        return max(info_gain, 0.0)

    def plot_bo(
        self,
        output_path: Path,
        fig_size: tuple[int, int] = (12, 8),
        dpi: int = 100,
    ) -> Path | None:
        warnings.warn(
            "Bayesian Optimization plotting is disabled as the dependency was removed.",
            stacklevel=1,
        )
        return None

    def plot_posterior_history(self, output_path: Path) -> Path | None:
        if not self.posterior_history:
            return None

        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        num_posteriors = len(self.posterior_history)
        indices_to_plot = np.linspace(0, num_posteriors - 1, 5, dtype=int)

        for i in indices_to_plot:
            ax.plot(self.freq_grid, self.posterior_history[i], label=f"Step {i}")

        ax.set_title("Posterior Distribution Evolution")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Probability Density")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path.as_posix(), bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_convergence_stats(self, output_path: Path) -> Path | None:
        if not self.utility_history:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=100, sharex=True)

        uncertainty_history = [
            np.sqrt(np.sum((self.freq_grid - np.sum(self.freq_grid * p)) ** 2 * p))
            for p in self.posterior_history
        ]

        steps = range(len(uncertainty_history))
        ax1.plot(steps, uncertainty_history, marker="o")
        ax1.set_title("Uncertainty Convergence")
        ax1.set_ylabel("Uncertainty (Hz)")
        ax1.grid(True)

        utility_steps = range(len(self.utility_history))
        ax2.plot(utility_steps, self.utility_history, marker="o", color="orange")
        ax2.set_title("Expected Information Gain")
        ax2.set_xlabel("Measurement Step")
        ax2.set_ylabel("Utility")
        ax2.grid(True)

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

    def _rescale_priors_if_needed(self, scan: ScanBatch | None):
        """Rescale priors if scan bounds suggest a normalized coordinate system."""
        if scan is None:
            return

        scan_bounds = (scan.x_min, scan.x_max)
        is_normalized_scan = np.isclose(scan_bounds[0], 0.0) and np.isclose(scan_bounds[1], 1.0)
        is_unscaled = np.isclose(self.prior_bounds[0], self._unscaled_prior_bounds[0])

        if is_normalized_scan and is_unscaled:
            scale_factor = self._unscaled_prior_bounds[1] - self._unscaled_prior_bounds[0]
            if scale_factor > 0:
                self.prior_bounds = (0.0, 1.0)
                self.linewidth_prior = tuple(
                    p / scale_factor for p in self._unscaled_linewidth_prior
                )
                self.gaussian_width_prior = tuple(
                    p / scale_factor for p in self._unscaled_gaussian_width_prior
                )
                self.split_prior = tuple(p / scale_factor for p in self._unscaled_split_prior)
                self.reset_run_state()

    def propose_next(
        self,
        history: Sequence | pl.DataFrame,
        scan: ScanBatch | None = None,
        repeats: pl.DataFrame | None = None,
    ) -> float | pl.DataFrame:
        """Propose the next measurement point.

        This method supports both the old (single-value) and new (batched) interfaces:
        1. If called with just history and scan, returns a single float (old interface)
        2. If called with repeats, returns a DataFrame with repeat_id and x columns (new interface)
        """
        if repeats is not None:
            # Batched interface - delegate to the batched adapter
            from nvision.sim.locs.nv_center._bayesian_adapter import (
                NVCenterSequentialBayesianLocatorBatched,
            )

            adapter = NVCenterSequentialBayesianLocatorBatched(
                max_evals=self.max_evals,
                prior_bounds=self.prior_bounds,
                noise_model=self.noise_model,
                acquisition_function=self.acquisition_function,
                convergence_threshold=self.convergence_threshold,
                min_uncertainty_reduction=self.min_uncertainty_reduction,
                n_monte_carlo=self.n_monte_carlo,
                grid_resolution=self.grid_resolution,
                linewidth_prior=self.linewidth_prior,
                distribution=self.distribution,
                gaussian_width_prior=self.gaussian_width_prior,
                split_prior=self.split_prior,
                amplitude_prior=self.amplitude_prior,
                background_prior=self.background_prior,
                bo_enabled=self.bo_enabled,
                bo_acq_func=self.bo_acq_func,
                bo_kappa=self.bo_kappa,
                bo_xi=self.bo_xi,
                bo_random_state=self.bo_random_state,
                utility_history_window=self.utility_history_window,
                n_warmup=self.n_warmup,
            )
            return adapter.propose_next(history, repeats, scan)

        # Original single-value interface
        if not isinstance(history, pl.DataFrame):
            history = pl.DataFrame(history)

        self._rescale_priors_if_needed(scan)

        self._ingest_history(history)
        next_freq, _ = self._optimize_acquisition(self.prior_bounds)
        return float(next_freq)

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
