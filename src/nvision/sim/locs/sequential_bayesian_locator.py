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

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.stats
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .obs import Obs


@dataclass
class SequentialBayesianLocator:
    """
    Sequential Bayesian Experiment Design locator for ODMR of NV centers.

    This locator implements the methodology from Dushenko et al. (2020) for
    optimal Bayesian experimental design in NV center magnetometry, providing
    significant speedup over conventional frequency-swept measurements.

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
    prior_bounds: tuple[float, float] = (2.6e9, 3.1e9)  # Typical NV center frequency range (Hz)
    noise_model: str = "gaussian"
    acquisition_function: str = "expected_information_gain"
    convergence_threshold: float = 1e6  # 1 MHz
    min_uncertainty_reduction: float = 0.01
    n_monte_carlo: int = 100
    grid_resolution: int = 1000
    linewidth_prior: tuple[float, float] = (1e6, 50e6)  # Typical linewidth range (Hz)

    def __post_init__(self):
        """Initialize the locator after dataclass creation."""
        self.reset_posterior()

        # History tracking
        self.measurement_history: list[dict[str, float]] = []
        self.utility_history: list[float] = []
        self.posterior_history: list[np.ndarray] = []

    def reset_posterior(self):
        """Reset posterior distributions to priors."""
        # Create frequency grid for discrete approximation
        self.freq_grid = np.linspace(
            self.prior_bounds[0], self.prior_bounds[1], self.grid_resolution
        )

        # Initialize uniform prior over frequency
        self.freq_posterior = np.ones(self.grid_resolution) / self.grid_resolution

        # Parameters for current best estimates
        self.current_estimates = {
            "frequency": np.mean(self.prior_bounds),
            "linewidth": np.mean(self.linewidth_prior),
            "amplitude": 1.0,
            "background": 0.1,
            "uncertainty": np.inf,
        }

    def odmr_model(self, frequency: float, params: dict[str, float]) -> float:
        """
        ODMR lineshape model - Lorentzian for NV centers.

        The NV center ODMR signal typically shows a Lorentzian dip in fluorescence
        intensity at the resonance frequency due to spin-dependent shelving.

        Args:
            frequency: Measurement frequency (Hz)
            params: Model parameters {frequency, linewidth, amplitude, background}

        Returns:
            Expected signal intensity
        """
        f0 = params["frequency"]
        gamma = params["linewidth"]
        amplitude = params["amplitude"]
        bg = params["background"]

        # Lorentzian lineshape (typical for NV centers)
        # ODMR shows a dip in fluorescence, so we subtract from background
        lorentzian = amplitude * (gamma / 2) ** 2 / ((frequency - f0) ** 2 + (gamma / 2) ** 2)
        return bg - lorentzian

    def likelihood(self, measurement: dict[str, float], params: dict[str, float]) -> float:
        """
        Compute log-likelihood of measurement given parameters.

        Args:
            measurement: Measurement data {frequency, intensity, uncertainty}
            params: Model parameters

        Returns:
            Log-likelihood value
        """
        predicted = self.odmr_model(measurement["frequency"], params)
        observed = measurement["intensity"]
        sigma = measurement["uncertainty"]

        if self.noise_model == "gaussian":
            # Gaussian likelihood for continuous measurements
            return -0.5 * ((observed - predicted) / sigma) ** 2 - 0.5 * np.log(2 * np.pi * sigma**2)
        elif self.noise_model == "poisson":
            # Poisson likelihood for photon counting statistics
            if predicted <= 0:
                return -np.inf
            return observed * np.log(predicted) - predicted - scipy.special.gammaln(observed + 1)
        else:
            raise ValueError(f"Unknown noise model: {self.noise_model}")

    def update_posterior(self, measurement: dict[str, float]):
        """
        Update posterior distribution using Bayes' rule.

        This is the core of the Bayesian inference, updating our belief about
        the NV center parameters based on the new measurement.

        Args:
            measurement: New measurement data
        """
        # Update frequency posterior using grid-based approximation
        log_likelihoods = np.zeros(self.grid_resolution)

        for i, freq in enumerate(self.freq_grid):
            # Sample parameters for this frequency
            params = {
                "frequency": freq,
                "linewidth": self.current_estimates["linewidth"],
                "amplitude": self.current_estimates["amplitude"],
                "background": self.current_estimates["background"],
            }
            log_likelihoods[i] = self.likelihood(measurement, params)

        # Update posterior (in log space for numerical stability)
        log_posterior = np.log(self.freq_posterior + 1e-300) + log_likelihoods
        log_posterior -= logsumexp(log_posterior)  # Normalize
        self.freq_posterior = np.exp(log_posterior)

        # Update point estimates
        self.current_estimates["frequency"] = np.sum(self.freq_grid * self.freq_posterior)
        self.current_estimates["uncertainty"] = np.sqrt(
            np.sum(
                (self.freq_grid - self.current_estimates["frequency"]) ** 2 * self.freq_posterior
            )
        )

        # Store history for analysis
        self.measurement_history.append(measurement.copy())
        self.posterior_history.append(self.freq_posterior.copy())

    def expected_information_gain(self, test_frequency: float) -> float:
        """
        Compute expected information gain for a potential measurement.

        This is the key utility function for Sequential Bayesian Experiment Design,
        measuring how much information (entropy reduction) we expect to gain
        from a measurement at the proposed frequency.

        Args:
            test_frequency: Proposed measurement frequency

        Returns:
            Expected information gain (utility)
        """
        # Current entropy of posterior distribution
        current_entropy = -np.sum(self.freq_posterior * np.log(self.freq_posterior + 1e-300))

        # Monte Carlo estimation of expected entropy after measurement
        expected_entropy = 0.0
        n_samples = min(self.n_monte_carlo // 10, 100)  # Reduced for computational efficiency

        for _ in range(n_samples):
            # Sample true parameters from current posterior
            freq_idx = np.random.choice(self.grid_resolution, p=self.freq_posterior)
            true_freq = self.freq_grid[freq_idx]

            true_params = {
                "frequency": true_freq,
                "linewidth": self.current_estimates["linewidth"],
                "amplitude": self.current_estimates["amplitude"],
                "background": self.current_estimates["background"],
            }

            # Simulate measurement at test frequency
            expected_signal = self.odmr_model(test_frequency, true_params)

            # Add noise based on model
            if self.noise_model == "gaussian":
                noise_std = 0.05 * abs(expected_signal) + 0.01  # Relative + absolute noise
                simulated_intensity = np.random.normal(expected_signal, noise_std)
                uncertainty = noise_std
            else:
                # Poisson noise for photon counting
                rate = max(expected_signal, 0.1)
                simulated_intensity = np.random.poisson(rate)
                uncertainty = np.sqrt(rate)

            # Create simulated measurement
            sim_measurement = {
                "frequency": test_frequency,
                "intensity": simulated_intensity,
                "uncertainty": uncertainty,
            }

            # Compute posterior after this hypothetical measurement
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

            # Add entropy of this posterior to expected value
            entropy = -np.sum(temp_posterior * np.log(temp_posterior + 1e-300))
            expected_entropy += entropy

        expected_entropy /= n_samples

        # Information gain is reduction in entropy
        info_gain = current_entropy - expected_entropy
        return max(info_gain, 0)  # Ensure non-negative

    def mutual_information_criterion(self, test_frequency: float) -> float:
        """
        Alternative utility function based on mutual information.

        This provides a computationally lighter alternative to the full
        information gain calculation.

        Args:
            test_frequency: Proposed measurement frequency

        Returns:
            Expected mutual information
        """
        # Expected measurement sensitivity at this frequency
        # This is a simplified calculation - in practice would compute derivative
        sensitivity = 0.0
        for idx, freq in enumerate(self.freq_grid):
            df_dfreq = abs(test_frequency - freq) / self.current_estimates["linewidth"]
            sensitivity += self.freq_posterior[idx] * df_dfreq**2

        # Information gain approximation based on Fisher information
        noise_var = (0.05) ** 2  # Assumed measurement noise variance
        expected_var_reduction = sensitivity / (sensitivity + 1.0 / noise_var)

        return max(0.0, min(1.0, expected_var_reduction))

    def _ingest_history(self, history: Sequence[Obs]) -> None:
        """Integrate new observations into the posterior."""

        for obs in history:
            measurement = {
                "frequency": obs.x,
                "intensity": obs.intensity,
                "uncertainty": getattr(obs, "uncertainty", 0.05),
            }
            is_new = not any(
                existing["frequency"] == measurement["frequency"]
                for existing in self.measurement_history
            )
            if is_new:
                self.update_posterior(measurement)

    def _initial_candidate(self, domain: tuple[float, float]) -> float | None:
        """Return an exploratory candidate frequency for early measurements."""

        if len(self.measurement_history) >= 3:
            return None

        domain_low, domain_high = domain
        candidates = [domain_low, float(np.mean(domain)), domain_high]
        tested_freqs = {m["frequency"] for m in self.measurement_history}
        for candidate in candidates:
            if candidate not in tested_freqs:
                return candidate
        return None

    def _optimize_acquisition(self, domain: tuple[float, float]) -> tuple[float, float]:
        """Optimize the acquisition objective over the domain."""

        domain_low, domain_high = domain

        def negative_utility(freq: float) -> float:
            try:
                if self.acquisition_function == "expected_information_gain":
                    return -self.expected_information_gain(freq)
                if self.acquisition_function == "mutual_information":
                    return -self.mutual_information_criterion(freq)
                return -self.expected_information_gain(freq)
            except Exception as e:  # pragma: no cover - defensive
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
        except Exception as e:  # pragma: no cover - defensive
            warnings.warn(f"Optimization failed: {e}, using fallback strategy", stacklevel=2)
            optimal_freq = self.current_estimates["frequency"]
            utility = 0.0

        return optimal_freq, utility

    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float:
        """
        Propose next measurement frequency using Sequential Bayesian Design.

        This is the core method that implements the adaptive experimental design,
        choosing the frequency that maximizes expected information gain.

        Args:
            history: Previous measurements
            domain: Frequency domain bounds (Hz)

        Returns:
            Optimal next measurement frequency (Hz)
        """
        self._ingest_history(history)

        candidate = self._initial_candidate(domain)
        if candidate is not None:
            return candidate

        optimal_freq, utility = self._optimize_acquisition(domain)
        self.utility_history.append(utility)

        return optimal_freq

    def should_stop(self, history: Sequence[Obs]) -> bool:
        """
        Determine if measurement sequence should stop.

        Stopping criteria based on convergence and efficiency considerations.

        Args:
            history: Measurement history

        Returns:
            True if should stop, False otherwise
        """
        if len(history) >= self.max_evals:
            return True

        # Stop if uncertainty is below threshold
        if self.current_estimates["uncertainty"] < self.convergence_threshold:
            return True

        # Stop if recent measurements don't provide significant information gain
        if len(self.utility_history) >= 3:
            recent_utilities = self.utility_history[-3:]
            if all(u < self.min_uncertainty_reduction for u in recent_utilities):
                return True

        return False

    def finalize(self, history: Sequence[Obs]) -> dict[str, float]:
        """
        Finalize parameter estimates and detect number of peaks.

        Args:
            history: Complete measurement history

        Returns:
            Final parameter estimates including peak detection
        """
        # Ensure all measurements are processed
        for obs in history:
            measurement = {
                "frequency": obs.x,
                "intensity": obs.intensity,
                "uncertainty": getattr(obs, "uncertainty", 0.05),
            }
            if not any(
                m["frequency"] == measurement["frequency"] for m in self.measurement_history
            ):
                self.update_posterior(measurement)

        # Detect number of peaks by analyzing posterior distribution
        # Apply smoothing to reduce noise in peak detection
        posterior_smooth = np.convolve(self.freq_posterior, np.ones(5) / 5, mode="same")
        peaks = []

        # Find local maxima above threshold
        threshold = 0.1 * np.max(posterior_smooth)
        for i in range(1, len(posterior_smooth) - 1):
            if (
                posterior_smooth[i] > posterior_smooth[i - 1]
                and posterior_smooth[i] > posterior_smooth[i + 1]
                and posterior_smooth[i] > threshold
            ):
                peaks.append(i)

        n_peaks = len(peaks)

        if n_peaks == 0:
            # No clear peaks - use mode of posterior
            max_idx = np.argmax(self.freq_posterior)
            return {
                "n_peaks": 1.0,
                "x1": self.freq_grid[max_idx],
                "uncert": self.current_estimates["uncertainty"],
            }
        elif n_peaks == 1:
            # Single peak detected
            peak_idx = peaks[0]
            return {
                "n_peaks": 1.0,
                "x1": self.freq_grid[peak_idx],
                "uncert": self.current_estimates["uncertainty"],
            }
        else:
            # Multiple peaks - return two strongest
            peak_strengths = [(i, self.freq_posterior[i]) for i in peaks]
            peak_strengths.sort(key=lambda x: x[1], reverse=True)

            return {
                "n_peaks": 2.0,
                "x1": self.freq_grid[peak_strengths[0][0]],
                "x2": self.freq_grid[peak_strengths[1][0]]
                if len(peak_strengths) > 1
                else self.freq_grid[peak_strengths[0][0]],
                "uncert": self.current_estimates["uncertainty"],
            }
