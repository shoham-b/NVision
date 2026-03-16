"""
Sequential Bayesian Experiment Design Locator for ODMR of NV Centers

This module implements a Sequential Bayesian Experiment Design (SBED) strategy
for Optically Detected Magnetic Resonance (ODMR) measurements.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from nvision.sim.locs.nv_center._jit_kernels import (
    _expected_info_gain_jit,
)
from nvision.sim.locs.nv_center.bayesian_base import NVCenterBayesianLocatorBase

log = logging.getLogger(__name__)


@dataclass
class NVCenterSequentialBayesianLocator(NVCenterBayesianLocatorBase):
    """
    An advanced locator implementing Sequential Bayesian Experiment Design (SBED).

    This specific implementation uses Expectation of Information Gain (EIG)
    to select the next measurement.
    """

    def expected_information_gain(self, test_frequency: float) -> float:
        n_samples = min(self.n_monte_carlo // 10, 100)

        # Use JIT kernel if distribution is lorentzian
        if self.distribution == "lorentzian":
            current_estimates_array = np.array(
                [
                    self.current_estimates["linewidth"],
                    self.current_estimates["amplitude"],
                    self.current_estimates["background"],
                ],
                dtype=np.float64,
            )

            noise_model_code = 0 if self.noise_model == "gaussian" else 1

            return _expected_info_gain_jit(
                test_frequency,
                self.freq_grid,
                self.freq_posterior,
                n_samples,
                current_estimates_array,
                noise_model_code,
            )

        # Fallback for other distributions
        current_entropy = -np.sum(self.freq_posterior * np.log(self.freq_posterior + 1e-300))

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
            "k_np": self.current_estimates["k_np"],
        }
        expected_signals = self.odmr_model(test_frequency, true_params)

        if self.noise_model == "gaussian":
            noise_stds = 0.05 * np.abs(expected_signals) + 0.01
            simulated_intensities = np.random.normal(expected_signals, noise_stds)
        else:  # poisson
            rates = np.maximum(expected_signals, 0.1)
            simulated_intensities = np.random.poisson(rates)

        # `calculate_log_likelihoods_grid` expects scalar `signal_values`, but here we
        # have a Monte Carlo ensemble. Use the mean intensity to construct a single
        # representative measurement for the EIG calculation.
        sim_measurement = {
            "x": float(test_frequency),
            "signal_values": float(np.mean(simulated_intensities)),
        }

        base_params = {k: v for k, v in self.current_estimates.items() if k != "frequency"}
        log_likelihoods = self._calculate_log_likelihoods(sim_measurement, base_params)

        # Posterior update and entropy calculation.
        # For this fallback path, `_calculate_log_likelihoods` returns a 1D array
        # over the frequency grid. We treat this as a single Monte Carlo sample.
        log_post = np.log(self.freq_posterior + 1e-300) + log_likelihoods
        log_post -= logsumexp(log_post)
        post = np.exp(log_post)

        entropy = -np.sum(post * np.log(post + 1e-300))
        expected_entropy = float(entropy)

        info_gain = current_entropy - expected_entropy
        return max(info_gain, 0.0)

    def _optimize_acquisition(self, domain: tuple[float, float]) -> tuple[float, float]:
        domain_low, domain_high = domain

        def negative_utility(freq: float) -> float:
            return -self.expected_information_gain(freq)

        result = minimize_scalar(
            negative_utility,
            bounds=(domain_low, domain_high),
            method="bounded",
            options={"maxiter": 100},
        )
        optimal_freq = result.x
        utility = -result.fun

        return optimal_freq, utility
