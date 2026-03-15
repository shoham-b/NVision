from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from nvision.sim.locs.nv_center._jit_classes import Bayesian2DState
from .sequential_bayesian_locator import (
    NVCenterSequentialBayesianLocator,
)

log = logging.getLogger(__name__)


@dataclass
class ProjectBayesianLocator(NVCenterSequentialBayesianLocator):
    """
    Bayesian Locator based on the 'Project Book Submission'.

    Uses a variance-based utility function and probabilistic frequency selection.
    """

    pickiness: float = 5.0
    linewidth_resolution: int = 200  # Smaller resolution for 2D grid to keep it fast

    linewidth_resolution: int = 200

    # Internal state
    state: Bayesian2DState | None = None

    @property
    def posterior_2d(self):
        return self.state.posterior_2d if self.state else None

    def reset_posterior(self):
        """Reset posterior distributions to priors (2D JIT State)."""
        self.freq_grid = np.linspace(self.prior_bounds[0], self.prior_bounds[1], self.grid_resolution).astype(
            np.float32
        )

        self.linewidth_grid = np.linspace(
            self.linewidth_prior[0], self.linewidth_prior[1], self.linewidth_resolution
        ).astype(np.float32)

        # Initialize JIT State
        self.state = Bayesian2DState(self.freq_grid, self.linewidth_grid)

        # Create dummy posteriors for visualizer compatibility using public property
        # self.freq_posterior is used by base class plotting
        est_freq, est_gamma, u_f, _, ent, max_p = self.state.get_estimates()

        # We need to manually reconstruct freq_posterior for plotting if needed
        # Or expose it from JIT class. JIT class has posterior_2d.
        # Let's perform initial marginalization for consisteny
        self.freq_posterior = np.sum(self.state.posterior_2d, axis=1)

        self.current_estimates = {
            "frequency": float(est_freq),
            "linewidth": float(est_gamma),
            "amplitude": np.mean(self.amplitude_prior),
            "background": np.mean(self.background_prior),
            "uncertainty": float(u_f),
            "gaussian_width": np.mean(self.gaussian_width_prior),
            "split": np.mean(self.split_prior),
            "k_np": np.mean(self.k_np_prior),
            "entropy": float(ent),
            "max_prob": float(max_p),
        }
        self._init_bayes_optimizer()

    def update_posterior(self, measurement: dict[str, float]):
        measurement = self._coerce_measurement(measurement)

        mx = float(measurement["x"])
        my = float(measurement["signal_values"])
        uncert = float(measurement.get("uncertainty", 0.05))

        amp = float(self.current_estimates["amplitude"])
        bg = float(self.current_estimates["background"])
        noise_model_code = 0 if self.noise_model == "gaussian" else 1

        # Delegate to JIT State
        self.state.update(mx, my, uncert, amp, bg, noise_model_code)

        # Sync Estimates
        est_freq, est_gamma, u_f, _, ent, max_p = self.state.get_estimates()

        self.current_estimates["frequency"] = float(est_freq)
        self.current_estimates["linewidth"] = float(est_gamma)
        self.current_estimates["uncertainty"] = float(u_f)
        self.current_estimates["entropy"] = float(ent)
        self.current_estimates["max_prob"] = float(max_p)

        # update marginal for base class
        # (Usually JIT class arrays are wrapped numpy arrays, so slicing works)
        # But Summing inside JIT class returns new array.
        # We can re-sum here or add get_marginal_freq to JIT class.
        # Let's assume posterior_2d is accessible as numpy array (it is for jitclass)
        self.freq_posterior = np.sum(self.state.posterior_2d, axis=1)

        self.measurement_history.append(measurement.copy())
        self.posterior_history.append(self.freq_posterior.copy())

        # Optimize other params (amplitude, background)?
        # For now, keep using the base method (which does scipy minimize)
        # BUT base method might overwrite frequency/linewidth if we are not careful?
        # Base method `_optimize_lineshape_params` optimizes ["linewidth", "amplitude", "background"].
        # We should PROBABLY DISABLE linewidth optimization in base if we are estimating it via Bayes.
        # However, for now, let's let base optimize amplitude/background.
        # The base `_optimize_lineshape_params` implementation sets `self.current_estimates[key]`.
        # If "linewidth" is in params, it will overwrite our Bayesian estimate.
        # We should prevent that.

        # Override _get_optim_config in this class?
        # Or just manually optimize here.
        # Let's rely on base for now but maybe override _get_optim_config to exclude linewidth.
        self._optimize_lineshape_params()

        self.parameter_history.append(self.current_estimates.copy())

    def _get_optim_config(self):
        # Override to exclude linewidth from optimization since we estimate it via Bayes
        if self.distribution == "lorentzian":
            param_keys = ["amplitude", "background"]
            bounds = [self.amplitude_prior, self.background_prior]
        else:
            # Fallback to base for others
            return super()._get_optim_config()
        return param_keys, bounds

    def _calculate_utility_grid(self) -> np.ndarray:
        n_samples = min(self.n_monte_carlo, 100)

        amp = float(self.current_estimates["amplitude"])
        bg = float(self.current_estimates["background"])

        # Sigma is not really stored in current_estimates for Gaussian?
        # Use hardcoded or infer
        sigma = 0.05

        return self.state.calculate_utility(n_samples, amp, bg, sigma)

    def _optimize_acquisition(self, domain: tuple[float, float]) -> tuple[float, float]:
        """
        Select next frequency probabilistically based on utility.
        """
        utility_grid = self._calculate_utility_grid()

        # Avoid numerical issues with utility
        utility_grid = np.maximum(utility_grid, 1e-9)

        # Calculate selection probability: P(f) ~ Utility(f)^Pickiness
        # Use log space for stability
        log_prob = self.pickiness * np.log(utility_grid)
        log_prob -= logsumexp(log_prob)
        # Add small epsilon to prevent prob sum being 0 or nan
        prob = np.exp(log_prob)
        prob /= np.sum(prob)  # Renormalize to be safe

        # Sample from the grid
        idx = np.random.choice(self.grid_resolution, p=prob)
        selected_freq = self.freq_grid[idx]

        return selected_freq, utility_grid[idx]
