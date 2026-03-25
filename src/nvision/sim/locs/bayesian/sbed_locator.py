"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from numba import njit

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


@njit(cache=True)
def _sbed_eig_utilities_from_mu_and_noise(
    mu_preds: np.ndarray,
    noise_chunk: np.ndarray,
    inv_noise_std: float,
    eps: float,
) -> np.ndarray:
    """
    Compute the SBED utility for a chunk of candidates.

    Notes
    -----
    This mirrors the original NumPy tensor math but avoids materializing the
    full (m, n_outcomes, n_particles) intermediate via explicit loops.
    """
    m, n_particles = mu_preds.shape
    _, n_outcomes = noise_chunk.shape

    log_eps = math.log(eps)
    utilities = np.empty(m, dtype=np.float64)

    for i in range(m):
        acc_entropy = 0.0

        # For each hypothetical outcome o, form the discrete posterior over
        # particles and compute its Shannon entropy.
        for o in range(n_outcomes):
            y = mu_preds[i, o] + noise_chunk[i, o]

            # Stable normalization: subtract max log-likelihood over particles.
            max_log_lik = -1e300
            for p in range(n_particles):
                diff = y - mu_preds[i, p]
                log_lik = -0.5 * (diff * inv_noise_std) * (diff * inv_noise_std)
                if log_lik > max_log_lik:
                    max_log_lik = log_lik

            sum_exp = 0.0
            for p in range(n_particles):
                diff = y - mu_preds[i, p]
                log_lik = -0.5 * (diff * inv_noise_std) * (diff * inv_noise_std)
                sum_exp += math.exp(log_lik - max_log_lik)

            inv_sum_exp = 1.0 / (sum_exp + 1e-300)
            entropy = 0.0

            for p in range(n_particles):
                diff = y - mu_preds[i, p]
                log_lik = -0.5 * (diff * inv_noise_std) * (diff * inv_noise_std)
                e = math.exp(log_lik - max_log_lik)
                w = e * inv_sum_exp

                # Match np.log(np.clip(weights, eps, None)) semantics.
                if w < eps:
                    entropy += -w * log_eps
                else:
                    entropy += -w * math.log(w)

            acc_entropy += entropy

        # Utility is negative expected entropy (so argmax picks minimum entropy).
        utilities[i] = -(acc_entropy / n_outcomes)

    return utilities


class SequentialBayesianExperimentDesignLocator(SequentialBayesianLocator):
    """Sequential Bayesian Experiment Design acquisition.

    Evaluates exact utility using Monte Carlo simulation of posterior Shannon entropy,
    as defined in the physical NV ODMR experiment design paper.
    """

    def _acquire(self) -> float:
        # Sequential Bayesian Experiment Design Utility calculation:
        # Evaluate the mathematically exact Expected Information Gain (Shannon Entropy Reduction)
        # by simulating hypothetical measurements and estimating the expected posterior entropy.
        candidates = np.linspace(*self.belief.get_param(self._scan_param).bounds, 200)
        num_samples = 100

        sampled = self.belief.sample(num_samples)
        param_names = self.belief.model.parameter_names()

        # Calculate expected noise level based on belief using normalized parameters
        # to ensure signal scaling correctly compares with [0,1] range parameters
        uncertainties = self.belief.uncertainty()
        norm_unc = uncertainties.values_ordered[0]
        if hasattr(self.belief, "physical_param_bounds"):
            p_name = param_names[0]
            lo, hi = self.belief.physical_param_bounds[p_name]
            norm_unc = norm_unc / (hi - lo) if hi > lo else 0.0

        noise_std = max(0.01, norm_unc * 0.1) if len(uncertainties) > 0 else 0.01

        # Pre-sample hypothetical measurement noise once so results remain
        # deterministic for a given RNG state.
        measurement_noise = np.random.normal(0.0, noise_std, size=(len(candidates), num_samples))

        model = self.belief.model

        # Compute utilities in candidate chunks. The heavy O(num_samples^2)
        # tensor math (likelihood normalization + entropy) is kept in NumPy for
        # speed; Polars is used only for the final argmax selection.
        chunk_size = 64 if len(candidates) > 64 else len(candidates)
        utilities = np.zeros(len(candidates), dtype=float)
        eps = 1e-12

        for start in range(0, len(candidates), chunk_size):
            end = min(len(candidates), start + chunk_size)
            xs = candidates[start:end]

            # mu_preds shape: (m, num_samples)
            mu_preds = model.compute_vectorized_many(xs, sampled)

            noise_chunk = measurement_noise[start:end]  # (m, num_samples)
            utilities[start:end] = _sbed_eig_utilities_from_mu_and_noise(
                mu_preds=mu_preds,
                noise_chunk=noise_chunk,
                inv_noise_std=1.0 / noise_std,
                eps=eps,
            )

        best_idx = int(pl.Series(utilities).arg_max())
        return float(candidates[best_idx])
