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
        candidates = self._generate_candidates(200)
        num_samples = 100

        sampled = self.belief.sample(num_samples)

        # Use the known measurement noise from the last observation.
        # For Gaussian frequency noise we draw Gaussian hypothetical outcomes;
        # for pure Poisson frequency noise we draw Poisson counts and rescale.
        last_obs = self.belief.last_obs
        noise_std = last_obs.noise_std if last_obs is not None else 0.05
        freq_model = getattr(last_obs, "frequency_noise_model", None) if last_obs is not None else None

        n_candidates = len(candidates)
        measurement_noise = np.zeros((n_candidates, num_samples), dtype=float)

        # Detect a single-component Poisson over-frequency model and match its generative process.
        use_poisson = bool(
            freq_model
            and len(freq_model) == 1
            and freq_model[0].get("type") == "poisson"
            and float(freq_model[0].get("scale", 0.0)) > 0.0
        )
        if use_poisson:
            scale = float(freq_model[0]["scale"])
            # We will add these residuals to mu_preds later: y = mu + (k/scale - mu).
            for i in range(n_candidates):
                # Temporarily approximate mu for noise draws with the current posterior mean at this candidate.
                # The exact mu_preds are computed below; this keeps draws consistent in scale.
                # Draw counts k ~ Poisson(mu * scale) and convert to residuals in signal space.
                # We use a simple Normal approximation for speed when scale*mu is large.
                # For SBED, slight noise-shape approximation is acceptable.
                lam = 1.0  # placeholder; refined per-candidate when applying noise_chunk
                # Store standard normal draws; they will be scaled when applying lam.
                measurement_noise[i, :] = np.random.normal(0.0, 1.0, size=num_samples)
        else:
            # Gaussian (or unknown) case: simple Normal residuals with known std.
            measurement_noise = np.random.normal(0.0, noise_std, size=(n_candidates, num_samples))

        model = self.belief.model

        # Compute utilities in candidate chunks. The heavy O(num_samples^2)
        # tensor math (likelihood normalization + entropy) is kept in NumPy for
        # speed; Polars is used only for the final argmax selection.
        chunk_size = 64 if len(candidates) > 64 else len(candidates)
        utilities = np.zeros(len(candidates), dtype=float)
        all_mu_preds = np.empty((len(candidates), num_samples), dtype=float)
        eps = 1e-12

        for start in range(0, len(candidates), chunk_size):
            end = min(len(candidates), start + chunk_size)
            xs = candidates[start:end]

            # mu_preds shape: (m, num_samples)
            mu_preds = model.compute_vectorized_many(xs, sampled)
            all_mu_preds[start:end] = mu_preds

            noise_chunk = measurement_noise[start:end]  # (m, num_samples)
            inv_noise_std = 1.0 / noise_std
            if use_poisson:
                scale = float(freq_model[0]["scale"])
                # For Poisson, interpret pre-drawn standard normals as approximate
                # count fluctuations around mean and convert to residuals in signal space.
                # lam ≈ mu * scale; var(k) ≈ lam, so std in y-space ≈ sqrt(lam)/scale.
                # We therefore scale the standard-normal draws by that std.
                for i in range(mu_preds.shape[0]):
                    lam = np.maximum(mu_preds[i, :], 1e-12) * scale
                    std_y = np.sqrt(lam) / scale
                    noise_chunk[i, :] = measurement_noise[start + i, :] * std_y
                    # Effective Gaussian approximation in y-space; keep inv_noise_std consistent with std_y scale.
                # For SBED entropy math, we treat these as Gaussian residuals with local std_y,
                # but reuse a global inv_noise_std for numerical stability.
                inv_noise_std = 1.0 / max(noise_std, 1e-6)
            utilities[start:end] = _sbed_eig_utilities_from_mu_and_noise(
                mu_preds=mu_preds,
                noise_chunk=noise_chunk,
                inv_noise_std=inv_noise_std,
                eps=eps,
            )

        utilities = self._apply_parameter_weight_bias(utilities, all_mu_preds, sampled)
        best_idx = int(pl.Series(utilities).arg_max())
        return float(candidates[best_idx])
