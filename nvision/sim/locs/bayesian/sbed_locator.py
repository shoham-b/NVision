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
    inv_noise_std: np.ndarray,
    eps: float,
) -> np.ndarray:
    """
    Compute the SBED utility for a chunk of candidates.

    ``inv_noise_std`` is a 2-D array (m, n_outcomes) so that heteroscedastic
    noise (e.g. Poisson) can use the correct per-outcome precision.

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
            inv_sigma = inv_noise_std[i, o]

            # Stable normalization: subtract max log-likelihood over particles.
            max_log_lik = -1e300
            for p in range(n_particles):
                diff = y - mu_preds[i, p]
                log_lik = -0.5 * (diff * inv_sigma) * (diff * inv_sigma)
                if log_lik > max_log_lik:
                    max_log_lik = log_lik

            sum_exp = 0.0
            for p in range(n_particles):
                diff = y - mu_preds[i, p]
                log_lik = -0.5 * (diff * inv_sigma) * (diff * inv_sigma)
                sum_exp += math.exp(log_lik - max_log_lik)

            inv_sum_exp = 1.0 / (sum_exp + 1e-300)
            entropy = 0.0

            for p in range(n_particles):
                diff = y - mu_preds[i, p]
                log_lik = -0.5 * (diff * inv_sigma) * (diff * inv_sigma)
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

    def _generate_posterior_candidates(self, n: int = 200) -> np.ndarray:
        """Sample candidate measurement points from the posterior scan-parameter marginal.

        For SMC beliefs the particles already encode where the signal is likely
        located.  Sampling candidates from the particle distribution (weighted by
        posterior weight) concentrates EIG computation on regions with actual
        posterior support, avoiding waste on flat signal shoulders.

        Returns an empty array when the belief does not expose particle internals.
        """
        belief = self.belief
        if not hasattr(belief, "_particles") or not hasattr(belief, "_weights"):
            return np.array([])

        # Extract scan-parameter column from particles
        param_names = getattr(belief, "_param_names", None)
        if param_names is None or self._scan_param not in param_names:
            return np.array([])

        idx = param_names.index(self._scan_param)
        particle_vals = belief._particles[:, idx]
        weights = belief._weights

        # Weighted sampling with replacement (concentrates where posterior mass is)
        probs = weights / (weights.sum() + 1e-300)
        chosen = np.random.choice(len(particle_vals), size=n, p=probs, replace=True)
        candidates = particle_vals[chosen].copy()

        # Clip to acquisition bounds and sort
        lo, hi = self._acquisition_bounds()
        candidates = np.clip(candidates, lo, hi)
        candidates.sort()

        # Add small jitter to avoid exact duplicates (which waste EIG evaluations)
        jitter = np.random.normal(0.0, (hi - lo) * 0.005, size=n)
        candidates = np.clip(candidates + jitter, lo, hi)
        candidates.sort()

        # Deduplicate so we don't evaluate EIG at the exact same x twice
        # (tolerance ~ 0.1% of window width)
        tol = max((hi - lo) * 1e-3, 1e-12)
        mask = np.concatenate(([True], np.diff(candidates) > tol))
        candidates = candidates[mask]
        return candidates

    def _acquire(self) -> float:
        # Sequential Bayesian Experiment Design Utility calculation:
        # Evaluate the mathematically exact Expected Information Gain (Shannon Entropy Reduction)
        # by simulating hypothetical measurements and estimating the expected posterior entropy.
        num_candidates = 200
        num_samples = 100

        # Generate candidates from the posterior scan-parameter marginal instead of
        # a uniform grid.  With SMC the particles already encode where the signal is;
        # measuring near particle frequency hypotheses concentrates EIG evaluations
        # on regions with actual posterior support rather than wasting them on flat
        # shoulders of the dip where the signal is uninformative.
        candidates = self._generate_posterior_candidates(num_candidates)
        if len(candidates) == 0:
            # Fallback: uniform grid when belief has no particles
            candidates = self._generate_candidates(num_candidates)

        # Use information-gain-based particle selection instead of random sampling
        candidates_arr = np.asarray(candidates)
        sampled = self.belief.select_max_information_gain(candidates_arr, num_samples)

        # Use the known measurement noise from the last observation.
        # For Gaussian frequency noise we draw Gaussian hypothetical outcomes;
        # for pure Poisson frequency noise we draw Poisson counts and rescale.
        last_obs = self.belief.last_obs
        noise_std = last_obs.noise_std if last_obs is not None else 0.05
        freq_model = getattr(last_obs, "frequency_noise_model", None) if last_obs is not None else None

        # Detect a single-component Poisson over-frequency model.
        use_poisson = bool(
            freq_model
            and len(freq_model) == 1
            and freq_model[0].get("type") == "poisson"
            and float(freq_model[0].get("scale", 0.0)) > 0.0
        )
        poisson_scale = float(freq_model[0]["scale"]) if use_poisson else 1.0

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
            m = mu_preds.shape[0]

            if use_poisson:
                # Poisson: std_y depends on the mean signal at each candidate/outcome.
                # lam = mu * scale,  var(k) = lam,  std_y = sqrt(lam)/scale.
                lam = np.maximum(mu_preds, 1e-12) * poisson_scale
                std_y = np.sqrt(lam) / poisson_scale  # (m, num_samples)
                noise_chunk = np.random.normal(0.0, 1.0, size=(m, num_samples)) * std_y
                inv_noise_std_arr = 1.0 / std_y  # (m, num_samples)
            else:
                # Gaussian (or unknown): homogeneous noise.
                noise_chunk = np.random.normal(0.0, noise_std, size=(m, num_samples))
                inv_noise_std_arr = np.full((m, num_samples), 1.0 / noise_std)

            utilities[start:end] = _sbed_eig_utilities_from_mu_and_noise(
                mu_preds=mu_preds,
                noise_chunk=noise_chunk,
                inv_noise_std=inv_noise_std_arr,
                eps=eps,
            )

        utilities = self._apply_parameter_weight_bias(utilities, all_mu_preds, sampled, candidates)
        best_idx = int(pl.Series(utilities).arg_max())
        return float(candidates[best_idx])
