"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import numpy as np

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


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

        utilities = np.zeros(len(candidates))
        for j, x in enumerate(candidates):
            mu_pred = self.belief.model.compute_vectorized(float(x), sampled)

            # Simulate num_samples hypothetical measurement outcomes
            y_sim = mu_pred + np.random.normal(0, noise_std, num_samples)

            # Vectorized calculate of posterior weights for all outcomes across all particles
            # diff[i, k] = y_sim[i] - mu_pred[k]
            diff = y_sim[:, None] - mu_pred[None, :]
            log_lik = -0.5 * (diff / noise_std) ** 2
            log_lik -= np.max(log_lik, axis=1, keepdims=True)  # stability
            lik = np.exp(log_lik)

            weights = lik / np.sum(lik, axis=1, keepdims=True)

            # Calculate Shannon entropy H for each posterior: -sum(w * log(w))
            entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, None)), axis=1)

            # Maximize EIG = Minimize Expected Posterior Entropy
            utilities[j] = -np.mean(entropy)

        return float(candidates[np.argmax(utilities)])
