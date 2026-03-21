"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import numpy as np

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class EIGLocator(SequentialBayesianLocator):
    """Expected Information Gain acquisition.

    Samples around posterior mass quantiles where learning a measurement
    outcome would maximally reduce posterior entropy.
    """

    def _acquire(self) -> float:
        # If the belief supports analytical EIG, use it directly
        try:
            candidates = np.linspace(*self.belief.get_param(self._scan_param).bounds, 100)
            eigs = [self.belief.expected_information_gain(x) for x in candidates]
            return float(candidates[np.argmax(eigs)])
        except NotImplementedError:
            pass

        # Fall back to empirical SMC/Grid approximation
        candidates = np.linspace(*self.belief.get_param(self._scan_param).bounds, 100)
        cdf = self.belief.marginal_cdf(self._scan_param, candidates)
        # Handle small numerical issues with CDF
        cdf = np.clip(cdf, 0.0, 1.0)
        target_quantiles = np.linspace(0.1, 0.9, 9)
        # Ensure CDF is monotonically increasing for interp
        idx = np.argsort(cdf)
        chosen_candidates = np.interp(target_quantiles, cdf[idx], candidates[idx])
        return float(chosen_candidates[int(self.step_count % len(chosen_candidates))])
