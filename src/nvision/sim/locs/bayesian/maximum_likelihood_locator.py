"""Maximum Likelihood Bayesian acquisition locator."""

from __future__ import annotations

import numpy as np

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class MaximumLikelihoodLocator(SequentialBayesianLocator):
    """Maximum Likelihood (Mode) acquisition.

    Measures where the marginal posterior distribution is maximized.
    """

    def _acquire(self) -> float:
        candidates = np.linspace(*self.belief.get_param(self._scan_param).bounds, 200)
        pdf = self.belief.marginal_pdf(self._scan_param, candidates)
        best_idx = int(np.argmax(pdf))
        return float(candidates[best_idx])
