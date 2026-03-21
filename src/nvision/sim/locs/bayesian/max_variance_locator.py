"""Maximum-variance Bayesian acquisition locator."""

from __future__ import annotations

import numpy as np

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class MaxVarianceLocator(SequentialBayesianLocator):
    """Maximum posterior-variance acquisition.

    Measures where Bernoulli-style variance ``p(1-p)`` is largest.
    """

    def _acquire(self) -> float:
        candidates = np.linspace(*self.belief.get_param(self._scan_param).bounds, 100)
        pdf = self.belief.marginal_pdf(self._scan_param, candidates)
        prob = pdf / (np.sum(pdf) + 1e-12)
        best_idx = int(np.argmax(prob * (1.0 - prob)))
        return float(candidates[best_idx])
