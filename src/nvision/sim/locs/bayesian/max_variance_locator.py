"""Maximum-variance Bayesian acquisition locator."""

from __future__ import annotations

import numpy as np
from numba import njit

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


@njit(cache=True)
def _argmax_bernoulli_variance(prob: np.ndarray) -> int:
    best_idx = 0
    best_score = -1.0
    for i in range(prob.shape[0]):
        p = prob[i]
        score = p * (1.0 - p)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


class MaxVarianceLocator(SequentialBayesianLocator):
    """Maximum posterior-variance acquisition.

    Measures where Bernoulli-style variance ``p(1-p)`` is largest.
    """

    def _acquire(self) -> float:
        candidates = np.linspace(*self.belief.get_param(self._scan_param).bounds, 100)
        pdf = self.belief.marginal_pdf(self._scan_param, candidates)
        prob = pdf / (np.sum(pdf) + 1e-12)
        best_idx = int(_argmax_bernoulli_variance(prob))
        return float(candidates[best_idx])
