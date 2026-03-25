"""Upper Confidence Bound (UCB) Bayesian acquisition locator."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


@njit(cache=True)
def _argmax_ucb(prob: np.ndarray) -> int:
    best_idx = 0
    best_score = -1e300
    for i in range(prob.shape[0]):
        p = prob[i]
        var_term = p * (1.0 - p)
        if var_term < 0.0:
            var_term = 0.0
        bonus = math.sqrt(var_term)
        score = p + 2.0 * bonus
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


class UCBLocator(SequentialBayesianLocator):
    """Upper Confidence Bound acquisition.

    Balances exploitation (high posterior mass) with exploration
    (high posterior uncertainty).
    """

    def _acquire(self) -> float:
        candidates = np.linspace(*self.belief.get_param(self._scan_param).bounds, 100)
        pdf = self.belief.marginal_pdf(self._scan_param, candidates)
        # Normalize PDF to act like discrete probabilities for the heuristic
        prob = pdf / (np.sum(pdf) + 1e-12)
        best_idx = int(_argmax_ucb(prob))
        return float(candidates[best_idx])
