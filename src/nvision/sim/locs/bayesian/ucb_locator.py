"""Upper Confidence Bound (UCB) Bayesian acquisition locator."""

from __future__ import annotations

import numpy as np

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


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
        bonus = np.sqrt(np.maximum(prob * (1.0 - prob), 0.0))
        best_idx = int(np.argmax(prob + 2.0 * bonus))
        return float(candidates[best_idx])
