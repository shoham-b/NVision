"""Uniform-random Bayesian acquisition baseline."""

from __future__ import annotations

import numpy as np

from nvision.sim.locs.bayesian.bayesian_locator import BayesianLocator


class RandomLocator(BayesianLocator):
    """Uniform-random acquisition baseline."""

    def _acquire(self) -> float:
        p = self.scan_posterior
        return float(np.random.uniform(*p.bounds))
