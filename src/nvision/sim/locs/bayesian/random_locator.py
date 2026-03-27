"""Uniform-random Bayesian acquisition baseline."""

from __future__ import annotations

import numpy as np

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class RandomLocator(SequentialBayesianLocator):
    """Uniform-random acquisition baseline."""

    def _acquire(self) -> float:
        return float(np.random.uniform(*self._acquisition_bounds()))
