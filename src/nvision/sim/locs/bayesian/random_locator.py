"""Uniform-random Bayesian acquisition baseline."""

from __future__ import annotations

import numpy as np

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class RandomLocator(SequentialBayesianLocator):
    """Uniform-random acquisition baseline."""

    def _acquire(self) -> float:
        lo, hi = self._acquisition_bounds()
        is_scale = getattr(self.belief.model, "is_scale_parameter", lambda name: False)(self._scan_param)
        if is_scale and lo > 0 and hi > lo:
            return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
        return float(np.random.uniform(lo, hi))
