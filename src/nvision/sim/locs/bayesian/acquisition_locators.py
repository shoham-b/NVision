"""Backward-compatible barrel for Bayesian acquisition locators.

Prefer importing from the dedicated modules, e.g.
``nvision.sim.locs.bayesian.eig_locator``.
"""

from __future__ import annotations

from nvision.sim.locs.bayesian.eig_locator import EIGLocator
from nvision.sim.locs.bayesian.max_variance_locator import MaxVarianceLocator
from nvision.sim.locs.bayesian.random_locator import RandomLocator
from nvision.sim.locs.bayesian.ucb_locator import UCBLocator
from nvision.sim.locs.bayesian.utility_sampling_locator import UtilitySamplingLocator

__all__ = [
    "EIGLocator",
    "MaxVarianceLocator",
    "RandomLocator",
    "UCBLocator",
    "UtilitySamplingLocator",
]
