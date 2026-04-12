"""Backward-compatible barrel for Bayesian acquisition locators.

Prefer importing from the dedicated modules, e.g.
``nvision.sim.locs.bayesian.sbed_locator``.
"""

from __future__ import annotations

from nvision.sim.locs.bayesian.maximum_likelihood_locator import MaximumLikelihoodLocator
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.sim.locs.bayesian.utility_sampling_locator import UtilitySamplingLocator

__all__ = [
    "MaximumLikelihoodLocator",
    "SequentialBayesianExperimentDesignLocator",
    "UtilitySamplingLocator",
]
