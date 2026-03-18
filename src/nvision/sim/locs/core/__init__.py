"""Locators using new core architecture."""

from nvision.sim.locs.core.sweep_locator import SimpleSweepLocator
from nvision.sim.locs.core.nv_bayesian_locator import NVCenterBayesianLocator

__all__ = [
    "SimpleSweepLocator",
    "NVCenterBayesianLocator",
]
