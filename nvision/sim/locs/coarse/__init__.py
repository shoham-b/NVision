"""Non-Bayesian / coarse search locators (sweep, Sobol, two-phase)."""

from nvision.sim.locs.coarse.sobol_locator import SobolLocator
from nvision.sim.locs.coarse.sweep_locator import SimpleSweepLocator

__all__ = [
    "SimpleSweepLocator",
    "SobolLocator",
]
