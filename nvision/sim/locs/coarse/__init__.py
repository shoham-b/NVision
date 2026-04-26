"""Non-Bayesian / coarse search locators (sweep, Sobol, two-phase)."""

from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator
from nvision.sim.locs.coarse.sobol_locator import StagedSobolSweepLocator
from nvision.sim.locs.coarse.sweep_locator import SweepingLocator

# SimpleSweepLocator is an alias for GenericSweepLocator (simple uniform sweep)
SimpleSweepLocator = GenericSweepLocator

__all__ = [
    "GenericSweepLocator",
    "SimpleSweepLocator",
    "StagedSobolSweepLocator",
    "SweepingLocator",
]
