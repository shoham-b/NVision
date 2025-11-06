from nvision.sim.locs.nv_center._bayesian_adapter import NVCenterSequentialBayesianLocatorBatched
from nvision.sim.locs.nv_center.sweep_locator import NVCenterSweepLocator

# Alias the batched version as the main class name for backward compatibility
NVCenterSequentialBayesianLocator = NVCenterSequentialBayesianLocatorBatched

__all__ = [
    "NVCenterSequentialBayesianLocator",
    "NVCenterSequentialBayesianLocatorBatched",
    "NVCenterSweepLocator",
]
