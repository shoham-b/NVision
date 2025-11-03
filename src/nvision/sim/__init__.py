"""Simulation framework for NV-center measurement experiments.

Provides extensible interfaces and simple reference implementations for:
- Data generation (ideal signals)
- Noise models (including compound application)
- Measurement strategies (estimators)
- Runner utilities to evaluate strategies under noise
- Iterative locators to find target positions in 1-D scans
"""

from .core import (
    CompositeNoise,
    CompositeOverFrequencyNoise,
    CompositeOverTimeNoise,
    DataBatch,
    DataGenerator,
    MeasurementStrategy,
    OverFrequencyNoise,
    OverTimeNoise,
)
from .gen import (
    ExponentialDecayManufacturer,
    GaussianManufacturer,
    MultiPeakGenerator,
    NVCenterGenerator,
    OnePeakGenerator,
    TwoPeakGenerator,
)
from .loc_runner import LocatorRunner

# Locators layer
from .locs import (
    Locator,
    NVCenterSequentialBayesianLocator,
    NVCenterSweepLocator,
    OnePeakGoldenLocator,
    OnePeakGridLocator,
    OnePeakSweepLocator,
    ScanBatch,
    TwoPeakGoldenLocator,
    TwoPeakGridLocator,
    TwoPeakSweepLocator,
    run_locator,
)

# Noise implementations: export concrete full class names (no short aliases)
from .noises import (
    OverFrequencyGaussianNoise,
    OverFrequencyOutlierSpikes,
    OverFrequencyPoissonNoise,
    OverTimeDriftNoise,
    OverTimeRandomWalkNoise,
)

__all__ = [
    "CompositeNoise",
    "CompositeOverFrequencyNoise",
    "CompositeOverTimeNoise",
    "DataBatch",
    "DataGenerator",
    "ExponentialDecayManufacturer",
    "GaussianManufacturer",
    "Locator",
    "LocatorRunner",
    "MeasurementStrategy",
    "MultiPeakGenerator",
    "OnePeakGenerator",
    "TwoPeakGenerator",
    "NVCenterGenerator",
    "OverFrequencyGaussianNoise",
    "OverFrequencyNoise",
    "OverFrequencyOutlierSpikes",
    "OverFrequencyPoissonNoise",
    "OverTimeDriftNoise",
    "OverTimeNoise",
    "OverTimeRandomWalkNoise",
    "ScanBatch",
    "run_locator",
    # Category-specific locators
    "OnePeakGridLocator",
    "OnePeakGoldenLocator",
    "OnePeakSweepLocator",
    "TwoPeakGridLocator",
    "TwoPeakGoldenLocator",
    "TwoPeakSweepLocator",
    "NVCenterSweepLocator",
    "NVCenterSequentialBayesianLocator",
]
