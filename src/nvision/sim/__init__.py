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
    CompositeOverProbeNoise,
    DataBatch,
    DataGenerator,
    MeasurementStrategy,
    OverFrequencyNoise,
    OverProbeNoise,
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
    SimpleSequentialLocator,
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
    OverProbeDriftNoise,
    OverProbeRandomWalkNoise,
)

__all__ = [
    "CompositeNoise",
    "CompositeOverFrequencyNoise",
    "CompositeOverProbeNoise",
    "DataBatch",
    "DataGenerator",
    "ExponentialDecayManufacturer",
    "GaussianManufacturer",
    "Locator",
    "LocatorRunner",
    "MeasurementStrategy",
    "MultiPeakGenerator",
    "NVCenterGenerator",
    "NVCenterSequentialBayesianLocator",
    "NVCenterSweepLocator",
    "OnePeakGenerator",
    "OnePeakGoldenLocator",
    "OnePeakGridLocator",
    "OnePeakSweepLocator",
    "OverFrequencyGaussianNoise",
    "OverFrequencyNoise",
    "OverFrequencyOutlierSpikes",
    "OverFrequencyPoissonNoise",
    "OverProbeDriftNoise",
    "OverProbeNoise",
    "OverProbeRandomWalkNoise",
    "ScanBatch",
    "SimpleSequentialLocator",
    "TwoPeakGenerator",
    "TwoPeakGoldenLocator",
    "TwoPeakGridLocator",
    "TwoPeakSweepLocator",
    "run_locator",
]
