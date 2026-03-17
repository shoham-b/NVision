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
from .runner_v2 import LocatorRunnerV2

# Locators layer (v2 only)
from .locs import (
    Locator,
    NVCenterSweepLocatorV2,
    Observation,
    Runner,
    ScanBatch,
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
    "LocatorRunnerV2",
    "MeasurementStrategy",
    "MultiPeakGenerator",
    "NVCenterGenerator",
    "NVCenterSweepLocatorV2",
    "Observation",
    "OnePeakGenerator",
    "OverFrequencyGaussianNoise",
    "OverFrequencyNoise",
    "OverFrequencyOutlierSpikes",
    "OverFrequencyPoissonNoise",
    "OverProbeDriftNoise",
    "OverProbeNoise",
    "OverProbeRandomWalkNoise",
    "Runner",
    "ScanBatch",
    "TwoPeakGenerator",
]
