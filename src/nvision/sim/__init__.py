"""Simulation framework for NV-center measurement experiments.

Provides extensible interfaces and simple reference implementations for:
- Data generation (ideal signals)
- Noise models (including compound application)
- Measurement strategies (estimators)
- Runner utilities to evaluate strategies under noise
- Iterative locators to find target positions in 1-D scans
"""

from .core import CompositeNoise, DataBatch, DataGenerator, MeasurementStrategy, NoiseModel
from .gen import (
    ExponentialDecayManufacturer,
    GaussianManufacturer,
    MultiPeakGenerator,
    OnePeakGenerator,
    TwoPeakGenerator,
)
from .loc_runner import LocatorRunner

# Locators layer
from .locs import (
    BayesianLocator,
    GoldenSectionSearchLocator,
    GridScanLocator,
    LocatorStrategy,
    MeasurementProcess,
    ODMRLocator,
    ScalarMeasure,
    ScanBatch,
    TwoPeakGreedyLocator,
)

# Noise implementations: export concrete full class names (no short aliases)
from .noises import (
    OverTimeDriftNoise,
    OverTimeRandomWalkNoise,
    OverVoltageGaussianNoise,
    OverVoltageOutlierSpikes,
    OverVoltagePoissonNoise,
)

__all__ = [
    "BayesianLocator",
    "CompositeNoise",
    "DataBatch",
    "DataGenerator",
    "ExponentialDecayManufacturer",
    "GaussianManufacturer",
    "GoldenSectionSearchLocator",
    "GridScanLocator",
    "LocatorRunner",
    "LocatorStrategy",
    "MeasurementProcess",
    "MeasurementStrategy",
    "MultiPeakGenerator",
    "NoiseModel",
    "ODMRLocator",
    "OnePeakGenerator",
    "OverTimeDriftNoise",
    "OverTimeRandomWalkNoise",
    "OverVoltageGaussianNoise",
    "OverVoltageOutlierSpikes",
    "OverVoltagePoissonNoise",
    "ScalarMeasure",
    "ScanBatch",
    "SymmetricTwoPeakGenerator",
    "TwoPeakGenerator",
    "TwoPeakGreedyLocator",
]
