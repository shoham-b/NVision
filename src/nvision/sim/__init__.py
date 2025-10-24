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
    GaussianManufacturer,
    MultiPeakGenerator,
    OnePeakGenerator,
    RabiManufacturer,
    SymmetricTwoPeakGenerator,
    T1DecayManufacturer,
    TwoPeakGenerator,
)
from .loc_runner import LocatorRunner

# Locators layer
from .locs import (
    BayesianLocator,
    GoldenSectionSearch,
    GridScan,
    LocatorStrategy,
    MeasurementProcess,
    ODMRLocator,
    ScalarMeasure,
    ScanBatch,
    TwoPeakGreedy,
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
    "GaussianManufacturer",
    "GoldenSectionSearch",
    "GridScan",
    "LocatorRunner",
    "LocatorStrategy",
    "MeasurementProcess",
    "MeasurementStrategy",
    "MultiPeakGenerator",
    "NoiseModel",
    "ODMRLocator",
    "OnePeakGenerator",
    "OverVoltageOutlierSpikes",
    "OverVoltagePoissonNoise",
    "RabiManufacturer",
    "ScalarMeasure",
    "ScanBatch",
    "SymmetricTwoPeakGenerator",
    "T1DecayManufacturer",
    "TwoPeakGenerator",
    "TwoPeakGreedy",
    # Export full noise class names
    "OverTimeDriftNoise",
    "OverVoltageGaussianNoise",
    "OverTimeRandomWalkNoise",
]
