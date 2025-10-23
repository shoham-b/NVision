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
from .noises import DriftNoise, GaussianNoise, OutlierSpikes, PoissonNoise

__all__ = [
    "BayesianLocator",
    "CompositeNoise",
    "DataBatch",
    "DataGenerator",
    "DriftNoise",
    "GaussianManufacturer",
    "GaussianNoise",
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
    "OutlierSpikes",
    "PoissonNoise",
    "RabiManufacturer",
    "ScalarMeasure",
    "ScanBatch",
    "SymmetricTwoPeakGenerator",
    "T1DecayManufacturer",
    "TwoPeakGenerator",
    "TwoPeakGreedy",
]
