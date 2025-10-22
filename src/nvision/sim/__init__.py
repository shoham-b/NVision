"""Simulation framework for NV-center measurement experiments.

Provides extensible interfaces and simple reference implementations for:
- Data generation (ideal signals)
- Noise models (including compound application)
- Measurement strategies (estimators)
- Runner utilities to evaluate strategies under noise
- Iterative locators to find target positions in 1-D scans
"""

from .core import CompositeNoise, DataBatch, DataGenerator, MeasurementStrategy, NoiseModel
from .generators import RabiGenerator, T1Generator
from .loc_runner import LocatorRunner

# Locators layer
from .locators import (
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
from .measurements import FluorescenceCount, RabiEstimate, T1Estimate
from .noise import DriftNoise, GaussianNoise, OutlierSpikes, PoissonNoise
from .runner import ExperimentRunner
from .scangen import OnePeakGenerator, TwoPeakGenerator

__all__ = [
    "BayesianLocator",
    "CompositeNoise",
    # core
    "DataBatch",
    "DataGenerator",
    "DriftNoise",
    # runner
    "ExperimentRunner",
    # measurements
    "FluorescenceCount",
    # noise
    "GaussianNoise",
    "GoldenSectionSearch",
    # locators
    "GridScan",
    "LocatorRunner",
    "LocatorStrategy",
    "MeasurementProcess",
    "MeasurementStrategy",
    "NoiseModel",
    "ODMRLocator",
    "OnePeakGenerator",
    "OutlierSpikes",
    "PoissonNoise",
    "RabiEstimate",
    # generators
    "RabiGenerator",
    "ScalarMeasure",
    "ScanBatch",
    "T1Estimate",
    "T1Generator",
    "TwoPeakGenerator",
    "TwoPeakGreedy",
]
