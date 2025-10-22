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
    # core
    "DataBatch",
    "MeasurementStrategy",
    "NoiseModel",
    "DataGenerator",
    "CompositeNoise",
    # generators
    "RabiGenerator",
    "T1Generator",
    # noise
    "GaussianNoise",
    "PoissonNoise",
    "DriftNoise",
    "OutlierSpikes",
    # measurements
    "FluorescenceCount",
    "RabiEstimate",
    "T1Estimate",
    # runner
    "ExperimentRunner",
    # locators
    "GridScan",
    "GoldenSectionSearch",
    "TwoPeakGreedy",
    "ODMRLocator",
    "BayesianLocator",
    "MeasurementProcess",
    "ScanBatch",
    "ScalarMeasure",
    "LocatorStrategy",
    "OnePeakGenerator",
    "TwoPeakGenerator",
    "LocatorRunner",
]
