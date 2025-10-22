"""Simulation framework for NV-center measurement experiments.

Provides extensible interfaces and simple reference implementations for:
- Data generation (ideal signals)
- Noise models (including compound application)
- Measurement strategies (estimators)
- Runner utilities to evaluate strategies under noise
- Iterative locators to find target positions in 1-D scans
"""
from .core import DataBatch, MeasurementStrategy, NoiseModel, DataGenerator, CompositeNoise
from .generators import RabiGenerator, T1Generator
from .noise import GaussianNoise, PoissonNoise, DriftNoise, OutlierSpikes
from .measurements import FluorescenceCount, RabiEstimate, T1Estimate
from .runner import ExperimentRunner
# Locators layer
from .locators import (
    GridScan,
    GoldenSectionSearch,
    TwoPeakGreedy,
    MeasurementProcess,
    ScanBatch,
    ScalarMeasure,
    LocatorStrategy,
)
from .scangen import OnePeakGenerator, TwoPeakGenerator
from .loc_runner import LocatorRunner

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
    "MeasurementProcess",
    "ScanBatch",
    "ScalarMeasure",
    "LocatorStrategy",
    "OnePeakGenerator",
    "TwoPeakGenerator",
    "LocatorRunner",
]
