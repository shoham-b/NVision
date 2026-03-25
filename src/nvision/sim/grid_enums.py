"""Enums for combination-grid filters (no imports from :mod:`cases` or heavy sim graph — avoids cycles)."""

from __future__ import annotations

from enum import StrEnum


class GeneratorCategory(StrEnum):
    """High-level generator family used by :meth:`nvision.sim.combinations.CombinationGrid.generator_category`."""

    NVCENTER = "NVCenter"
    ONE_PEAK = "OnePeak"
    TWO_PEAK = "TwoPeak"


class StrategyFilter(StrEnum):
    """Substring matched against strategy names in :meth:`nvision.sim.combinations.CombinationGrid.iter`."""

    SIMPLE_SWEEP = "SimpleSweep"
    BAYESIAN = "Bayesian"
    BAYESIAN_SBED = "Bayesian-SBED"
    BAYESIAN_UCB = "Bayesian-UCB"
    BAYESIAN_MAX_VARIANCE = "Bayesian-MaxVariance"
    BAYESIAN_MAXIMUM_LIKELIHOOD = "Bayesian-MaximumLikelihood"
    BAYESIAN_UTILITY_SAMPLING = "Bayesian-UtilitySampling"


class GeneratorName(StrEnum):
    """Registered generator keys from :func:`nvision.sim.cases.generators_basic`."""

    ONE_PEAK_GAUSSIAN = "OnePeak-gaussian"
    ONE_PEAK_LORENTZIAN = "OnePeak-lorentzian"
    TWO_PEAK_GAUSSIAN = "TwoPeak-gaussian"
    TWO_PEAK_LORENTZIAN = "TwoPeak-lorentzian"
    NVCENTER_ONE_PEAK = "NVCenter-one_peak"
    NVCENTER_ZEEMAN = "NVCenter-zeeman"
    NVCENTER_VOIGT_ONE_PEAK = "NVCenter-voigt_one_peak"
    NVCENTER_VOIGT_ZEEMAN = "NVCenter-voigt_zeeman"
