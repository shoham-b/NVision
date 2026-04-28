"""Enums for combination-grid filters (no imports from :mod:`presets` or heavy sim graph — avoids cycles)."""

from __future__ import annotations

from enum import StrEnum


class GeneratorCategory(StrEnum):
    """High-level generator family used by :meth:`nvision.sim.combinations.CombinationGrid.generator_category`."""

    NVCENTER = "NVCenter"


class StrategyFilter(StrEnum):
    """Substring matched against strategy names in :meth:`nvision.sim.combinations.CombinationGrid.iter`."""

    SIMPLE_SWEEP = "SimpleSweep"
    SWEEP = "Sweep,StagedSobolSweep"  # Matches GenericSweep and StagedSobolSweep
    BAYESIAN = "Bayesian"
    BAYESIAN_SBED = "Bayesian-SBED"
    BAYESIAN_UCB = "Bayesian-UCB"
    BAYESIAN_MAX_VARIANCE = "Bayesian-MaxVariance"
    BAYESIAN_MAXIMUM_LIKELIHOOD = "Bayesian-MaximumLikelihood"
    BAYESIAN_UTILITY_SAMPLING = "Bayesian-UtilitySampling"


class GeneratorName(StrEnum):
    """Registered generator keys from :func:`nvision.sim.presets.generators_basic`."""

    NVCENTER_LORENTZIAN = "NVCenter-lorentzian"
    NVCENTER_VOIGT = "NVCenter-voigt"


class NoiseName(StrEnum):
    """Noise family names matched as prefixes against registered noise keys."""

    NO_NOISE = "NoNoise"
    GAUSS = "Gauss"
    POISSON = "Poisson"
    OVER_PROBE_DRIFT = "OverProbeDrift"
    HEAVY = "Heavy"
