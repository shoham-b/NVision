"""Enums for combination-grid filters (no imports from :mod:`presets` or heavy sim graph — avoids cycles)."""

from __future__ import annotations

from enum import StrEnum


class GeneratorCategory(StrEnum):
    """High-level generator family used by :meth:`nvision.sim.combinations.CombinationGrid.generator_category`."""

    NVCENTER = "NVCenter"


class StrategyFilter(StrEnum):
    """Strategies available in the active simulation grid."""

    SIMPLE_SWEEP = "SimpleSweep"
    STAGED_SOBOL_SWEEP = "StagedSobolSweep"
    BAYESIAN_SBED = "Bayesian-SBED"
    SIMPLE_SOBOL = "SimpleSobol"


class GeneratorName(StrEnum):
    """Registered generator keys from :func:`nvision.sim.presets.generators_basic`."""

    NVCENTER_LORENTZIAN = "NVCenter-lorentzian"
    NVCENTER_VOIGT = "NVCenter-voigt"


class NoiseName(StrEnum):
    """Noise family names matched as prefixes against registered noise keys."""

    GAUSS = "Gauss"
    POISSON = "Poisson"
