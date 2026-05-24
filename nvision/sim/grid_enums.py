"""Enums for combination-grid filters (no imports from :mod:`presets` or heavy sim graph — avoids cycles)."""

from __future__ import annotations

from enum import StrEnum


class GeneratorCategory(StrEnum):
    """High-level generator family used by :meth:`nvision.sim.combinations.CombinationGrid.generator_category`."""

    NVCENTER = "NVCenter"


class StrategyFilter(StrEnum):
    """Strategies available in the active simulation grid."""

    GENERIC_SWEEP = "GenericSweep"
    STAGED_SOBOL_SWEEP = "StagedSobolSweep"
    BAYESIAN_SBED = "Bayesian-SBED"
    SIMPLE_SOBOL = "SimpleSobol"
    GAUSSIAN_MIXTURE = "GaussianMixture"
    BAYESIAN_EKF_D = "Bayesian-EKF-D"
    BAYESIAN_EKF_A = "Bayesian-EKF-A"
    BAYESIAN_EKF_PARTICLE_FREQUENCY = "Bayesian-EKF-ParticleFrequency"


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
