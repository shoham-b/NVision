"""Enums for combination-grid filters (no imports from :mod:`presets` or heavy sim graph — avoids cycles)."""

from __future__ import annotations

from enum import StrEnum


class StrategyFilter(StrEnum):
    """Strategies available in the active simulation grid."""

    SIMPLE_SWEEP = "SimpleSweep"
    BAYESIAN_SBED = "Bayesian-SBED"
    BAYESIAN_SBED_DRIVE = "Bayesian-SBED-Drive"


class GeneratorName(StrEnum):
    """Registered generator keys from :func:`nvision.sim.presets.generators_basic`."""

    NVCENTER_LORENTZIAN = "NVCenter-lorentzian"
    NVCENTER_VOIGT = "NVCenter-voigt"


class NoiseName(StrEnum):
    """Noise family names matched as prefixes against registered noise keys."""

    GAUSS = "Gauss"


def is_plain_sweep(strategy_name: str) -> bool:
    """True for the plain (non-adaptive) sweep strategies: ``SimpleSweep`` and its ``SimpleSweep-d<k>`` spacings."""
    return strategy_name == StrategyFilter.SIMPLE_SWEEP or strategy_name.startswith(f"{StrategyFilter.SIMPLE_SWEEP}-d")
