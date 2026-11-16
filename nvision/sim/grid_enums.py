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


def sweep_strategy_name(points_per_fwhm: float) -> str:
    """Strategy name of the plain sweep with ``points_per_fwhm`` grid points per FWHM (fractional allowed).

    Whole numbers print without a decimal point, so ``3`` and ``3.0`` are ``SimpleSweep-3perFWHM`` and ``1.5`` is
    ``SimpleSweep-1.5perFWHM`` (the form ``refine_rates_runs`` parses).
    """
    return f"{StrategyFilter.SIMPLE_SWEEP}-{float(points_per_fwhm):g}perFWHM"


# The plain sweep whose grid step is the analytically optimal k* of its task (nvision.sim.locs.sweep_optimal). One
# fixed name: k* varies per grid cell, so it is not part of the name (it is recorded in each repeat's metrics).
OPTIMAL_SWEEP_STRATEGY = f"{StrategyFilter.SIMPLE_SWEEP}-optimal"


def is_plain_sweep(strategy_name: str) -> bool:
    """True for the plain (non-adaptive) sweep strategies: ``SimpleSweep-<k>perFWHM`` and ``SimpleSweep-optimal``."""
    return strategy_name == StrategyFilter.SIMPLE_SWEEP or strategy_name.startswith(f"{StrategyFilter.SIMPLE_SWEEP}-")
