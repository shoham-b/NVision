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


def sweep_strategy_name(points_per_fwhm: int) -> str:
    """Strategy name of the plain sweep with ``points_per_fwhm`` grid points per FWHM, e.g. ``SimpleSweep-3perFWHM``."""
    return f"{StrategyFilter.SIMPLE_SWEEP}-{points_per_fwhm}perFWHM"


# A fixed-budget variant of a strategy runs with the stop rule off: every repeat measures the whole shared budget,
# so the error is known at every measurement count (the refine-rate fits of docs/evaluation_metrics.md need that).
# Its results are cached under their own name, ``<strategy>-fixed``; it is only reachable by that name (never in the
# default grid).
FIXED_BUDGET_SUFFIX = "-fixed"


def fixed_budget_name(strategy_name: str) -> str:
    """Name of the fixed-budget (stop rule off) variant of ``strategy_name``, e.g. ``Bayesian-SBED-fixed``."""
    return f"{strategy_name}{FIXED_BUDGET_SUFFIX}"


def is_fixed_budget(strategy_name: str) -> bool:
    """True for a ``<strategy>-fixed`` name."""
    return strategy_name.endswith(FIXED_BUDGET_SUFFIX)


def base_strategy_name(strategy_name: str) -> str:
    """``strategy_name`` without the fixed-budget suffix (unchanged when it has none)."""
    return strategy_name.removesuffix(FIXED_BUDGET_SUFFIX)


def is_plain_sweep(strategy_name: str) -> bool:
    """True for the plain (non-adaptive) sweep strategies: ``SimpleSweep-<k>perFWHM``."""
    return strategy_name == StrategyFilter.SIMPLE_SWEEP or strategy_name.startswith(f"{StrategyFilter.SIMPLE_SWEEP}-")
