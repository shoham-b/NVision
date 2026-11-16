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
    ``SimpleSweep-1.5perFWHM`` (the form ``refine_rates_runs`` parses). The study no longer runs fixed-``k`` sweeps;
    the name stays valid for sweeps built directly.
    """
    return f"{StrategyFilter.SIMPLE_SWEEP}-{float(points_per_fwhm):g}perFWHM"


# The plain sweep whose grid step is the analytically optimal k* of its task (nvision.sim.locs.sweep_optimal). One
# fixed name: k* varies per grid cell, so it is not part of the name (it is recorded in each repeat's metrics).
OPTIMAL_SWEEP_STRATEGY = f"{StrategyFilter.SIMPLE_SWEEP}-optimal"

# The sweeps of the study are the optimal one and its multiples: a sweep whose grid step is ``multiplier * k*`` points
# per FWHM. Running ``multiplier`` != 1 tests that k* really is the cheapest step (cost should rise on both sides).
OPTIMAL_SWEEP_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0)


def optimal_sweep_strategy_name(multiplier: float) -> str:
    """Strategy name of the sweep at ``multiplier * k*``: ``SimpleSweep-optimal`` for 1, else ``...-optimal-x0.5``."""
    if not (multiplier > 0.0):
        raise ValueError(f"multiplier must be positive, got {multiplier!r}")
    if float(multiplier) == 1.0:
        return OPTIMAL_SWEEP_STRATEGY
    return f"{OPTIMAL_SWEEP_STRATEGY}-x{float(multiplier):g}"


def optimal_sweep_multiplier(strategy_name: str) -> float | None:
    """The k* multiplier encoded in ``strategy_name`` (1 for ``SimpleSweep-optimal``), ``None`` for any other name."""
    if strategy_name == OPTIMAL_SWEEP_STRATEGY:
        return 1.0
    prefix = f"{OPTIMAL_SWEEP_STRATEGY}-x"
    if strategy_name.startswith(prefix):
        try:
            return float(strategy_name[len(prefix) :])
        except ValueError:
            return None
    return None


def is_plain_sweep(strategy_name: str) -> bool:
    """True for the plain (non-adaptive) sweeps: ``SimpleSweep-<k>perFWHM`` and ``SimpleSweep-optimal[-x<m>]``."""
    return strategy_name == StrategyFilter.SIMPLE_SWEEP or strategy_name.startswith(f"{StrategyFilter.SIMPLE_SWEEP}-")
