"""Run group registry — explicit preset combinations for the CLI.

Each :class:`RunGroup` holds concrete lists of generator, noise, and strategy
names.  The runner resolves them through :class:`CombinationGrid` rather than
relying on string filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from nvision.sim.combinations import CombinationGrid


@dataclass(frozen=True, slots=True)
class RunGroup:
    """Named preset that enumerates exactly which (generator, noise, strategy)
    triples to run."""

    name: str
    description: str
    generator_names: list[str]
    noise_names: list[str]
    strategy_names: list[str]


def _all_noise_names() -> list[str]:
    grid = CombinationGrid()
    return list(grid.noises.keys())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _group_lorentzian_sbed() -> RunGroup:
    noises = [n for n in _all_noise_names() if n.startswith("Gauss(")]
    return RunGroup(
        name="lorentzian-sbed",
        description="All Gauss noises for Bayesian-SBED on NVCenter-lorentzian.",
        generator_names=["NVCenter-lorentzian"],
        noise_names=noises,
        strategy_names=["Bayesian-SBED", "SimpleSobol", "SimpleSweep"],
    )


def _group_lorentzian_sbed_only() -> RunGroup:
    noises = [n for n in _all_noise_names() if n.startswith("Gauss(")]
    return RunGroup(
        name="lorentzian-sbed-only",
        description="All Gauss noises for Bayesian-SBED only (no sweep/sobol baselines) on NVCenter-lorentzian.",
        generator_names=["NVCenter-lorentzian"],
        noise_names=noises,
        strategy_names=["Bayesian-SBED"],
    )


@lru_cache(maxsize=1)
def _run_groups_tuple() -> tuple[RunGroup, ...]:
    return (_group_lorentzian_sbed(), _group_lorentzian_sbed_only())


def run_groups() -> list[RunGroup]:
    return list(_run_groups_tuple())


@lru_cache(maxsize=1)
def _run_group_by_normalized_name() -> dict[str, RunGroup]:
    return {g.name.lower().replace("-", "_"): g for g in _run_groups_tuple()}


def get_run_group(name: str) -> RunGroup:
    key = name.strip().lower().replace("-", "_")
    try:
        return _run_group_by_normalized_name()[key]
    except KeyError:
        raise KeyError(f"Unknown run group: {name!r}") from None


def clear_run_group_cache() -> None:
    """Drop lookup caches (e.g. if presets are monkeypatched in tests)."""
    _run_groups_tuple.cache_clear()
    _run_group_by_normalized_name.cache_clear()


def default_run_group() -> RunGroup:
    return _group_lorentzian_sbed()
