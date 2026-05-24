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


def _all_generator_names() -> list[str]:
    grid = CombinationGrid()
    return list(grid.generators.keys())


def _all_noise_names() -> list[str]:
    grid = CombinationGrid()
    return list(grid.noises.keys())


def _all_strategy_names_for(generators: list[str]) -> list[str]:
    grid = CombinationGrid()
    names: set[str] = set()
    for g in generators:
        for s_name, _ in grid.strategies_for(g):
            names.add(s_name)
    return sorted(names)


def _sweep_strategy_names() -> list[str]:
    return ["GenericSweep", "StagedSobolSweep"]


def _bayesian_strategy_names() -> list[str]:
    return ["Bayesian-SBED", "Bayesian-SBED-NoSweep"]


def _ekf_strategy_names() -> list[str]:
    return ["Bayesian-EKF-D", "Bayesian-EKF-A", "Bayesian-EKF-ParticleFrequency"]


def _gmm_strategy_names() -> list[str]:
    return ["GaussianMixture"]


def _wide_generators() -> list[str]:
    return ["NVCenter-lorentzian", "NVCenter-voigt"]


def _default_noise_names() -> list[str]:
    return _all_noise_names()


def _default_strategy_names(gens: list[str]) -> list[str]:
    return _all_strategy_names_for(gens)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _group_all() -> RunGroup:
    gens = _all_generator_names()
    noises = _default_noise_names()
    strats = _default_strategy_names(gens)
    return RunGroup(
        name="all",
        description="All generators (Lorentzian, Voigt), noises (Gauss + Poisson), and strategies (Sweep + Staged + SBED).",  # noqa: E501
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


def _group_wide() -> RunGroup:
    gens = _wide_generators()
    noises = _default_noise_names()
    strats = _default_strategy_names(gens)
    return RunGroup(
        name="wide",
        description="Lorentzian and Voigt generators with all allowed noises and strategies.",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


def _group_sweep_only() -> RunGroup:
    gens = _all_generator_names()
    noises = _default_noise_names()
    strats = _sweep_strategy_names()
    return RunGroup(
        name="sweep_only",
        description="Sweep locators only (GenericSweep, StagedSobolSweep).",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


def _group_sbed_only() -> RunGroup:
    gens = _all_generator_names()
    noises = _default_noise_names()
    strats = _bayesian_strategy_names()
    return RunGroup(
        name="sbed_only",
        description="SBED locators only (with and without initial sweep).",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


def _group_ekf_only() -> RunGroup:
    gens = _all_generator_names()
    noises = _default_noise_names()
    strats = _ekf_strategy_names()
    return RunGroup(
        name="ekf_only",
        description="EKF locators only (D-optimal, A-optimal, ParticleFrequency).",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


def _group_gmm_only() -> RunGroup:
    gens = _all_generator_names()
    noises = _default_noise_names()
    strats = _gmm_strategy_names()
    return RunGroup(
        name="gmm_only",
        description="Gaussian Mixture locator only.",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


@lru_cache(maxsize=1)
def _run_groups_tuple() -> tuple[RunGroup, ...]:
    return (
        _group_all(),
        _group_wide(),
        _group_sweep_only(),
        _group_sbed_only(),
        _group_ekf_only(),
        _group_gmm_only(),
    )


def run_groups() -> list[RunGroup]:
    return list(_run_groups_tuple())


@lru_cache(maxsize=1)
def _run_group_by_normalized_name() -> dict[str, RunGroup]:
    return {g.name.lower(): g for g in _run_groups_tuple()}


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
    return _group_all()
