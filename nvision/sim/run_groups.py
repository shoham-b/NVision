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
    return ["GenericSweep", "SobolSweep", "StagedSobolSweep"]


def _bayesian_strategy_names() -> list[str]:
    return ["Bayesian-SBED", "Bayesian-UtilitySampling"]


def _bayesian_nosweep_strategy_names() -> list[str]:
    return ["Bayesian-SBED-NoSweep",  "Bayesian-UtilitySampling-NoSweep"]


def _narrow_strategy_names() -> list[str]:
    return [
        "Bayesian-SBED-NoSweep",
        "Bayesian-MaximumLikelihood-NoSweep",
        "Bayesian-UtilitySampling-NoSweep",
        "StudentsTApproximation",
    ]


def _nv_generators() -> list[str]:
    return ["NVCenter-lorentzian", "NVCenter-voigt"]


def _nv_narrow_generators() -> list[str]:
    return ["NVCenter-lorentzian-narrow", "NVCenter-voigt-narrow"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _group_all() -> RunGroup:
    gens = _all_generator_names()
    noises = _all_noise_names()
    strats = [
        s
        for s in _all_strategy_names_for(gens)
        if "MaximumLikelihood" not in s and "UtilitySampling" not in s
    ]
    return RunGroup(
        name="all",
        description="All generators, noises, and strategies.",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


def _group_sweep_only() -> RunGroup:
    gens = _nv_generators()
    noises = _all_noise_names()
    strats = _sweep_strategy_names()
    return RunGroup(
        name="sweep_only",
        description="Sweep locators only (GenericSweep, SobolSweep, StagedSobolSweep).",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


def _group_sweep_then_bayesian() -> RunGroup:
    gens = _nv_generators()
    noises = _all_noise_names()
    strats = _sweep_strategy_names() + _bayesian_strategy_names()
    return RunGroup(
        name="sweep_then_bayesian",
        description="Sweep locators followed by Bayesian acquisition (includes initial sweep).",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


def _group_demo() -> RunGroup:
    gens = _nv_generators()
    noises = _all_noise_names()
    strats = _sweep_strategy_names() + _bayesian_strategy_names()
    return RunGroup(
        name="demo",
        description="Quick demo: sweep + Bayesian on standard NV generators with all noises.",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


def _group_bayesian_only() -> RunGroup:
    gens = _nv_narrow_generators()
    noises = _all_noise_names()
    strats = _bayesian_nosweep_strategy_names()
    return RunGroup(
        name="bayesian_only",
        description="Bayesian locators without initial sweep on narrow-domain generators.",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


def _group_bayesian_clean() -> RunGroup:
    gens = _nv_narrow_generators()
    noises = ["NoNoise", "Gauss(0.01)", "Poisson(3000.0)"]
    strats = _bayesian_nosweep_strategy_names()
    return RunGroup(
        name="bayesian_clean",
        description="Bayesian without initial sweep on narrow-domain generators, limited to basic noises.",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


def _group_narrow_only() -> RunGroup:
    gens = _nv_narrow_generators()
    noises = _all_noise_names()
    strats = _narrow_strategy_names()
    return RunGroup(
        name="narrow_only",
        description="All narrow-domain (no-sweep) locators on narrow generators.",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


def _group_smc_only() -> RunGroup:
    gens = _nv_generators()
    noises = _all_noise_names()
    strats = _bayesian_strategy_names()
    return RunGroup(
        name="smc_only",
        description="SMC-based Bayesian locators (SBED, UtilitySampling).",
        generator_names=gens,
        noise_names=noises,
        strategy_names=strats,
    )


@lru_cache(maxsize=1)
def _run_groups_tuple() -> tuple[RunGroup, ...]:
    return (
        _group_all(),
        _group_sweep_only(),
        _group_sweep_then_bayesian(),
        _group_demo(),
        _group_bayesian_only(),
        _group_bayesian_clean(),
        _group_narrow_only(),
        _group_smc_only(),
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
