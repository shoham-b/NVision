"""Run group registry — explicit preset combinations for the CLI.

Each :class:`RunGroup` holds concrete lists of generator, noise, and strategy
names.  The runner resolves them through :class:`CombinationGrid` rather than
relying on string filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from nvision.sim import presets as sim_presets


@dataclass(frozen=True, slots=True)
class RunGroup:
    """Named preset that enumerates exactly which (generator, noise, strategy)
    triples to run."""

    name: str
    description: str
    generator_names: list[str]
    noise_names: list[str]
    strategy_names: list[str]
    # Generator objects for names not present in the default CombinationGrid
    # (e.g. the width x contrast study grid below), keyed by generator name.
    extra_generators: dict[str, object] | None = None


def _sbed_param_grid() -> dict[str, object]:
    """Saturation x sigma_inhom grid (saturation-Voigt lineshape) — the default
    multi-parameter generator set for the SBED run-groups below, so they run a
    parametric study out of the box. Note: despite the "lorentzian-*" group
    names (kept for CLI backward compatibility), these groups use the
    saturation-Voigt model, not plain Lorentzian."""
    return dict(sim_presets.saturation_voigt_param_grid_generators())


def _sbed_noise_names() -> list[str]:
    """Noise grid for the SBED run-groups — its own dedicated range/step count,
    swept the same way as width/contrast (see sim.presets.sbed_study_noises)."""
    return [name for name, _ in sim_presets.sbed_study_noises()]


def _voigt_inhom_param_grid() -> dict[str, object]:
    """Width x contrast x sigma_inhom grid (plain Voigt) for the voigt-inhom
    run-groups below. sigma_inhom=0 is the pure-Lorentzian limit (Voigt's
    reparam draws lorentz_frac=1.0 there), so this single grid's sigma_inhom
    axis already spans "plain Lorentzian" through "plain Voigt" -- there is no
    separate plain-Lorentzian or non-inhom plain-Voigt run-group."""
    return dict(sim_presets.voigt_sigma_inhom_param_grid_generators())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _group_lorentzian_sbed() -> RunGroup:
    extra_generators = _sbed_param_grid()
    return RunGroup(
        name="lorentzian-sbed",
        description=(
            "Saturation x sigma_inhom x noise grid (saturation-Voigt) for "
            "Bayesian-SBED/SimpleSobol/SimpleSweep."
        ),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["Bayesian-SBED", "SimpleSobol", "SimpleSweep"],
        extra_generators=extra_generators,
    )


def _group_lorentzian_sbed_only() -> RunGroup:
    extra_generators = _sbed_param_grid()
    return RunGroup(
        name="lorentzian-sbed-only",
        description=(
            "Saturation x sigma_inhom x noise grid (saturation-Voigt) for "
            "Bayesian-SBED only (no sweep/sobol baselines)."
        ),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["Bayesian-SBED"],
        extra_generators=extra_generators,
    )


def _group_lorentzian_sweep_only() -> RunGroup:
    extra_generators = _sbed_param_grid()
    return RunGroup(
        name="lorentzian-sweep-only",
        description=(
            "Saturation x sigma_inhom x noise grid (saturation-Voigt) for "
            "SimpleSweep only (no SBED/Sobol baselines)."
        ),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["SimpleSweep"],
        extra_generators=extra_generators,
    )


def _group_voigt_inhom_sbed() -> RunGroup:
    extra_generators = _voigt_inhom_param_grid()
    return RunGroup(
        name="voigt-inhom-sbed",
        description=(
            "Width x contrast x sigma_inhom x noise grid (plain Voigt; "
            "sigma_inhom=0 is the pure-Lorentzian limit) for "
            "Bayesian-SBED/SimpleSobol/SimpleSweep."
        ),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["Bayesian-SBED", "SimpleSobol", "SimpleSweep"],
        extra_generators=extra_generators,
    )


def _group_voigt_inhom_sbed_only() -> RunGroup:
    extra_generators = _voigt_inhom_param_grid()
    return RunGroup(
        name="voigt-inhom-sbed-only",
        description=(
            "Width x contrast x sigma_inhom x noise grid (plain Voigt; "
            "sigma_inhom=0 is the pure-Lorentzian limit) for "
            "Bayesian-SBED only (no sweep/sobol baselines)."
        ),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["Bayesian-SBED"],
        extra_generators=extra_generators,
    )


def _group_voigt_inhom_sweep_only() -> RunGroup:
    extra_generators = _voigt_inhom_param_grid()
    return RunGroup(
        name="voigt-inhom-sweep-only",
        description=(
            "Width x contrast x sigma_inhom x noise grid (plain Voigt; "
            "sigma_inhom=0 is the pure-Lorentzian limit) for "
            "SimpleSweep only (no SBED/Sobol baselines)."
        ),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["SimpleSweep"],
        extra_generators=extra_generators,
    )


@lru_cache(maxsize=1)
def _run_groups_tuple() -> tuple[RunGroup, ...]:
    return (
        _group_lorentzian_sbed(),
        _group_lorentzian_sbed_only(),
        _group_lorentzian_sweep_only(),
        _group_voigt_inhom_sbed(),
        _group_voigt_inhom_sbed_only(),
        _group_voigt_inhom_sweep_only(),
    )


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
