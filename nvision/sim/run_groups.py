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


_ALL_STRATEGIES = ["Bayesian-SBED", "SimpleSobol", "SimpleSweep"]

_BOTH_GRID = "Width x contrast x sigma_inhom x noise grid (plain Voigt; sigma_inhom=0 is the pure-Lorentzian limit) for"
_DRIFT_GRID = (
    "Width x contrast grid (plain Lorentzian) x drift scenarios (center and Zeeman splitting "
    "moving during the run, plus a no-drift control) for"
)


def _sbed_noise_names() -> list[str]:
    """Noise grid for the SBED run-groups — its own dedicated range/step count,
    swept the same way as width/contrast (see sim.presets.sbed_study_noises)."""
    return [name for name, _ in sim_presets.sbed_study_noises()]


def _drift_noise_names() -> list[str]:
    """No-drift control plus every drift scenario (see sim.presets.drift_study_noises)."""
    return [name for name, _ in sim_presets.drift_study_noises()]


def _voigt_inhom_param_grid() -> dict[str, object]:
    """Width x contrast x sigma_inhom grid (plain Voigt, inhomogeneous broadening as an
    explicit axis).

    sigma_inhom=0 makes NVCenterVoigtModel's reparam draw lorentz_frac=1.0 -- a
    pure-Lorentzian-shaped pseudo-Voigt profile, so the Lorentzian limit is one endpoint
    of this grid's own sigma_inhom axis instead of a second model class."""
    return dict(sim_presets.voigt_sigma_inhom_param_grid_generators())


def _lorentzian_plain_param_grid() -> dict[str, object]:
    """Width x contrast grid (plain Lorentzian lineshape), used by the drift study."""
    return dict(sim_presets.param_grid_generators(variant="lorentzian"))


def _group(
    name: str,
    description: str,
    extra_generators: dict[str, object],
    noise_names: list[str],
    strategy_names: list[str],
) -> RunGroup:
    return RunGroup(
        name=name,
        description=description,
        generator_names=list(extra_generators.keys()),
        noise_names=noise_names,
        strategy_names=strategy_names,
        extra_generators=extra_generators,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _group_both_sbed() -> RunGroup:
    return _group(
        "both-sbed",
        f"{_BOTH_GRID} Bayesian-SBED/SimpleSobol/SimpleSweep.",
        _voigt_inhom_param_grid(),
        _sbed_noise_names(),
        _ALL_STRATEGIES,
    )


def _group_both_sbed_only() -> RunGroup:
    return _group(
        "both-sbed-only",
        f"{_BOTH_GRID} Bayesian-SBED only (no sweep/sobol baselines).",
        _voigt_inhom_param_grid(),
        _sbed_noise_names(),
        ["Bayesian-SBED"],
    )


def _group_both_sweep_only() -> RunGroup:
    return _group(
        "both-sweep-only",
        f"{_BOTH_GRID} SimpleSweep only (no SBED/Sobol baselines).",
        _voigt_inhom_param_grid(),
        _sbed_noise_names(),
        ["SimpleSweep"],
    )


def _group_lorentzian_plain_drift() -> RunGroup:
    return _group(
        "lorentzian-plain-drift",
        f"{_DRIFT_GRID} Bayesian-SBED/SimpleSobol/SimpleSweep.",
        _lorentzian_plain_param_grid(),
        _drift_noise_names(),
        _ALL_STRATEGIES,
    )


def _group_lorentzian_plain_drift_sbed_only() -> RunGroup:
    return _group(
        "lorentzian-plain-drift-sbed-only",
        f"{_DRIFT_GRID} Bayesian-SBED only (no sweep/sobol baselines).",
        _lorentzian_plain_param_grid(),
        _drift_noise_names(),
        ["Bayesian-SBED"],
    )


@lru_cache(maxsize=1)
def _run_groups_tuple() -> tuple[RunGroup, ...]:
    return (
        _group_both_sbed(),
        _group_both_sbed_only(),
        _group_both_sweep_only(),
        _group_lorentzian_plain_drift(),
        _group_lorentzian_plain_drift_sbed_only(),
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
    return _group_both_sbed()
