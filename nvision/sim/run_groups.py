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


def _voigt_param_grid() -> dict[str, object]:
    """Width x contrast grid (plain Voigt lineshape) for the voigt run-groups below."""
    return dict(sim_presets.param_grid_generators(variant="voigt"))


def _lorentzian_plain_param_grid() -> dict[str, object]:
    """Width x contrast grid (plain Lorentzian lineshape) for the lorentzian-plain
    run-groups below. Distinct from ``_sbed_param_grid()``, which despite its use
    in the "lorentzian-sbed" group names actually generates saturation-Voigt data."""
    return dict(sim_presets.param_grid_generators(variant="lorentzian"))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _group_lorentzian_sbed() -> RunGroup:
    extra_generators = _sbed_param_grid()
    return RunGroup(
        name="lorentzian-sbed",
        description=(
            "Saturation x sigma_inhom x noise grid (saturation-Voigt) for Bayesian-SBED/SimpleSobol/SimpleSweep."
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
            "Saturation x sigma_inhom x noise grid (saturation-Voigt) for SimpleSweep only (no SBED/Sobol baselines)."
        ),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["SimpleSweep"],
        extra_generators=extra_generators,
    )


def _group_voigt_sbed() -> RunGroup:
    extra_generators = _voigt_param_grid()
    return RunGroup(
        name="voigt-sbed",
        description=("Width x contrast x noise grid (plain Voigt) for Bayesian-SBED/SimpleSobol/SimpleSweep."),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["Bayesian-SBED", "SimpleSobol", "SimpleSweep"],
        extra_generators=extra_generators,
    )


def _group_voigt_sbed_only() -> RunGroup:
    extra_generators = _voigt_param_grid()
    return RunGroup(
        name="voigt-sbed-only",
        description=("Width x contrast x noise grid (plain Voigt) for Bayesian-SBED only (no sweep/sobol baselines)."),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["Bayesian-SBED"],
        extra_generators=extra_generators,
    )


def _group_voigt_sweep_only() -> RunGroup:
    extra_generators = _voigt_param_grid()
    return RunGroup(
        name="voigt-sweep-only",
        description=("Width x contrast x noise grid (plain Voigt) for SimpleSweep only (no SBED/Sobol baselines)."),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["SimpleSweep"],
        extra_generators=extra_generators,
    )


def _voigt_inhom_param_grid() -> dict[str, object]:
    """Width x contrast x sigma_inhom grid (plain Voigt, inhomogeneous broadening
    as an explicit axis) for the voigt-inhom run-groups below."""
    return dict(sim_presets.voigt_sigma_inhom_param_grid_generators())


def _group_voigt_inhom_sbed() -> RunGroup:
    extra_generators = _voigt_inhom_param_grid()
    return RunGroup(
        name="voigt-inhom-sbed",
        description=(
            "Width x contrast x sigma_inhom x noise grid (plain Voigt, "
            "inhomogeneous/Gaussian broadening selectable) for "
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
            "Width x contrast x sigma_inhom x noise grid (plain Voigt, "
            "inhomogeneous/Gaussian broadening selectable) for "
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
            "Width x contrast x sigma_inhom x noise grid (plain Voigt, "
            "inhomogeneous/Gaussian broadening selectable) for "
            "SimpleSweep only (no SBED/Sobol baselines)."
        ),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["SimpleSweep"],
        extra_generators=extra_generators,
    )


def _group_lorentzian_plain_sbed() -> RunGroup:
    extra_generators = _lorentzian_plain_param_grid()
    return RunGroup(
        name="lorentzian-plain-sbed",
        description=("Width x contrast x noise grid (plain Lorentzian) for Bayesian-SBED/SimpleSobol/SimpleSweep."),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["Bayesian-SBED", "SimpleSobol", "SimpleSweep"],
        extra_generators=extra_generators,
    )


def _group_lorentzian_plain_sbed_only() -> RunGroup:
    extra_generators = _lorentzian_plain_param_grid()
    return RunGroup(
        name="lorentzian-plain-sbed-only",
        description=(
            "Width x contrast x noise grid (plain Lorentzian) for Bayesian-SBED only (no sweep/sobol baselines)."
        ),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["Bayesian-SBED"],
        extra_generators=extra_generators,
    )


def _group_lorentzian_plain_sweep_only() -> RunGroup:
    extra_generators = _lorentzian_plain_param_grid()
    return RunGroup(
        name="lorentzian-plain-sweep-only",
        description=(
            "Width x contrast x noise grid (plain Lorentzian) for SimpleSweep only (no SBED/Sobol baselines)."
        ),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["SimpleSweep"],
        extra_generators=extra_generators,
    )


def _group_both_sbed() -> RunGroup:
    # sigma_inhom=0 makes NVCenterVoigtModel's reparam draw lorentz_frac=1.0 -- a
    # pure-Lorentzian-shaped pseudo-Voigt profile (verified: the Thompson-Cox-Hastings
    # eta polynomial evaluates to exactly 1.0 there too). Since NVCenterVoigtModel now
    # uses the same homogeneous_linewidth/c_total conventions as NVCenterLorentzianModel
    # (not the old fwhm_total/lorentz_frac/dip_depth parametrization), this is genuinely
    # the same physical lineshape a separate Lorentzian generator would draw, reached as
    # one endpoint of this grid's own sigma_inhom axis instead of a second model class.
    # "both" now means "the whole sigma_inhom range, Lorentzian limit included."
    extra_generators = _voigt_inhom_param_grid()
    return RunGroup(
        name="both-sbed",
        description=(
            "Width x contrast x sigma_inhom x noise grid (plain Voigt; sigma_inhom=0 "
            "is the pure-Lorentzian limit) for Bayesian-SBED/SimpleSobol/SimpleSweep."
        ),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["Bayesian-SBED", "SimpleSobol", "SimpleSweep"],
        extra_generators=extra_generators,
    )


def _group_both_sbed_only() -> RunGroup:
    extra_generators = _voigt_inhom_param_grid()
    return RunGroup(
        name="both-sbed-only",
        description=(
            "Width x contrast x sigma_inhom x noise grid (plain Voigt; sigma_inhom=0 "
            "is the pure-Lorentzian limit) for Bayesian-SBED only (no sweep/sobol baselines)."
        ),
        generator_names=list(extra_generators.keys()),
        noise_names=_sbed_noise_names(),
        strategy_names=["Bayesian-SBED"],
        extra_generators=extra_generators,
    )


def _group_both_sweep_only() -> RunGroup:
    extra_generators = _voigt_inhom_param_grid()
    return RunGroup(
        name="both-sweep-only",
        description=(
            "Width x contrast x sigma_inhom x noise grid (plain Voigt; sigma_inhom=0 "
            "is the pure-Lorentzian limit) for SimpleSweep only (no SBED/Sobol baselines)."
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
        _group_voigt_sbed(),
        _group_voigt_sbed_only(),
        _group_voigt_sweep_only(),
        _group_voigt_inhom_sbed(),
        _group_voigt_inhom_sbed_only(),
        _group_voigt_inhom_sweep_only(),
        _group_lorentzian_plain_sbed(),
        _group_lorentzian_plain_sbed_only(),
        _group_lorentzian_plain_sweep_only(),
        _group_both_sbed(),
        _group_both_sbed_only(),
        _group_both_sweep_only(),
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
