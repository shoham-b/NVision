"""Run group registry — explicit preset combinations for the CLI.

Each :class:`RunGroup` holds concrete lists of generator, noise, and strategy
names.  The runner resolves them through :class:`CombinationGrid` rather than
relying on string filtering.

A group is a *scenario* (what is simulated: generator grid x noise axis) with the locators compared on it.
To run a subset of a group's locators or signals use ``--filter-strategy`` / ``--filter-signal``; the former
``*-only`` groups are kept as aliases of exactly that.

====================  ==========================================  ==================  ===========================
group                 generators                                  noise               strategies
====================  ==========================================  ==================  ===========================
``voigt-grid``        Voigt window/FWHM x share x isotope         SNR grid            locator study (below)
``drive-control``     CW / pulsed drive-controlled NV x isotope   SNR grid (nominal)  Bayesian-SBED, -SBED-Drive
``all``               the two above                               the two above       each scenario's own set
====================  ==========================================  ==================  ===========================

The locator study is SimpleSweep-<k>perFWHM (k = 2..4), Smart-Sweep, Bayesian-SBED and Smart-SBED.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

from nvision.sim import presets as sim_presets
from nvision.sim.combinations import SWEEP_SPACING_DIVISORS
from nvision.sim.grid_enums import sweep_strategy_name
from nvision.sim.locs.search_grid import SWEEP_POINTS_PER_FWHM


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
    # A composite group (``all``) runs each member's own combinations; its generator/noise/strategy lists are the
    # unions over the members and are not a Cartesian product.
    members: tuple[RunGroup, ...] = ()

    def combinations(self) -> list[tuple[str, str, str]]:
        """Every (generator, noise, strategy) triple of this group, without duplicates."""
        if self.members:
            return list(dict.fromkeys(c for m in self.members for c in m.combinations()))
        return [(g, n, s) for g in self.generator_names for n in self.noise_names for s in self.strategy_names]

    @property
    def task_count(self) -> int:
        """Number of (generator, noise, strategy) combinations."""
        return len(self.combinations())


_DEFAULT_SWEEP = sweep_strategy_name(SWEEP_POINTS_PER_FWHM)

# The locators compared on the standard scenarios: the sweeps at several spacings (points per FWHM), the plain
# SBED, and the two locators that search coarsely first and then work inside the focus (docs/locator_comparison.md).
_LOCATOR_STUDY_STRATEGIES = [
    *(sweep_strategy_name(k) for k in SWEEP_SPACING_DIVISORS),
    "Smart-Sweep",
    "Bayesian-SBED",
    "Smart-SBED",
]

_STUDY_LOCATORS = "all locators (sweeps, Smart-Sweep, Bayesian-SBED, Smart-SBED)"


def _voigt_snr_window_param_grid() -> dict[str, object]:
    """Window-in-FWHM x sigma_inhom x isotope grid (plain Voigt, fixed linewidth, fixed contrast).

    The noise axis (:func:`sim.presets.snr_study_noises`) supplies the SNR; the signal axes are the window
    as a multiple of the FWHM, the inhomogeneous broadening and the isotope. sigma_inhom=0 makes
    NVCenterVoigtModel's reparam draw lorentz_frac=1.0 -- a pure-Lorentzian-shaped pseudo-Voigt profile, so
    the Lorentzian limit is one endpoint of the sigma_inhom axis instead of a second model class. The earlier
    width x contrast x sigma_inhom grid is still resolvable by name (``sim.presets``)."""
    return dict(sim_presets.voigt_snr_window_param_grid_generators())


def _snr_noise_names() -> list[str]:
    """Noise grid of the SNR studies: one SNR(x) per level (see sim.presets.snr_study_noises)."""
    return [name for name, _ in sim_presets.snr_study_noises()]


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


def _group_voigt_grid() -> RunGroup:
    return _group(
        "voigt-grid",
        f"Voigt window/FWHM x Gaussian share x isotope (fixed total FWHM) x SNR grid; {_STUDY_LOCATORS}.",
        _voigt_snr_window_param_grid(),
        _snr_noise_names(),
        list(_LOCATOR_STUDY_STRATEGIES),
    )


def _group_drive_control() -> RunGroup:
    """Drive-choosing SBED vs fixed-drive SBED on the CW and pulsed drive-controlled generators."""
    return _group(
        "drive-control",
        "CW/pulsed drive-controlled NV generators across isotopes; "
        "Bayesian-SBED-Drive (picks the MW amplitude) vs Bayesian-SBED.",
        dict(sim_presets.drive_controlled_generators()),
        _snr_noise_names(),
        ["Bayesian-SBED", "Bayesian-SBED-Drive"],
    )


def _group_all() -> RunGroup:
    """Every scenario with every locator used in it."""
    members = (_group_voigt_grid(), _group_drive_control())

    def union(lists):
        return list(dict.fromkeys(x for xs in lists for x in xs))

    return RunGroup(
        name="all",
        description="Every scenario above with all of its locators (voigt-grid + drive-control).",
        generator_names=union(m.generator_names for m in members),
        noise_names=union(m.noise_names for m in members),
        strategy_names=union(m.strategy_names for m in members),
        extra_generators={k: v for m in members for k, v in (m.extra_generators or {}).items()},
        members=members,
    )


# Former group names: a canonical group restricted to a subset of its strategies (what ``--filter-strategy``
# does on the canonical group). They resolve through get_run_group but are not listed by run_groups().
_ALIASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "both-sbed": ("voigt-grid", ("Bayesian-SBED", _DEFAULT_SWEEP)),
    "both-sbed-only": ("voigt-grid", ("Bayesian-SBED",)),
    "both-sweep-only": ("voigt-grid", (_DEFAULT_SWEEP,)),
}


def alias_names() -> list[str]:
    """Former group names that still resolve (see :func:`get_run_group`)."""
    return list(_ALIASES)


@lru_cache(maxsize=1)
def _run_groups_tuple() -> tuple[RunGroup, ...]:
    return (
        _group_voigt_grid(),
        _group_drive_control(),
        _group_all(),
    )


def run_groups() -> list[RunGroup]:
    """The listed groups, one per scenario; former names are in :func:`alias_names`."""
    return list(_run_groups_tuple())


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_")


@lru_cache(maxsize=1)
def _run_group_by_normalized_name() -> dict[str, RunGroup]:
    by_name = {_normalize(g.name): g for g in _run_groups_tuple()}
    for alias, (canonical, strategies) in _ALIASES.items():
        by_name[_normalize(alias)] = replace(
            by_name[_normalize(canonical)],
            name=alias,
            description=f"Alias for {canonical} restricted to {', '.join(strategies)}.",
            strategy_names=list(strategies),
        )
    return by_name


def get_run_group(name: str) -> RunGroup:
    try:
        return _run_group_by_normalized_name()[_normalize(name)]
    except KeyError:
        raise KeyError(f"Unknown run group: {name!r}") from None


def snr_of_noise(noise_name: str) -> float | None:
    """The SNR of an ``SNR(x)`` noise name; ``None`` for any other name."""
    from nvision.sim.combinations import parse_snr

    return parse_snr(noise_name)


def select_snr_window(
    combinations: list[tuple[str, str, str]],
    generators: dict[str, object],
    *,
    snr: list[float] | None = None,
    signal_occupancy: list[float] | None = None,
) -> list[tuple[str, str, str]]:
    """The (generator, noise, strategy) triples at the given SNR levels and/or signal occupancies.

    The signal occupancy is the share of the drive-frequency window one dip group covers.

    ``generators`` maps generator names to their objects; a generator without a ``signal_occupancy`` (not a window-grid
    signal) matches no ``signal_occupancy`` selection, a noise without an SNR matches no ``snr`` selection.
    """

    def matches(value: float | None, wanted: list[float] | None) -> bool:
        return wanted is None or (value is not None and any(math.isclose(value, w, rel_tol=1e-6) for w in wanted))

    return [
        (g, n, s)
        for g, n, s in combinations
        if matches(getattr(generators.get(g), "signal_occupancy", None), signal_occupancy)
        and matches(snr_of_noise(n), snr)
    ]


def clear_run_group_cache() -> None:
    """Drop lookup caches (e.g. if presets are monkeypatched in tests)."""
    _run_groups_tuple.cache_clear()
    _run_group_by_normalized_name.cache_clear()


def default_run_group() -> RunGroup:
    return _run_groups_tuple()[0]
