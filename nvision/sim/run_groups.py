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
``drive-control``    CW / pulsed drive-controlled NV x isotope   SNR grid (nominal)  Bayesian-SBED, -SBED-Drive
``hyperfine-evidence`` Lorentzian, one window x isotope          one SNR             Bayesian-SBED, sweep at k*
``article``           Lorentzian single dip x window span        SNR grid            Bayesian-SBED, sweep at k*
                      (a composite: new article scenarios are members)
``all``               ``voigt-grid`` only (not ``drive-control``) SNR grid            locator study (below)
====================  ==========================================  ==================  ===========================

The locator study is SimpleSweep-optimal (the sweep at its task's analytically optimal step k*, the reference),
its multiples SimpleSweep-optimal-x0.5 and -x2, and Bayesian-SBED.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

from nvision.sim import presets as sim_presets
from nvision.sim.combinations import PLAIN_SWEEP_STRATEGIES
from nvision.sim.grid_enums import OPTIMAL_SWEEP_STRATEGY


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


_DEFAULT_SWEEP = OPTIMAL_SWEEP_STRATEGY

# The locators compared on the standard scenarios: the plain sweeps (at the optimal step k* and at multiples of it),
# and the plain SBED (docs/locator_comparison.md).
_LOCATOR_STUDY_STRATEGIES = [
    *PLAIN_SWEEP_STRATEGIES,
    "Bayesian-SBED",
]

_STUDY_LOCATORS = "all locators (sweeps, Bayesian-SBED)"


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


# The one window and SNR at which the resolved isotopes are compared with the single dip. Span 10 of the single dip
# is the 20 MHz window of the lab sweeps at about their linewidth; the span is in widths one dip group covers, so
# every isotope has the same occupancy (1/10) and a wider absolute window.
_HYPERFINE_EVIDENCE_SPAN = 10.0
_HYPERFINE_EVIDENCE_SNR = 5.0


def _group_hyperfine_evidence() -> RunGroup:
    """The single dip vs the resolved 14N triplet and 15N doublet, Lorentzian, at one window and one SNR.

    Evidence that SBED behaves the same with resolved hyperfine lines (the lab line is a single dip, its nuclear spin
    polarized near the excited-state level anticrossing). The single-dip combinations are the voigt-grid's own, so
    they come from the cache."""
    from nvision.sim.defaults import NVISION_SBED_HYPERFINE_VALUES

    for iso in ("unresolved", "n14", "n15"):
        if iso not in NVISION_SBED_HYPERFINE_VALUES:
            raise ValueError(f"hyperfine-evidence needs isotope {iso!r} in NVISION_SBED_HYPERFINE_VALUES")
    generators = {
        name: gen
        for name, gen in _voigt_snr_window_param_grid().items()
        if "-gs0-" in name and math.isclose(1.0 / gen.signal_occupancy, _HYPERFINE_EVIDENCE_SPAN, rel_tol=1e-6)
    }
    if len(generators) != 3:
        raise ValueError(f"hyperfine-evidence expected 3 generators (one per isotope), got {sorted(generators)}")
    snr_noise = [n for n in _snr_noise_names() if snr_of_noise(n) == _HYPERFINE_EVIDENCE_SNR]
    if len(snr_noise) != 1:
        raise ValueError(f"hyperfine-evidence needs SNR {_HYPERFINE_EVIDENCE_SNR:g} in NVISION_SBED_SNR_VALUES")
    return _group(
        "hyperfine-evidence",
        f"Lorentzian single dip vs 14N triplet vs 15N doublet at span {_HYPERFINE_EVIDENCE_SPAN:g}, "
        f"SNR {_HYPERFINE_EVIDENCE_SNR:g}; Bayesian-SBED and the sweep at k*.",
        generators,
        snr_noise,
        ["Bayesian-SBED", _DEFAULT_SWEEP],
    )


def _group_article_lorentzian_search() -> RunGroup:
    """The article's main simulation: the pure Lorentzian single dip at every window span, over the SNR grid.

    Bayesian-SBED against the sweep at its optimal step k* only (no multiples). Every combination is also a
    voigt-grid combination, so it is served from the same cache."""
    generators = {
        name: gen
        for name, gen in _voigt_snr_window_param_grid().items()
        if gen.sigma_inhom == 0.0 and gen.hyperfine == "unresolved"
    }
    from nvision.sim.defaults import NVISION_SBED_WINDOW_SPANS

    spans = sorted(1.0 / gen.signal_occupancy for gen in generators.values())
    expected = sorted(NVISION_SBED_WINDOW_SPANS)
    if len(generators) != len(expected) or not all(
        math.isclose(a, b, rel_tol=1e-6) for a, b in zip(spans, expected, strict=True)
    ):
        raise ValueError(
            f"article expected one pure-Lorentzian single-dip generator per span {expected}, got {sorted(generators)}"
        )
    return _group(
        "article-lorentzian-search",
        "Pure Lorentzian single dip at every window span x SNR grid; Bayesian-SBED vs the sweep at k*.",
        generators,
        _snr_noise_names(),
        ["Bayesian-SBED", _DEFAULT_SWEEP],
    )


def _group_article() -> RunGroup:
    """Every simulated use case of the article, as a composite so further scenarios (a known-shape lab mirror,
    tracking) are added as members."""
    members = (_group_article_lorentzian_search(),)

    def union(lists):
        return list(dict.fromkeys(x for xs in lists for x in xs))

    return RunGroup(
        name="article",
        description="Exactly the simulated use cases of the article (Lorentzian single dip, span x SNR; "
        "Bayesian-SBED vs the sweep at k*).",
        generator_names=union(m.generator_names for m in members),
        noise_names=union(m.noise_names for m in members),
        strategy_names=union(m.strategy_names for m in members),
        extra_generators={k: v for m in members for k, v in (m.extra_generators or {}).items()},
        members=members,
    )


def _group_all() -> RunGroup:
    """Every standard scenario with every locator used in it. ``drive-control`` is run explicitly, not as part of
    ``all``."""
    members = (_group_voigt_grid(),)

    def union(lists):
        return list(dict.fromkeys(x for xs in lists for x in xs))

    return RunGroup(
        name="all",
        description="The standard scenario (voigt-grid) with all of its locators; drive-control is not included.",
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
        _group_hyperfine_evidence(),
        _group_article(),
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
