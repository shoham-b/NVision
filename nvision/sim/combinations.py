"""Combination grid — the full (generator x noise x strategy) space.

``CombinationGrid`` is the single source of truth for which generators,
noises, and locator strategies exist and how they combine.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from nvision.belief.smc_marginal import (
    NVISION_SMC_A_PARAM,
    NVISION_SMC_ESS_THRESHOLD,
    NVISION_SMC_NUM_PARTICLES,
)
from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
from nvision.noises import OverFrequencyGaussianNoise
from nvision.sim import presets as sim_presets
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.sobol_bayesian_locator import SimpleSobolBayesianLocator
from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator
from nvision.sim.locs.coarse.sobol_locator import StagedSobolSweepLocator


@dataclass(frozen=True, slots=True)
class Combination:
    """One (generator, noise, strategy) triple — the 'what' of a run."""

    generator_name: str
    generator: object
    noise_name: str
    noise: CompositeNoise | None
    strategy_name: str
    strategy: type | dict[str, Any]


_NV_SMC: dict[str, object] = {
    "builder": nv_center_smc_belief,
    "num_particles": NVISION_SMC_NUM_PARTICLES,
    "ess_threshold": NVISION_SMC_ESS_THRESHOLD,
    "a_param": NVISION_SMC_A_PARAM,
}

_GAUSS_RE = re.compile(r"^Gauss\(([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\)$")


def parse_gauss_sigma(name: str) -> float | None:
    """Extract the numeric sigma from a ``Gauss(sigma)`` noise name, else ``None``.

    Gives metrics/plots a numeric noise axis instead of the opaque preset string.
    """
    m = _GAUSS_RE.match(name)
    return float(m.group(1)) if m is not None else None


def _parse_noise(name: str) -> CompositeNoise | None:
    """Dynamically parse a ``Gauss(sigma)`` noise descriptor into a :class:`CompositeNoise`.

    Makes ``run-single`` work for any sigma value even when
    :envvar:`NVISION_NOISE_MAX_GAUSS` caps the preset grid below the requested level.

    Args:
        name: A string like ``'Gauss(0.05)'``.

    Returns:
        A :class:`CompositeNoise` for the parsed descriptor, or ``None`` if
        *name* does not match the expected pattern.
    """
    m = _GAUSS_RE.match(name)
    if m is not None:
        sigma = float(m.group(1))
        return CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(sigma)]))
    return None


def _strategy_matches(pattern: str, strat_name: str) -> bool:
    """Match strategy name against pattern (regex if valid, else substring)."""
    # Try regex first
    try:
        if pattern.startswith("^") or pattern.endswith("$") or any(c in pattern for c in ".*+?[]{}|()"):
            return bool(re.search(pattern, strat_name))
    except re.error:
        pass
    # Fall back to simple substring match
    return pattern in strat_name


class CombinationGrid:
    """Enumerates every (generator x noise x strategy) combination.

    Holds the full configuration grid in one place so the runner, CLI,
    and render code never duplicate the mapping logic.
    """

    def __init__(self, extra_generators: dict[str, object] | None = None) -> None:
        self._generators: dict[str, object] = dict(sim_presets.generators_basic())
        # Name -> generator lookup used only by resolve(), never by iter()/all_combinations().
        # Always includes the SBED run-groups' study grids so cache/metrics/render
        # tooling can resolve historical results by name regardless of how they were produced,
        # without expanding what plain (non-run-group) `nvision run` iterates over.
        self._resolve_generators: dict[str, object] = dict(self._generators)
        self._resolve_generators.update(dict(sim_presets.param_grid_generators()))
        self._resolve_generators.update(dict(sim_presets.param_grid_generators(variant="voigt")))
        self._resolve_generators.update(dict(sim_presets.saturation_voigt_param_grid_generators()))
        if extra_generators:
            self._resolve_generators.update(extra_generators)
        self._noises: dict[str, CompositeNoise | None] = dict(sim_presets.noises_single_each())

    @property
    def generators(self) -> dict[str, object]:
        return self._generators

    @property
    def noises(self) -> dict[str, CompositeNoise | None]:
        return self._noises

    @staticmethod
    def generator_category(name: str) -> str:
        if name.startswith("NVCenter-"):
            return "NVCenter"
        return "Unknown"

    def strategies_for(self, generator_name: str) -> list[tuple[str, Any]]:
        """Return the locator strategies appropriate for *generator_name*."""
        # Belief inference must match the generator's lineshape, or the locator
        # models the wrong signal shape entirely. nv_center_smc_belief defaults
        # to "lorentzian"; only study grids for other lineshapes need an override.
        # Scoped to the "-w"-suffixed width x contrast grid names specifically --
        # the bare "NVCenter-voigt" generator from generators_basic() has the same
        # gap but is intentionally left alone here (pre-existing, separate issue).
        nv_smc_config = dict(_NV_SMC)
        if generator_name.startswith("NVCenter-saturation_voigt"):
            nv_smc_config["lineshape"] = "saturation_voigt"
        elif generator_name.startswith("NVCenter-voigt-w"):
            nv_smc_config["lineshape"] = "voigt"
            # param_grid_generators(variant="voigt") (names WITHOUT a "-si" suffix)
            # explicitly draws a real hyperfine triplet per Zeeman group
            # (with_hyperfine_splitting=True, see presets.py) to preserve that grid's
            # historical 6-dip behavior -- the belief must match. The separate
            # voigt_sigma_inhom_param_grid_generators() grid (names WITH a "-si"
            # suffix) does NOT set with_hyperfine_splitting (generator default: False,
            # a plain Zeeman-only 2-dip signal) -- forcing True here for it as well
            # was a real belief/generator mismatch (the belief hunting for a
            # split/k_np hyperfine substructure that the true signal never has),
            # so it's excluded from this override and falls through to
            # nv_center_smc_belief's own with_hyperfine_splitting=False default.
            if "-si" not in generator_name:
                nv_smc_config["with_hyperfine_splitting"] = True

        strats = [
            ("SimpleSweep", GenericSweepLocator),
            ("StagedSobolSweep", StagedSobolSweepLocator),
            (
                "Bayesian-SBED",
                {
                    "class": SequentialBayesianExperimentDesignLocator,
                    "config": {"max_steps": 200, **nv_smc_config},
                },
            ),
            (
                "SimpleSobol",
                {
                    "class": SimpleSobolBayesianLocator,
                    "config": {
                        "max_steps": 10000,
                        **nv_smc_config,
                    },
                },
            ),
        ]

        return strats

    def __iter__(self) -> Iterator[Combination]:
        """Iterate all combinations (no filtering, no dedup)."""
        return self.all_combinations()

    def iter(  # noqa: C901
        self,
        filter_category: str | None = None,
        filter_strategy: str | None = None,
        filter_generator: str | None = None,
        filter_noise: str | None = None,
        filter_signal: str | None = None,
    ) -> Iterator[Combination]:
        """Yield every matching combination, deduplicating automatically.

        .. deprecated::
            Filtering via ``iter()`` is kept for backward compatibility.
            Prefer :meth:`all_combinations` or :meth:`resolve`.
        """
        seen: set[tuple[str, str, str]] = set()

        for gen_name, gen_obj in self._generators.items():
            if filter_category and self.generator_category(gen_name) != filter_category:
                continue
            if filter_generator is not None and gen_name != filter_generator:
                continue
            if filter_signal is not None and filter_signal not in gen_name:
                continue

            for strat_name, strat_obj in self.strategies_for(gen_name):
                if filter_strategy:
                    # Support comma-separated strategy filters with optional regex
                    # (e.g., "Sweep,Sobol" matches both, "^Bayesian.*" uses regex)
                    patterns = [p.strip() for p in filter_strategy.split(",")]
                    if not any(_strategy_matches(p, strat_name) for p in patterns):
                        continue

                for noise_name, noise_obj in self._noises.items():
                    if filter_noise is not None:
                        noise_patterns = [p.strip() for p in filter_noise.split(",")]
                        if not any(noise_name.startswith(p) for p in noise_patterns):
                            continue
                    key = (gen_name, noise_name, strat_name)
                    if key in seen:
                        continue
                    seen.add(key)

                    yield Combination(
                        generator_name=gen_name,
                        generator=gen_obj,
                        noise_name=noise_name,
                        noise=noise_obj,
                        strategy_name=strat_name,
                        strategy=strat_obj,
                    )

    def all_combinations(self) -> Iterator[Combination]:
        """Yield every registered combination (full Cartesian product)."""
        seen: set[tuple[str, str, str]] = set()
        for gen_name, gen_obj in self._generators.items():
            for strat_name, strat_obj in self.strategies_for(gen_name):
                for noise_name, noise_obj in self._noises.items():
                    key = (gen_name, noise_name, strat_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield Combination(
                        generator_name=gen_name,
                        generator=gen_obj,
                        noise_name=noise_name,
                        noise=noise_obj,
                        strategy=strat_obj,
                        strategy_name=strat_name,
                    )

    def resolve(self, gen_name: str, noise_name: str, strat_name: str) -> Combination | None:
        """Resolve three preset names to a single :class:`Combination`.

        Returns ``None`` if the generator name or strategy name is not registered.

        For *noise_name*, the registered grid is checked first.  If the name is not
        found there (e.g. because :envvar:`NVISION_NOISE_MAX_GAUSS` caps the preset
        grid below the requested sigma), :func:`_parse_noise` is used to build the
        noise object on the fly from the descriptor string (e.g. ``'Gauss(0.15)'``).
        This ensures ``run-single`` works for any valid noise value regardless of
        the active grid configuration.
        """
        gen_obj = self._resolve_generators.get(gen_name)
        if gen_obj is None:
            return None
        noise_obj: CompositeNoise | None
        if noise_name in self._noises:
            noise_obj = self._noises[noise_name]
        else:
            # Build the noise on the fly for any descriptor not in the current grid.
            noise_obj = _parse_noise(noise_name)
            if noise_obj is None:
                return None
        for s_name, s_obj in self.strategies_for(gen_name):
            if s_name == strat_name:
                return Combination(
                    generator_name=gen_name,
                    generator=gen_obj,
                    noise_name=noise_name,
                    noise=noise_obj,
                    strategy_name=strat_name,
                    strategy=s_obj,
                )
        return None
