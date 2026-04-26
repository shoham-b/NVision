"""Combination grid — the full (generator x noise x strategy) space.

``CombinationGrid`` is the single source of truth for which generators,
noises, and locator strategies exist and how they combine.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from nvision.models.noise import CompositeNoise
from nvision.sim import cases as sim_cases
from nvision.sim.locs.bayesian.acquisition_locators import (
    MaximumLikelihoodLocator,
    SequentialBayesianExperimentDesignLocator,
    UtilitySamplingLocator,
)
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator
from nvision.sim.locs.coarse.sobol_locator import SobolSweepLocator, StagedSobolSweepLocator


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
    "num_particles": 5000,
    "jitter_scale": 0.05,
    "ess_threshold": 0.5,
    "use_full_covariance": True,  # NIST-style robust resampling (Dushenko et al.)
    "a_param": 0.98,
    "scale": True,
}


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

    def __init__(self) -> None:
        self._generators: dict[str, object] = dict(sim_cases.generators_basic())
        self._noises: dict[str, CompositeNoise | None] = dict(
            sim_cases.noises_none() + sim_cases.noises_single_each() + sim_cases.noises_complex()
        )

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
        if generator_name.startswith("NVCenter-"):
            return [
                ("GenericSweep", GenericSweepLocator),
                ("SimpleSweep", GenericSweepLocator),
                ("SobolSweep", SobolSweepLocator),
                ("StagedSobolSweep", StagedSobolSweepLocator),
                (
                    "Bayesian-SBED",
                    {"class": SequentialBayesianExperimentDesignLocator, "config": {"max_steps": 200, **_NV_SMC}},
                ),
                (
                    "Bayesian-MaximumLikelihood",
                    {"class": MaximumLikelihoodLocator, "config": {"max_steps": 200, **_NV_SMC}},
                ),
                (
                    "Bayesian-UtilitySampling",
                    {
                        "class": UtilitySamplingLocator,
                        "config": {
                            "max_steps": 200,
                            **_NV_SMC,
                            "pickiness": 4.0,
                            "noise_std": 0.02,
                            "cost": 1.0,
                            "n_mc_samples": 64,
                            "n_candidates": 64,
                        },
                    },
                ),
            ]

        return [
            ("GenericSweep", GenericSweepLocator),
            ("SimpleSweep", GenericSweepLocator),
            ("SobolSweep", SobolSweepLocator),
            ("StagedSobolSweep", StagedSobolSweepLocator),
        ]

    def __iter__(self) -> Iterator[Combination]:
        """Iterate all combinations (no filtering, no dedup)."""
        return self.iter(filter_category=None, filter_strategy=None, filter_generator=None, filter_noise=None)

    def iter(  # noqa: C901
        self,
        filter_category: str | None = None,
        filter_strategy: str | None = None,
        filter_generator: str | None = None,
        filter_noise: str | None = None,
        filter_signal: str | None = None,
    ) -> Iterator[Combination]:
        """Yield every matching combination, deduplicating automatically."""
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
                    if filter_noise is not None and not noise_name.startswith(filter_noise):
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
