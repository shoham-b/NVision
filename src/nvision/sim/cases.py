from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from nvision.models.noise import (
    CompositeNoise,
    CompositeOverFrequencyNoise,
    CompositeOverProbeNoise,
)

from .gen.core_generators import (
    NVCenterCoreGenerator,
    OnePeakCoreGenerator,
    TwoPeakCoreGenerator,
)
from .grid_enums import GeneratorCategory, GeneratorName, StrategyFilter
from .noises import (
    OverFrequencyGaussianNoise,
    OverFrequencyOutlierSpikes,
    OverFrequencyPoissonNoise,
    OverProbeDriftNoise,
)


class RunCaseName(StrEnum):
    ALL = "all"
    NVCENTER = "nvcenter"
    NVCENTER_BAYES_SBED = "nvcenter_bayes_sbed"
    NVCENTER_BAYES_UCB = "nvcenter_bayes_ucb"
    NVCENTER_BAYES_MAXVAR = "nvcenter_bayes_maxvar"
    NVCENTER_BAYES_MAXLIKELIHOOD = "nvcenter_bayes_maxlikelihood"
    NVCENTER_BAYES_UTILITY = "nvcenter_bayes_utility"


# Generators: three main categories with subcategories
# Now using core architecture with TrueSignal and explicit SignalModels
def generators_basic() -> list[tuple[str, object]]:
    return [
        # One Peak generators - for each signal type
        (
            "OnePeak-gaussian",
            OnePeakCoreGenerator(x_min=2.6e9, x_max=3.1e9, peak_type="gaussian"),
        ),
        (
            "OnePeak-lorentzian",
            OnePeakCoreGenerator(x_min=2.6e9, x_max=3.1e9, peak_type="lorentzian"),
        ),
        # Two Peak generators - for each signal type
        (
            "TwoPeak-gaussian",
            TwoPeakCoreGenerator(
                x_min=2.6e9,
                x_max=3.1e9,
                peak_type_left="gaussian",
                peak_type_right="gaussian",
            ),
        ),
        (
            "TwoPeak-lorentzian",
            TwoPeakCoreGenerator(
                x_min=2.6e9,
                x_max=3.1e9,
                peak_type_left="lorentzian",
                peak_type_right="lorentzian",
            ),
        ),
        # NV Center generators - different variants
        (
            "NVCenter-one_peak",
            NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian", zero_field=True),
        ),
        (
            "NVCenter-zeeman",
            NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian", zero_field=False),
        ),
        (
            "NVCenter-voigt_one_peak",
            NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="voigt", zero_field=True),
        ),
        (
            "NVCenter-voigt_zeeman",
            NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="voigt", zero_field=False),
        ),
    ]


# Noise tiers: start simple and evolve


def noises_none() -> list[tuple[str, CompositeNoise | None]]:
    return [("NoNoise", None)]


def noises_single_each() -> list[tuple[str, CompositeNoise | None]]:
    return [
        (
            "Gauss(0.05)",
            CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.05)])),
        ),
        (
            "Poisson(50)",
            CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyPoissonNoise(scale=50.0)])),
        ),
        (
            "OverProbeDrift(0.001)",
            CompositeNoise(over_probe_noise=CompositeOverProbeNoise([OverProbeDriftNoise(0.001)])),
        ),
    ]


def noises_complex() -> list[tuple[str, CompositeNoise | None]]:
    return [
        (
            "Heavy",
            CompositeNoise(
                over_frequency_noise=CompositeOverFrequencyNoise(
                    [OverFrequencyGaussianNoise(0.1), OverFrequencyOutlierSpikes(0.02, 0.5)]
                ),
                over_probe_noise=CompositeOverProbeNoise([OverProbeDriftNoise(0.001)]),
            ),
        ),
    ]


@dataclass(frozen=True, slots=True)
class RunCase:
    """Named run preset for the CLI (no user parameters required).

    ``filter_strategy`` uses :class:`nvision.sim.grid_enums.StrategyFilter` values
    (substring match in :meth:`nvision.sim.combinations.CombinationGrid.iter`).

    ``filter_generator`` optionally restricts to a :class:`~nvision.sim.grid_enums.GeneratorName`.
    """

    name: str
    filter_category: GeneratorCategory | None
    filter_strategy: StrategyFilter | None
    filter_generator: GeneratorName | None = None
    description: str = ""
    repeats: int = 5
    loc_max_steps: int = 150
    loc_timeout_s: int = 1500
    require_cache: bool = False
    log_level: str = "INFO"
    no_progress: bool = False


def run_case_nvcenter() -> RunCase:
    """Default NVCenter run case."""
    return RunCase(
        name=RunCaseName.NVCENTER.value,
        filter_category=GeneratorCategory.NVCENTER,
        filter_strategy=None,
        description="NVCenter generators with all available strategies (SimpleSweep + Bayesian).",
        repeats=5,
        loc_max_steps=150,
        loc_timeout_s=1500,
        require_cache=False,
        log_level="INFO",
        no_progress=False,
    )


def run_case_all() -> RunCase:
    """Run every generator/noise/strategy combination."""
    return RunCase(
        name=RunCaseName.ALL.value,
        filter_category=None,
        filter_strategy=None,
        description="All generators, noises, and strategies.",
        repeats=5,
        loc_max_steps=150,
        loc_timeout_s=1500,
        require_cache=False,
        log_level="INFO",
        no_progress=False,
    )


def run_case_nvcenter_bayes_sbed() -> RunCase:
    """NVCenter Bayesian SBED run case."""
    return RunCase(
        name=RunCaseName.NVCENTER_BAYES_SBED.value,
        filter_category=GeneratorCategory.NVCENTER,
        filter_strategy=StrategyFilter.BAYESIAN_SBED,
        description="NVCenter generators, SBED acquisition (matches strategy name 'Bayesian-SBED').",
        repeats=5,
        loc_max_steps=200,
        loc_timeout_s=2000,
        require_cache=False,
        log_level="INFO",
        no_progress=False,
    )


def run_case_nvcenter_bayes_ucb() -> RunCase:
    """NVCenter Bayesian UCB run case."""
    return RunCase(
        name=RunCaseName.NVCENTER_BAYES_UCB.value,
        filter_category=GeneratorCategory.NVCENTER,
        filter_strategy=StrategyFilter.BAYESIAN_UCB,
        description="NVCenter generators, UCB acquisition (matches 'Bayesian-UCB').",
        repeats=5,
        loc_max_steps=200,
        loc_timeout_s=2000,
        require_cache=False,
        log_level="INFO",
        no_progress=False,
    )


def run_case_nvcenter_bayes_maxvar() -> RunCase:
    """NVCenter Bayesian MaxVariance run case."""
    return RunCase(
        name=RunCaseName.NVCENTER_BAYES_MAXVAR.value,
        filter_category=GeneratorCategory.NVCENTER,
        filter_strategy=StrategyFilter.BAYESIAN_MAX_VARIANCE,
        description="NVCenter generators, max-variance acquisition (matches 'Bayesian-MaxVariance').",
        repeats=5,
        loc_max_steps=200,
        loc_timeout_s=2000,
        require_cache=False,
        log_level="INFO",
        no_progress=False,
    )


def run_case_nvcenter_bayes_maxlikelihood() -> RunCase:
    """NVCenter Bayesian MaximumLikelihood run case."""
    return RunCase(
        name=RunCaseName.NVCENTER_BAYES_MAXLIKELIHOOD.value,
        filter_category=GeneratorCategory.NVCENTER,
        filter_strategy=StrategyFilter.BAYESIAN_MAXIMUM_LIKELIHOOD,
        description="NVCenter generators, maximum likelihood acquisition (matches 'Bayesian-MaximumLikelihood').",
        repeats=5,
        loc_max_steps=200,
        loc_timeout_s=2000,
        require_cache=False,
        log_level="INFO",
        no_progress=False,
    )


def run_case_nvcenter_bayes_utility() -> RunCase:
    """NVCenter Bayesian UtilitySampling run case."""
    return RunCase(
        name=RunCaseName.NVCENTER_BAYES_UTILITY.value,
        filter_category=GeneratorCategory.NVCENTER,
        filter_strategy=StrategyFilter.BAYESIAN_UTILITY_SAMPLING,
        description="NVCenter generators, utility sampling (matches 'Bayesian-UtilitySampling').",
        repeats=5,
        loc_max_steps=200,
        loc_timeout_s=2000,
        require_cache=False,
        log_level="INFO",
        no_progress=False,
    )


@lru_cache(maxsize=1)
def _run_cases_tuple() -> tuple[RunCase, ...]:
    return (
        run_case_all(),
        run_case_nvcenter(),
        run_case_nvcenter_bayes_sbed(),
        run_case_nvcenter_bayes_ucb(),
        run_case_nvcenter_bayes_maxvar(),
        run_case_nvcenter_bayes_maxlikelihood(),
        run_case_nvcenter_bayes_utility(),
    )


def run_cases() -> list[RunCase]:
    return list(_run_cases_tuple())


@lru_cache(maxsize=1)
def _run_case_by_normalized_name() -> dict[str, RunCase]:
    return {c.name.lower(): c for c in _run_cases_tuple()}


def get_run_case(name: RunCaseName | str) -> RunCase:
    key = name.value if isinstance(name, RunCaseName) else name.strip().lower()
    try:
        return _run_case_by_normalized_name()[key]
    except KeyError:
        raise KeyError(f"Unknown run case: {name!r}") from None


def clear_run_case_cache() -> None:
    """Drop :func:`_run_cases_tuple` / lookup caches (e.g. if presets are monkeypatched in tests)."""
    _run_cases_tuple.cache_clear()
    _run_case_by_normalized_name.cache_clear()


def default_run_case() -> RunCase:
    return run_case_nvcenter()
