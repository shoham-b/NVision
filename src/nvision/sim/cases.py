from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .core import (
    CompositeNoise,
    CompositeOverFrequencyNoise,
    CompositeOverProbeNoise,
)
from .gen.core_generators import (
    NVCenterCoreGenerator,
    OnePeakCoreGenerator,
    TwoPeakCoreGenerator,
)
from .noises import (
    OverFrequencyGaussianNoise,
    OverFrequencyOutlierSpikes,
    OverFrequencyPoissonNoise,
    OverProbeDriftNoise,
)


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
    """Named run preset for the CLI (no user parameters required)."""

    name: str
    filter_category: Literal["NVCenter", "OnePeak", "TwoPeak"] | None
    filter_strategy: str | None
    repeats: int = 5
    seed: int = 123
    loc_max_steps: int = 150
    loc_timeout_s: int = 1500
    no_cache: bool = False
    require_cache: bool = False
    log_level: str = "INFO"
    no_progress: bool = False


def run_case_nvcenter() -> RunCase:
    """Default NVCenter run case."""
    return RunCase(
        name="nvcenter",
        filter_category="NVCenter",
        filter_strategy="SimpleSweep",
        repeats=5,
        seed=123,
        loc_max_steps=150,
        loc_timeout_s=1500,
        no_cache=False,
        require_cache=False,
        log_level="INFO",
        no_progress=False,
    )


def run_case_nvcenter_bayes_eig() -> RunCase:
    """NVCenter Bayesian EIG run case."""
    return RunCase(
        name="nvcenter_bayes_eig",
        filter_category="NVCenter",
        filter_strategy="NVCenter-Bayesian-EIG",
        repeats=5,
        seed=123,
        loc_max_steps=200,
        loc_timeout_s=2000,
        no_cache=False,
        require_cache=False,
        log_level="INFO",
        no_progress=False,
    )


def run_case_nvcenter_bayes_ucb() -> RunCase:
    """NVCenter Bayesian UCB run case."""
    return RunCase(
        name="nvcenter_bayes_ucb",
        filter_category="NVCenter",
        filter_strategy="NVCenter-Bayesian-UCB",
        repeats=5,
        seed=123,
        loc_max_steps=200,
        loc_timeout_s=2000,
        no_cache=False,
        require_cache=False,
        log_level="INFO",
        no_progress=False,
    )


def run_cases() -> list[RunCase]:
    return [
        run_case_nvcenter(),
        run_case_nvcenter_bayes_eig(),
        run_case_nvcenter_bayes_ucb(),
    ]


def get_run_case(name: str) -> RunCase:
    key = name.strip().lower()
    for case in run_cases():
        if case.name.lower() == key:
            return case
    raise KeyError(f"Unknown run case: {name!r}")


def default_run_case() -> RunCase:
    return run_case_nvcenter()
