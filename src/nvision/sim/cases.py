from __future__ import annotations

from .core import (
    CompositeNoise,
    CompositeOverFrequencyNoise,
    CompositeOverProbeNoise,
)
from .gen import (
    CauchyLorentzPeakManufacturer,
    ExponentialDecayManufacturer,
    GaussianManufacturer,
    NVCenterGenerator,
    OnePeakGenerator,
    TwoPeakGenerator,
)
from .noises import (
    OverFrequencyGaussianNoise,
    OverFrequencyOutlierSpikes,
    OverFrequencyPoissonNoise,
    OverProbeDriftNoise,
)


# Generators: three main categories with subcategories
def generators_basic() -> list[tuple[str, object]]:
    return [
        # One Peak generators - for each distribution manufacturer
        (
            "OnePeak-gaussian",
            OnePeakGenerator(manufacturer=GaussianManufacturer()),
        ),
        (
            "OnePeak-cauchy",
            OnePeakGenerator(manufacturer=CauchyLorentzPeakManufacturer()),
        ),
        (
            "OnePeak-exponential",
            OnePeakGenerator(manufacturer=ExponentialDecayManufacturer()),
        ),
        # Two Peak generators - for each distribution
        (
            "TwoPeak-gaussian",
            TwoPeakGenerator(
                manufacturer_left=GaussianManufacturer(),
                manufacturer_right=GaussianManufacturer(),
            ),
        ),
        (
            "TwoPeak-cauchy",
            TwoPeakGenerator(
                manufacturer_left=CauchyLorentzPeakManufacturer(),
                manufacturer_right=CauchyLorentzPeakManufacturer(),
            ),
        ),
        (
            "TwoPeak-exponential",
            TwoPeakGenerator(
                manufacturer_left=ExponentialDecayManufacturer(),
                manufacturer_right=ExponentialDecayManufacturer(),
            ),
        ),
        # NV Center generators - different variants
        (
            "NVCenter-one_peak",
            NVCenterGenerator(variant="one_peak"),
        ),
        (
            "NVCenter-zeeman",
            NVCenterGenerator(variant="zeeman"),
        ),
        (
            "NVCenter-voigt_one_peak",
            NVCenterGenerator(variant="voigt_one_peak"),
        ),
        (
            "NVCenter-voigt_zeeman",
            NVCenterGenerator(variant="voigt_zeeman"),
        ),
    ]


# Noise tiers: start simple and evolve


def noises_none() -> list[tuple[str, CompositeNoise | None]]:
    return [("NoNoise", None)]


def noises_single_each() -> list[tuple[str, CompositeNoise | None]]:
    return [
        (
            "Gauss(0.05)",
            CompositeNoise(
                over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.05)])
            ),
        ),
        (
            "Poisson(50)",
            CompositeNoise(
                over_frequency_noise=CompositeOverFrequencyNoise(
                    [OverFrequencyPoissonNoise(scale=50.0)]
                )
            ),
        ),
        (
            "OverProbeDrift(0.05)",
            CompositeNoise(over_probe_noise=CompositeOverProbeNoise([OverProbeDriftNoise(0.05)])),
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
                over_probe_noise=CompositeOverProbeNoise([OverProbeDriftNoise(0.05)]),
            ),
        ),
    ]
