from __future__ import annotations

from .core import CompositeNoise
from .gen import (
    GaussianManufacturer,
    OnePeakGenerator,
    RabiManufacturer,
    T1DecayManufacturer,
    TwoPeakGenerator,
)
from .noises import (
    OverTimeDriftNoise,
    OverVoltageGaussianNoise,
    OverVoltageOutlierSpikes,
    OverVoltagePoissonNoise,
)


# Generators: basic set of signal types (simple first)
def generators_basic() -> list[tuple[str, object]]:
    return [
        (
            "OnePeak-gaussian",
            OnePeakGenerator(manufacturer=GaussianManufacturer()),
        ),
        (
            "OnePeak-rabi",
            OnePeakGenerator(manufacturer=RabiManufacturer()),
        ),
        (
            "OnePeak-t1_decay",
            OnePeakGenerator(manufacturer=T1DecayManufacturer()),
        ),
        (
            "TwoPeak",
            TwoPeakGenerator(
                manufacturer_left=GaussianManufacturer(),
                manufacturer_right=GaussianManufacturer(),
            ),
        ),
    ]


# Noise tiers: start simple and evolve

def noises_none() -> list[tuple[str, CompositeNoise | None]]:
    return [("NoNoise", None)]


def noises_single_each() -> list[tuple[str, CompositeNoise | None]]:
    return [
        ("Gauss(0.05)", CompositeNoise([OverVoltageGaussianNoise(0.05)])),
        ("Poisson(50)", CompositeNoise([OverVoltagePoissonNoise(scale=50.0)])),
    ]


def noises_complex() -> list[tuple[str, CompositeNoise | None]]:
    return [
        (
            "Heavy",
            CompositeNoise([
                OverVoltageGaussianNoise(0.1),
                OverTimeDriftNoise(0.05),
                OverVoltageOutlierSpikes(0.02, 0.5),
            ]),
        ),
    ]
