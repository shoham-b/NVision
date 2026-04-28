"""Preset generators, noises, and constants for NVision simulations.

This module replaces the generator/noise definitions that used to live in
``nvision.sim.cases`` so they can be imported without pulling in the
``RunCase`` / ``RunGroup`` machinery.
"""

from __future__ import annotations

from nvision.models.noise import (
    CompositeNoise,
    CompositeOverFrequencyNoise,
    CompositeOverProbeNoise,
)
from nvision.noises import (
    OverFrequencyGaussianNoise,
    OverFrequencyOutlierSpikes,
    OverFrequencyPoissonNoise,
    OverProbeDriftNoise,
)

from .gen.nv_center_generator import NVCenterCoreGenerator

# Single source for ``nvision run`` / ``nvision render`` defaults.
DEFAULT_LOC_MAX_STEPS = 1500


# Generators: NV Center variants
# Now using core architecture with TrueSignal and explicit SignalModels
def generators_basic() -> list[tuple[str, object]]:
    return [
        # NV Center generators - different variants
        (
            "NVCenter-lorentzian",
            NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian"),
        ),
        (
            "NVCenter-voigt",
            NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="voigt"),
        ),
    ]


def generators_narrow() -> list[tuple[str, object]]:
    """Narrow-domain generators for bayesian_only group.

    The domain is 60 MHz wide (2.87–2.93 GHz) and the center frequency
    is constrained to the middle 10 % so the signal occupies ~10 %
    of the smaller range, making sweeping unnecessary.
    """
    return [
        (
            "NVCenter-lorentzian-narrow",
            NVCenterCoreGenerator(
                x_min=2.87e9, x_max=2.93e9, variant="lorentzian", center_freq_fraction=0.1
            ),
        ),
        (
            "NVCenter-voigt-narrow",
            NVCenterCoreGenerator(
                x_min=2.87e9, x_max=2.93e9, variant="voigt", center_freq_fraction=0.1
            ),
        ),
    ]


# Noise tiers: start simple and evolve


def noises_none() -> list[tuple[str, CompositeNoise | None]]:
    return [("NoNoise", None)]


def noises_single_each() -> list[tuple[str, CompositeNoise | None]]:
    gauss_noise = 0.01
    poisson_noise = 3000.0
    over_probe_noise = 0.001
    return [
        (
            f"Gauss({gauss_noise})",
            CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(gauss_noise)])),
        ),
        (
            f"Poisson({poisson_noise})",
            CompositeNoise(
                over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyPoissonNoise(scale=poisson_noise)])
            ),
        ),
        (
            f"OverProbeDrift({over_probe_noise})",
            CompositeNoise(over_probe_noise=CompositeOverProbeNoise([OverProbeDriftNoise(over_probe_noise)])),
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
