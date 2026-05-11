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
)
from nvision.noises.over_probe.drift_noise import OverProbeDriftNoise
from nvision.sim.defaults import (
    NVISION_DEFAULT_LOC_MAX_STEPS,
    NVISION_NOISE_GAUSS,
    NVISION_NOISE_OVER_PROBE,
    NVISION_NOISE_POISSON,
)

from .gen.nv_center_generator import NVCenterCoreGenerator

# Single source for ``nvision run`` / ``nvision render`` defaults.
DEFAULT_LOC_MAX_STEPS = NVISION_DEFAULT_LOC_MAX_STEPS


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
    """Narrow-parameter generators for bayesian_only group.

    The domain remains wide (2.6–3.1 GHz), but signal parameters (linewidth,
    splitting, etc.) are narrowed. Bayesian locators use Gaussian priors
    centered on the true values to localize the signal without requiring
    an initial sweep.
    """
    return [
        (
            "NVCenter-lorentzian-narrow",
            NVCenterCoreGenerator(
                x_min=2.6e9, x_max=3.1e9, variant="lorentzian", center_freq_fraction=0.1, narrow_signal=True
            ),
        ),
        (
            "NVCenter-voigt-narrow",
            NVCenterCoreGenerator(
                x_min=2.6e9, x_max=3.1e9, variant="voigt", center_freq_fraction=0.1, narrow_signal=True
            ),
        ),
    ]


# Noise tiers: start simple and evolve


def noises_none() -> list[tuple[str, CompositeNoise | None]]:
    # ARCHIVED: Only Poisson and Gauss are used now.
    return []


def noises_single_each() -> list[tuple[str, CompositeNoise | None]]:
    return [
        (
            f"Gauss({NVISION_NOISE_GAUSS})",
            CompositeNoise(
                over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(NVISION_NOISE_GAUSS)])
            ),
        ),
        (
            f"Poisson({NVISION_NOISE_POISSON})",
            CompositeNoise(
                over_frequency_noise=CompositeOverFrequencyNoise(
                    [OverFrequencyPoissonNoise(scale=NVISION_NOISE_POISSON)]
                )
            ),
        ),
        # ARCHIVED: OverProbeDrift is archived.
        # (
        #     f"OverProbeDrift({NVISION_NOISE_OVER_PROBE})",
        #     CompositeNoise(over_probe_noise=CompositeOverProbeNoise([OverProbeDriftNoise(NVISION_NOISE_OVER_PROBE)])),
        # ),
    ]


def noises_complex() -> list[tuple[str, CompositeNoise | None]]:
    # ARCHIVED: Heavy noise is archived.
    return []
