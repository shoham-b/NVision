"""Preset generators, noises, and constants for NVision simulations.

This module replaces the generator/noise definitions that used to live in
``nvision.sim.cases`` so they can be imported without pulling in the
``RunCase`` / ``RunGroup`` machinery.
"""

from __future__ import annotations

from nvision.models.noise import (
    CompositeNoise,
    CompositeOverFrequencyNoise,
)
from nvision.noises import (
    OverFrequencyGaussianNoise,
    OverFrequencyPoissonNoise,
)
from nvision.sim.defaults import (
    NVISION_DEFAULT_LOC_MAX_STEPS,
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


# Noise tiers: start simple and evolve


def noises_none() -> list[tuple[str, CompositeNoise | None]]:
    # ARCHIVED: Only Poisson and Gauss are used now.
    return []


def noises_single_each() -> list[tuple[str, CompositeNoise | None]]:
    import numpy as np

    from nvision.sim.defaults import (
        NVISION_NOISE_GAUSS_STEPS,
        NVISION_NOISE_MAX_GAUSS,
    )

    noises = []
    if NVISION_NOISE_GAUSS_STEPS > 1:
        sigmas = np.linspace(0.0, NVISION_NOISE_MAX_GAUSS, NVISION_NOISE_GAUSS_STEPS)
    elif NVISION_NOISE_GAUSS_STEPS == 1:
        sigmas = [NVISION_NOISE_MAX_GAUSS]
    else:
        sigmas = []

    for sigma in sigmas:
        sigma_val = float(round(sigma, 4))
        noises.append(
            (
                f"Gauss({sigma_val})",
                CompositeNoise(
                    over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(sigma_val)])
                ),
            )
        )

    noises.append(
        (
            f"Poisson({NVISION_NOISE_POISSON})",
            CompositeNoise(
                over_frequency_noise=CompositeOverFrequencyNoise(
                    [OverFrequencyPoissonNoise(scale=NVISION_NOISE_POISSON)]
                )
            ),
        )
    )
    return noises


def noises_complex() -> list[tuple[str, CompositeNoise | None]]:
    # ARCHIVED: Heavy noise is archived.
    return []
