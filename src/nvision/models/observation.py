"""Observation dataclass for a single measurement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nvision.models.measurement_noise import DEFAULT_MEASUREMENT_NOISE_STD

__all__ = [
    "DEFAULT_MEASUREMENT_NOISE_STD",
    "Observation",
    "gaussian_likelihood_std",
]


@dataclass
class Observation:
    """Single measurement observation.

    Attributes
    ----------
    x : float
        Position where measurement was taken
    signal_value : float
        Measured signal value at x
    noise_std : float
        Known (or estimated) standard deviation of the measurement noise.
        Used by belief distributions for the likelihood function.
        Defaults to 0.05 (suitable for normalized [0, 1] signals with no noise).
    frequency_noise_model : tuple[dict[str, Any], ...] | None
        Optional structured description of over-frequency noise components used
        to generate this observation. When present, Bayesian updates can use a
        component-specific likelihood (e.g. Poisson counting) instead of the
        default Gaussian approximation.
    """

    x: float
    signal_value: float
    noise_std: float = field(default=DEFAULT_MEASUREMENT_NOISE_STD)
    frequency_noise_model: tuple[dict[str, Any], ...] | None = field(default=None)


def gaussian_likelihood_std(obs: Observation | None) -> float:
    """Sigma for the Gaussian likelihood and Fisher terms: ``obs.noise_std`` or the global default."""
    if obs is not None and obs.noise_std > 0:
        return float(obs.noise_std)
    return DEFAULT_MEASUREMENT_NOISE_STD
