"""Observation dataclass for measurement data."""

from dataclasses import dataclass, field


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
    """

    x: float
    signal_value: float
    noise_std: float = field(default=0.05)
