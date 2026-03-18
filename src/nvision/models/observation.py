"""Observation dataclass for measurement data."""

from dataclasses import dataclass


@dataclass
class Observation:
    """Single measurement observation.

    Attributes
    ----------
    x : float
        Position where measurement was taken
    signal_value : float
        Measured signal value at x
    """

    x: float
    signal_value: float
