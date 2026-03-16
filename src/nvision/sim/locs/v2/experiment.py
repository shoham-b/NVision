"""Experiment definition for stateless locator architecture."""

import random
from dataclasses import dataclass
from typing import Optional

from nvision.sim.core import CompositeNoise
from nvision.sim.scan_batch import ScanBatch


@dataclass
class Experiment:
    """Experimental setup that generates noisy measurements.

    Attributes
    ----------
    scan : ScanBatch
        The underlying signal to measure
    noise : CompositeNoise | None
        Noise model to apply to measurements, or None for noiseless
    """

    scan: ScanBatch
    noise: Optional[CompositeNoise]

    def measure(self, x: float, rng: random.Random) -> float:
        """Take a measurement at position x.

        Parameters
        ----------
        x : float
            Position to measure at
        rng : random.Random
            Random number generator for reproducible noise

        Returns
        -------
        float
            Measured signal value (with noise if configured)
        """
        signal = self.scan.signal(x)

        if self.noise is not None:
            signal = self.noise.apply(signal, rng)

        return signal
