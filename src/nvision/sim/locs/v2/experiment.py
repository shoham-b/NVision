"""Experiment definition for stateless locator architecture."""

import random
from dataclasses import dataclass
from typing import Optional

from nvision.sim.core import CompositeNoise
from nvision.sim.scan_batch import ScanBatch as ScanBatch


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

    def measure(self, x: float, rng: random.Random, locator: object | None = None) -> float:
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
        # v2 locators operate in normalized x-space [0, 1].
        width = self.scan.x_max - self.scan.x_min
        x_phys = self.scan.x_min + x * width
        signal = self.scan.signal(x_phys)

        if self.noise is not None and self.noise.over_probe_noise is not None:
            # Over-probe noise models may optionally inspect the locator's current estimates.
            signal = self.noise.over_probe_noise.apply(signal, rng, locator)  # type: ignore[arg-type]

        return signal
