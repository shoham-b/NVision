"""Experiment setup for core architecture.

Replaces the legacy ScanBatch + Experiment pattern with a native TrueSignal approach.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from nvision.models.noise import CompositeNoise
from nvision.models.observation import Observation
from nvision.signal.signal import TrueSignal


@dataclass
class CoreExperiment:
    """Experimental setup with TrueSignal and noise.

    This replaces the legacy Experiment+ScanBatch pattern with native core types.

    Attributes
    ----------
    true_signal : TrueSignal
        Ground truth signal to measure
    noise : CompositeNoise | None
        Noise model to apply to measurements
    x_min : float
        Physical domain minimum
    x_max : float
        Physical domain maximum
    """

    true_signal: TrueSignal
    noise: CompositeNoise | None
    x_min: float
    x_max: float

    def measure(self, x_normalized: float, rng: random.Random) -> Observation:
        """Take a measurement at normalized position.

        Parameters
        ----------
        x_normalized : float
            Position in [0, 1] normalized space
        rng : random.Random
            Random number generator for noise

        Returns
        -------
        Observation
            Measurement with noise applied
        """
        # Denormalize to physical domain
        width = self.x_max - self.x_min
        x_physical = self.x_min + x_normalized * width

        # Get true signal value
        signal_value = self.true_signal(x_physical)

        # Apply noise components if configured
        if self.noise is not None:
            if self.noise.over_frequency_noise is not None:
                from nvision.sim.batch import DataBatch

                batch = DataBatch(x=[x_physical], signal_values=[signal_value])
                noisy_batch = self.noise.over_frequency_noise.apply(batch, rng)
                signal_value = float(noisy_batch.df["signal_values"][0])
            if self.noise.over_probe_noise is not None:
                signal_value = self.noise.over_probe_noise.apply(signal_value, rng, None)

        # Return observation in normalized space
        return Observation(x=x_normalized, signal_value=signal_value)

    @property
    def signal(self):
        """Physical-domain signal callable — for viz compatibility."""
        return self.true_signal

    @property
    def truth_positions(self) -> list[float]:
        """Ground truth peak positions extracted from TrueSignal parameters."""
        return [p.value for p in self.true_signal.parameters if "frequency" in p.name or "position" in p.name]

    def normalize_x(self, x_physical: float) -> float:
        """Convert physical x to normalized [0, 1]."""
        return (x_physical - self.x_min) / (self.x_max - self.x_min)

    def denormalize_x(self, x_normalized: float) -> float:
        """Convert normalized x to physical domain."""
        return self.x_min + x_normalized * (self.x_max - self.x_min)
