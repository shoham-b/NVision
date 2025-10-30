from __future__ import annotations

import random
from dataclasses import dataclass

from nvision.sim.core import CompositeNoise, DataBatch


@dataclass
class ScalarMeasure:
    """Applies existing CompositeNoise to a single scalar measurement."""

    noise: CompositeNoise | None = None

    def measure(self, x: float, y_clean: float, rng: random.Random) -> float:
        if self.noise is None:
            return y_clean
        db = DataBatch(time_points=[x], signal_values=[y_clean], meta={})
        noisy = self.noise.apply(db, rng)
        return float(noisy.signal_values[0])
