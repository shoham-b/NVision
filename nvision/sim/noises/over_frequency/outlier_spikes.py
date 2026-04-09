from __future__ import annotations

import random
from dataclasses import dataclass

import polars as pl

from nvision.sim import OverFrequencyNoise
from nvision.sim.batch import DataBatch


@dataclass
class OverFrequencyOutlierSpikes(OverFrequencyNoise):
    """Injects occasional large spikes or dips into the sampled frequency response."""

    probability: float = 0.02
    magnitude: float = 1.0

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        n = data.df.height
        if n == 0:
            return data
        spikes = [
            ((-1.0 if rng.random() < 0.5 else 1.0) * self.magnitude * (0.5 + rng.random()))
            if rng.random() < self.probability
            else 0.0
            for _ in range(n)
        ]
        noisy = data.df.get_column("signal_values") + pl.Series("spikes", spikes, dtype=pl.Float64)
        df = data.df.with_columns(noisy.alias("signal_values"))
        return DataBatch.from_frame(df, meta=dict(data.meta))
