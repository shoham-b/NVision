from __future__ import annotations

import random
from dataclasses import dataclass

import polars as pl

from ..core import DataBatch


@dataclass
class OutlierSpikes:
    """Injects occasional large spikes or dips into the signal."""

    probability: float = 0.02
    magnitude: float = 1.0

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        n = data.df.height
        if n == 0:
            return data
        mask = [rng.random() < self.probability for _ in range(n)]
        dirs = [(-1.0 if rng.random() < 0.5 else 1.0) for _ in range(n)]
        mags = [self.magnitude * (0.5 + rng.random()) for _ in range(n)]
        spikes = [(dirs[i] * mags[i]) if mask[i] else 0.0 for i in range(n)]
        df = data.df.with_columns(
            signal_values=pl.col("signal_values") + pl.Series(spikes, dtype=pl.Float64),
        )
        return DataBatch.from_frame(df, meta=dict(data.meta))
