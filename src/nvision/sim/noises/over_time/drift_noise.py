from __future__ import annotations

import random
from dataclasses import dataclass

import polars as pl

from ...core import DataBatch


@dataclass
class OverTimeDriftNoise:
    """Adds a slow linear drift across the signal."""

    drift_per_unit: float = 0.05

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        n = data.df.height
        if n == 0:
            return data
        start = -0.5 * self.drift_per_unit
        step = self.drift_per_unit / max(n - 1, 1)
        idx = pl.arange(0, n, eager=True)
        drift = pl.lit(start) + idx.cast(pl.Float64) * step
        df = data.df.with_columns(signal_values=pl.col("signal_values") + drift)
        return DataBatch.from_frame(df, meta=dict(data.meta))
