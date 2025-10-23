from __future__ import annotations

import random
from dataclasses import dataclass

import polars as pl

from ..core import DataBatch


@dataclass
class GaussianNoise:
    """Additive Gaussian noise with standard deviation sigma.

    If clip_min is set, values are clamped to [clip_min, clip_max] (clip_max optional).
    """

    sigma: float = 0.05
    clip_min: float | None = None
    clip_max: float | None = None

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        n = data.df.height
        if n == 0:
            return data
        noise_series = pl.Series([rng.gauss(0.0, self.sigma) for _ in range(n)], dtype=pl.Float64)
        expr = pl.col("signal_values") + pl.lit(noise_series)
        if self.clip_min is not None or self.clip_max is not None:
            expr = expr.clip(
                self.clip_min if self.clip_min is not None else None,
                self.clip_max if self.clip_max is not None else None,
            )
        df = data.df.with_columns(signal_values=expr)
        return DataBatch.from_frame(df, meta=dict(data.meta))
