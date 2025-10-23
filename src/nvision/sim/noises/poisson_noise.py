from __future__ import annotations

import math
import random
from dataclasses import dataclass

import polars as pl

from ..core import DataBatch


@dataclass
class PoissonNoise:
    """Poisson counting noise applied to positive signals interpreted as expected counts.

    Scales input by a scale, draws from Poisson, then rescales back.
    """

    scale: float = 100.0

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        out: list[float] = []
        for v in data.signal_values:
            lam = max(v * self.scale, 0.0)
            l_threshold = math.exp(-lam)
            k = 0
            p = 1.0
            while p > l_threshold:
                k += 1
                p *= rng.random()
                if k > 100000:
                    break
            k -= 1
            out.append(max(0.0, k / self.scale))
        df = data.df.with_columns(signal_values=pl.Series(out, dtype=pl.Float64))
        return DataBatch.from_frame(df, meta=dict(data.meta))
