from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..core import DataBatch


@dataclass
class PoissonNoise:
    """Poisson counting noise applied to positive signals interpreted as expected counts.

    Scales input by a scale, draws from Poisson, then rescales back.
    """

    scale: float = 100.0

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        # Create a numpy-compatible random number generator from the base random.Random
        # to ensure reproducibility while leveraging numpy's performance.
        np_rng = np.random.default_rng(rng.integers(2**32 - 1))

        # Use polars expressions for the entire transformation pipeline.
        # This is more idiomatic and allows polars' query optimizer to work.
        df = data.df.with_columns(
            pl.col("signal_values")
            # 1. Scale signal to get expected counts (lambda), ensuring it's non-negative.
            .mul(self.scale)
            .clip_lower(0.0)
            # 2. Apply the Poisson function to each element.
            #    `map_elements` is used to apply a Python function row-wise.
            .map_elements(lambda lam: np_rng.poisson(lam), return_dtype=pl.Int64)
            # 3. Rescale the counts back to the original signal's scale.
            .truediv(self.scale)
            .alias("signal_values")
        )
        return DataBatch.from_frame(df, meta=dict(data.meta))
