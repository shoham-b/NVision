from __future__ import annotations

import math
import random
from dataclasses import dataclass

import polars as pl

from .core import DataBatch


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
        # Generate Gaussian noise vector in Python (fast) and apply via Polars ops
        noise_series = pl.Series([rng.gauss(0.0, self.sigma) for _ in range(n)], dtype=pl.Float64)
        expr = pl.col("signal_values") + pl.lit(noise_series)
        if self.clip_min is not None or self.clip_max is not None:
            expr = expr.clip(
                self.clip_min if self.clip_min is not None else None,
                self.clip_max if self.clip_max is not None else None,
            )
        df = data.df.with_columns(signal_values=expr)
        return DataBatch.from_frame(df, meta=dict(data.meta))


@dataclass
class PoissonNoise:
    """Poisson counting noise applied to positive signals interpreted as expected counts.

    Scales input by a scale, draws from Poisson, then rescales back.
    """

    scale: float = 100.0  # larger scale -> closer to Gaussian

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        # Poisson sampling is not provided by Polars; sample in Python, then
        # assemble a Series and replace via Polars to keep columnar flow.
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


@dataclass
class DriftNoise:
    """Adds a slow linear drift across the signal."""

    drift_per_unit: float = 0.05

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        n = data.df.height
        if n == 0:
            return data
        start = -0.5 * self.drift_per_unit
        step = self.drift_per_unit / max(n - 1, 1)
        idx = pl.arange(0, n, eager=True)  # integer series
        drift = pl.lit(start) + idx.cast(pl.Float64) * step
        df = data.df.with_columns(signal_values=pl.col("signal_values") + drift)
        return DataBatch.from_frame(df, meta=dict(data.meta))


@dataclass
class OutlierSpikes:
    """Injects occasional large spikes or dips into the signal."""

    probability: float = 0.02
    magnitude: float = 1.0

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        n = data.df.height
        if n == 0:
            return data
        # Precompute random mask and spike magnitudes/directions
        mask = [rng.random() < self.probability for _ in range(n)]
        dirs = [(-1.0 if rng.random() < 0.5 else 1.0) for _ in range(n)]
        mags = [self.magnitude * (0.5 + rng.random()) for _ in range(n)]
        spikes = [(dirs[i] * mags[i]) if mask[i] else 0.0 for i in range(n)]
        df = data.df.with_columns(
            signal_values=pl.col("signal_values") + pl.Series(spikes, dtype=pl.Float64),
        )
        return DataBatch.from_frame(df, meta=dict(data.meta))
