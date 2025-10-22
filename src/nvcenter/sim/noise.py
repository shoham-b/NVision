from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence

import polars as pl

from .core import DataBatch, NoiseModel
from nvcenter.mathutils import clamp


@dataclass
class GaussianNoise:
    """Additive Gaussian noise with standard deviation sigma.

    If clip_min is set, values are clamped to [clip_min, clip_max] (clip_max optional).
    """

    sigma: float = 0.05
    clip_min: float | None = None
    clip_max: float | None = None

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        # Add Gaussian noise per-sample; clip if requested
        out: List[float] = []
        for v in data.signal_values:
            noisy = v + rng.gauss(0.0, self.sigma)
            if self.clip_min is not None:
                cmax = self.clip_max if self.clip_max is not None else float("inf")
                noisy = clamp(noisy, self.clip_min, cmax)
            elif self.clip_max is not None:
                noisy = min(noisy, self.clip_max)
            out.append(noisy)
        return data.with_y(out)


@dataclass
class PoissonNoise:
    """Poisson counting noise applied to positive signals interpreted as expected counts.

    Scales input by a scale, draws from Poisson, then rescales back.
    """

    scale: float = 100.0  # larger scale -> closer to Gaussian

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        out: List[float] = []
        for v in data.signal_values:
            lam = max(v * self.scale, 0.0)
            # Sample using Knuth's algorithm for Poisson
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= rng.random()
                # Guard against extreme loops for huge lambda by breaking
                if k > 100000:
                    break
            k -= 1
            out.append(k / self.scale)
        return data.with_y(out)


@dataclass
class DriftNoise:
    """Adds a slow linear drift across the signal."""

    drift_per_unit: float = 0.05

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        n = len(data.signal_values)
        if n == 0:
            return data
        start = -0.5 * self.drift_per_unit
        step = self.drift_per_unit / max(n - 1, 1)
        out = [v + (start + i * step) for i, v in enumerate(data.signal_values)]
        return data.with_y(out)


@dataclass
class OutlierSpikes:
    """Injects occasional large spikes or dips into the signal."""

    probability: float = 0.02
    magnitude: float = 1.0

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        out: List[float] = []
        for v in data.signal_values:
            if rng.random() < self.probability:
                direction = -1.0 if rng.random() < 0.5 else 1.0
                out.append(v + direction * (self.magnitude * (0.5 + rng.random())))
            else:
                out.append(v)
        return data.with_y(out)
