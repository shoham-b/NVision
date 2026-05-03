from __future__ import annotations

import random
from dataclasses import dataclass

import polars as pl

from nvision.sim import OverFrequencyNoise
from nvision.sim.batch import DataBatch
from nvision.spectra.noise_model import NoiseSignalModel


@dataclass
class OverFrequencyGaussianNoise(OverFrequencyNoise):
    """Additive Gaussian noise with standard deviation sigma applied to frequency axis.

    If clip_min is set, values are clamped to [clip_min, clip_max] (clip_max optional).
    """

    sigma: float = 0.015
    clip_min: float | None = None
    clip_max: float | None = None

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        n = data.df.height
        if n == 0:
            return data
        noise = pl.Series("noise", [rng.gauss(0.0, self.sigma) for _ in range(n)], dtype=pl.Float64)
        noisy = data.df.get_column("signal_values") + noise
        if self.clip_min is not None or self.clip_max is not None:
            noisy = noisy.clip(self.clip_min, self.clip_max)
        df = data.df.with_columns(noisy.alias("signal_values"))
        return DataBatch.from_frame(df, meta=dict(data.meta))

    def noise_std(self) -> float:
        return self.sigma

    def to_noise_signal_model(self) -> NoiseSignalModel:
        """Create the Bayesian counterpart with latent parameter priors."""
        from nvision.spectra.noise_model import GaussianNoiseSignalModel

        # Uncertainty window around the nominal sigma (±5x)
        prior_lo = self.sigma * 0.2
        prior_hi = self.sigma * 5.0
        return GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (prior_lo, prior_hi)})
