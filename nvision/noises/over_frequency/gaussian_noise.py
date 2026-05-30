import random
from dataclasses import dataclass

import polars as pl

from nvision.sim import OverFrequencyNoise
from nvision.sim.batch import DataBatch
from nvision.spectra.noise_model import NoiseSignalModel


@dataclass
class OverFrequencyGaussianNoise(OverFrequencyNoise):
    """Additive Gaussian noise applied over the frequency axis.

    The noise is drawn from N(0, std) and added to the signal, which has a
    baseline of 1.0. This means the signal remains centred at 1 and `std`
    directly controls the spread: ~68 % of samples fall within ±1 std of the
    mean, ~95 % within ±2 std, etc.

    Args:
        std: Standard deviation of the Gaussian noise (in signal units).
            Pass the std directly so you can reason about probabilities:
            e.g. std=0.05 means ≈95 % of noise samples lie within ±0.10.
        clip_min: If set, values are clamped to [clip_min, clip_max].
        clip_max: Optional upper clamp bound.
    """

    std: float = 0.015
    clip_min: float | None = None
    clip_max: float | None = None

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        n = data.df.height
        if n == 0:
            return data
        sigma = max(self.std, 0.0)
        noise = pl.Series("noise", [rng.gauss(0.0, sigma) for _ in range(n)], dtype=pl.Float64)
        noisy = data.df.get_column("signal_values") + noise
        if self.clip_min is not None or self.clip_max is not None:
            noisy = noisy.clip(self.clip_min, self.clip_max)
        df = data.df.with_columns(noisy.alias("signal_values"))
        return DataBatch.from_frame(df, meta=dict(data.meta))

    def noise_std(self) -> float:
        return max(self.std, 0.0)

    def to_noise_signal_model(self) -> NoiseSignalModel:
        """Create the Bayesian counterpart with latent parameter priors."""
        from nvision.spectra.noise_model import GaussianNoiseSignalModel

        # Uncertainty window around the nominal sigma (±5x)
        sigma = self.noise_std()
        prior_lo = sigma * 0.2
        prior_hi = sigma * 5.0
        return GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (prior_lo, prior_hi)})
