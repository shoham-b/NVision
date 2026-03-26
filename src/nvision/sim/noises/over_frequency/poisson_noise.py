from __future__ import annotations

import math
import random
from dataclasses import dataclass

import polars as pl

from nvision.sim import OverFrequencyNoise
from nvision.sim.batch import DataBatch


@dataclass
class OverFrequencyPoissonNoise(OverFrequencyNoise):
    """Poisson counting noise applied to positive signals interpreted as expected counts.

    Scales input by a scale, draws from Poisson, then rescales back.
    """

    scale: float = 100.0

    @staticmethod
    def _sample_poisson(lam: float, rng: random.Random) -> int:
        """Draw a Poisson sample using a numerically stable hybrid approach.

        Knuth's exact algorithm is used for small rates; for larger rates we use
        a Gaussian approximation to avoid ``exp(-lam)`` underflow.
        """
        if lam <= 0.0:
            return 0
        if lam < 30.0:
            l_threshold = math.exp(-lam)
            k = 0
            p = 1.0
            while p > l_threshold:
                k += 1
                p *= rng.random()
            return k - 1

        # For large lambda, Poisson is well-approximated by Normal(lam, lam).
        sample = round(rng.normalvariate(lam, math.sqrt(lam)))
        return max(0, sample)

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        true_amplitude = data.meta.get("amplitude", 1.0)
        current_scale = self.scale * true_amplitude

        out: list[float] = []
        for v in data.signal_values:
            lam = max(v * current_scale, 0.0)
            k = self._sample_poisson(lam, rng)
            out.append(max(0.0, k / current_scale))
        df = data.df.with_columns(signal_values=pl.Series(out, dtype=pl.Float64))
        return DataBatch.from_frame(df, meta=dict(data.meta))

    def noise_std(self) -> float:
        return 1 / self.scale
