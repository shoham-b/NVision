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

    scale: float = 1000.0

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
        # For Poisson(λ) noise with λ = v·scale, std(k/scale) = √v/√scale.
        # Evaluated at background level v=1.0 → std = 1/√scale.
        return 1.0 / math.sqrt(max(self.scale, 1e-12))

    def max_noise_deviation(self, n_samples: int = 20) -> float:
        # EVT maximum for n i.i.d. Poisson samples at signal level v=1.0.
        # std = 1/√scale; EVT: std × √(2·log(n)).
        std = self.noise_std()
        if std <= 0 or n_samples < 2:
            return 0.0
        return std * math.sqrt(2.0 * math.log(max(n_samples, 2)))
