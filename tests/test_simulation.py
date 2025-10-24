from __future__ import annotations

import random

from nvision.sim import DataBatch
from nvision.sim.core import CompositeNoise
from nvision.sim.noises import (
    OverTimeDriftNoise,
    OverVoltageGaussianNoise,
    OverVoltageOutlierSpikes,
    OverVoltagePoissonNoise,
)


def test_noise_composition_deterministic_and_length():
    y = [0.1 * i for i in range(50)]
    t = list(range(len(y)))
    data = DataBatch(time_points=t, signal_values=y, meta={})
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    noise = CompositeNoise(
        [
            OverVoltageGaussianNoise(0.1),
            OverTimeDriftNoise(0.05),
            OverVoltageOutlierSpikes(0.1, 0.5),
        ]
    )
    d1 = noise.apply(data, rng1)
    d2 = noise.apply(data, rng2)
    assert len(d1.signal_values) == len(y)
    assert len(d2.signal_values) == len(y)
    assert d1.signal_values == d2.signal_values  # same seed -> same result


def test_poisson_noise_non_negative():
    y = [0.0, 0.1, 1.0, 2.5]
    t = list(range(len(y)))
    data = DataBatch(time_points=t, signal_values=y, meta={})
    rng = random.Random(123)
    p = OverVoltagePoissonNoise(scale=50.0)
    out = p.apply(data, rng)
    assert all(v >= 0 for v in out.signal_values)
