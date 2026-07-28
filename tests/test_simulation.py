from __future__ import annotations

import random

from nvision import (
    CompositeOverFrequencyNoise,
    DataBatch,
    OverFrequencyGaussianNoise,
)


def test_noise_composition_deterministic_and_length():
    y = [0.1 * i for i in range(50)]
    t = list(range(len(y)))
    data = DataBatch(x=t, signal_values=y, meta={})
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    noise = CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.1)])
    d1 = noise.apply(data, rng1)
    d2 = noise.apply(data, rng2)
    assert len(d1.signal_values) == len(y)
    assert len(d2.signal_values) == len(y)
    assert d1.signal_values == d2.signal_values  # same seed -> same result
