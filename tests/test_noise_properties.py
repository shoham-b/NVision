from __future__ import annotations

import random

from hypothesis import given
from hypothesis import strategies as st

from nvision.sim import CompositeNoise, DataBatch, GaussianNoise, PoissonNoise


@given(st.lists(st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False), min_size=0, max_size=200))
def test_poisson_noise_properties(values):
    t = list(range(len(values)))
    data = DataBatch(time_points=t, signal_values=values, meta={})
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    p = PoissonNoise(scale=20.0)
    out1 = p.apply(data, rng1)
    out2 = p.apply(data, rng2)
    assert len(out1.signal_values) == len(values)
    assert len(out2.signal_values) == len(values)
    assert all(v >= 0 for v in out1.signal_values)
    # Deterministic with same seed/state
    assert out1.signal_values == out2.signal_values


@given(st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=0, max_size=200))
def test_composite_noise_preserves_length(values):
    t = list(range(len(values)))
    data = DataBatch(time_points=t, signal_values=values, meta={})
    rng = random.Random(999)
    comp = CompositeNoise([GaussianNoise(0.1), GaussianNoise(0.2)])
    out = comp.apply(data, rng)
    assert len(out.signal_values) == len(values)
