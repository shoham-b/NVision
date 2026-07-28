from __future__ import annotations

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from nvision import CompositeOverFrequencyNoise, DataBatch, OverFrequencyGaussianNoise


@settings(deadline=None)
@given(
    st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=200,
    ),
)
def test_composite_noise_preserves_length(values):
    t = list(range(len(values)))
    data = DataBatch(x=t, signal_values=values, meta={})
    rng = random.Random(999)
    comp = CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.1), OverFrequencyGaussianNoise(0.2)])
    out = comp.apply(data, rng)
    assert len(out.signal_values) == len(values)
