import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nvision import normalized_sum


def test_normalized_sum_basic():
    assert normalized_sum([0.0, 0.5, 1.0]) == pytest.approx(1.5)


def test_normalized_sum_empty():
    assert normalized_sum([]) == 0.0


def test_normalized_sum_constant_values():
    assert normalized_sum([5.0, 5.0, 5.0]) == 0.0


@given(st.lists(st.floats(allow_infinity=False, allow_nan=False), min_size=0, max_size=100))
def test_normalized_sum_never_nan(values):
    # Should never produce NaN even with edge cases
    result = normalized_sum(values)
    assert not math.isnan(result)
