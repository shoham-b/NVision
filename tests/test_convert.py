from __future__ import annotations

import pytest

from nvision.runner.convert import denormalize_x


def test_denormalize_x() -> None:
    # Boundary cases
    assert denormalize_x(0.0, 10.0, 20.0) == pytest.approx(10.0)
    assert denormalize_x(1.0, 10.0, 20.0) == pytest.approx(20.0)

    # Midpoint
    assert denormalize_x(0.5, 10.0, 20.0) == pytest.approx(15.0)

    # Negative bounds
    assert denormalize_x(0.0, -10.0, 10.0) == pytest.approx(-10.0)
    assert denormalize_x(1.0, -10.0, 10.0) == pytest.approx(10.0)
    assert denormalize_x(0.5, -10.0, 10.0) == pytest.approx(0.0)

    # Fractional
    assert denormalize_x(0.25, 0.0, 100.0) == pytest.approx(25.0)
