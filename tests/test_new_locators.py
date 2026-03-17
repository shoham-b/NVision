"""Test v2 locators and basic behaviors."""

from __future__ import annotations

import math
import random

import polars as pl
import pytest

from nvision.sim.locs.nv_center.sweep_locator_v2 import NVCenterSweepLocatorV2
from nvision.sim.locs.v2.simple import GridMaxLocator


def _empty_history() -> pl.DataFrame:
    """Create an empty v2 history DataFrame."""
    return pl.DataFrame(
        {
            "x": pl.Series("x", [], dtype=pl.Float64),
            "signal_value": pl.Series("signal_value", [], dtype=pl.Float64),
        }
    )


def _append_measurement(history: pl.DataFrame, x: float, y: float) -> pl.DataFrame:
    return pl.concat([history, pl.DataFrame({"x": [float(x)], "signal_value": [float(y)]})])


def test_one_peak_locators_basic():
    """Test that the v2 grid locator operates and returns an estimate."""
    rng = random.Random(123)

    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)

    locator = GridMaxLocator(n_points=11)
    history = _empty_history()

    for _i in range(3):
        x = locator.next(history)
        assert 0.0 <= x <= 1.0
        y = signal(x) + rng.gauss(0, 0.05)
        history = _append_measurement(history, x, y)

    assert isinstance(locator.done(history), bool)
    result = locator.result(history)
    assert isinstance(result, dict)
    assert "peak_x" in result


def test_nv_center_locators_basic():
    """Test NV center v2 sweep locator produces proposals and a peak estimate."""
    rng = random.Random(456)

    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.3) / 0.08) ** 2)

    locator = NVCenterSweepLocatorV2(coarse_points=15, refine_points=5)

    history = _empty_history()
    for _i in range(3):
        x = locator.next(history)
        assert 0.0 <= x <= 1.0
        y = signal(x) + rng.gauss(0, 0.03)
        history = _append_measurement(history, x, y)

    result = locator.result(history)
    assert "peak_x" in result


def test_uncertainty_calculation():
    """Basic sanity check that v2 locator runs on arbitrary signal."""
    rng = random.Random(789)

    def signal(x: float) -> float:
        return 1.0 + 0.5 * math.sin(2 * math.pi * x)

    locator = GridMaxLocator(n_points=21)
    history = _empty_history()
    for _i in range(5):
        x = locator.next(history)
        y_noisy = signal(x) + rng.gauss(0, 0.1)
        history = _append_measurement(history, x, y_noisy)

    assert history.height > 0
    assert "peak_x" in locator.result(history)


def test_locator_comparison():
    """Test that different locators produce different behaviors."""
    rng = random.Random(999)

    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.4) / 0.05) ** 2)

    # Use two different strategies that are guaranteed to propose different grids.
    grid_locator_a = GridMaxLocator(n_points=15)
    grid_locator_b = GridMaxLocator(n_points=11)

    hist_a = _empty_history()
    hist_b = _empty_history()

    for _i in range(6):
        xa = grid_locator_a.next(hist_a)
        ya = signal(xa) + rng.gauss(0, 0.02)
        hist_a = _append_measurement(hist_a, xa, ya)

        xb = grid_locator_b.next(hist_b)
        yb = signal(xb) + rng.gauss(0, 0.02)
        hist_b = _append_measurement(hist_b, xb, yb)

    res_a = grid_locator_a.result(hist_a)
    res_b = grid_locator_b.result(hist_b)

    assert 0.0 <= res_a["peak_x"] <= 1.0
    assert 0.0 <= res_b["peak_x"] <= 1.0
    assert res_a["peak_x"] != pytest.approx(res_b["peak_x"], abs=1e-6)
