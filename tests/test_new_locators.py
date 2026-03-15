"""Test category-specific locators with uncertainty calculation."""

from __future__ import annotations

import math
import random

import polars as pl
import pytest

from nvision.sim.locs.base import ScanBatch
from nvision.sim.locs.nv_center.sweep_locator import NVCenterSweepLocator
from nvision.sim.locs.one_peak import OnePeakGoldenLocator, OnePeakGridLocator


def _empty_history() -> pl.DataFrame:
    """Create an empty history DataFrame with the batched schema."""
    return pl.DataFrame(
        {
            "repeat_id": pl.Series("repeat_id", [], dtype=pl.Int64),
            "step": pl.Series("step", [], dtype=pl.Int64),
            "x": pl.Series("x", [], dtype=pl.Float64),
            "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
        }
    )


def _append_measurement(history: pl.DataFrame, x: float, y: float, repeat_id: int = 0) -> pl.DataFrame:
    """Append a measurement row with proper repeat_id and step columns."""
    step = history.filter(pl.col("repeat_id") == repeat_id).height
    new_row = pl.DataFrame(
        {
            "repeat_id": [repeat_id],
            "step": [step],
            "x": [x],
            "signal_values": [y],
        }
    )
    return pl.concat([history, new_row])


def test_one_peak_locators_basic():
    """Test that the one-peak locators operate and return uncertainty information."""
    rng = random.Random(123)

    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)

    scan = ScanBatch(
        x_min=0.0,
        x_max=1.0,
        truth_positions=[0.5],
        signal=signal,
        meta={"peak_width": 0.1},
    )

    grid_locator = OnePeakGridLocator(n_points=11)
    sweep_locator = OnePeakGoldenLocator(max_evals=10)

    repeats_df = pl.DataFrame({"repeat_id": [0], "active": [True]})

    for locator in (grid_locator, sweep_locator):
        current_history = _empty_history()
        for _i in range(3):
            next_proposal = locator.propose_next(current_history, repeats_df, scan)
            x = next_proposal.get_column("x")[0]
            assert 0.0 <= x <= 1.0
            y = signal(x) + rng.gauss(0, 0.05)
            current_history = _append_measurement(current_history, x, y)

            stop_df = locator.should_stop(current_history, repeats_df, scan)
            repeats_df = stop_df.select(["repeat_id", "stop"]).with_columns(pl.lit(True).alias("active"))

        final_stop = locator.should_stop(current_history, repeats_df, scan)
        assert isinstance(final_stop, pl.DataFrame)
        assert "stop" in final_stop.columns

        result = locator.finalize(current_history, repeats_df, scan)
        assert "n_peaks" in result.columns
        assert "x1_hat" in result.columns
        assert "uncert" in result.columns


def test_nv_center_locators_basic():
    """Test NV center locators return measurement counts and uncertainty."""
    rng = random.Random(456)

    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.3) / 0.08) ** 2)

    scan = ScanBatch(
        x_min=0.0,
        x_max=1.0,
        truth_positions=[0.3],
        signal=signal,
        meta={"peak_width": 0.08},
    )

    repeats_df = pl.DataFrame({"repeat_id": [0], "active": [True]})
    locator = NVCenterSweepLocator(coarse_points=15, refine_points=5)

    current_history = _empty_history()
    for _i in range(3):
        next_proposal = locator.propose_next(current_history, repeats_df, scan)
        x = next_proposal.get_column("x")[0]
        assert 0.0 <= x <= 1.0
        y = signal(x) + rng.gauss(0, 0.03)
        current_history = _append_measurement(current_history, x, y)

        stop_df = locator.should_stop(current_history, repeats_df, scan)
        repeats_df = stop_df.select(["repeat_id", "stop"]).with_columns(pl.lit(True).alias("active"))

    final_stop = locator.should_stop(current_history, repeats_df, scan)
    assert isinstance(final_stop, pl.DataFrame)
    assert "stop" in final_stop.columns

    result = locator.finalize(current_history, repeats_df, scan)
    assert "n_peaks" in result.columns
    assert "uncert" in result.columns


def test_uncertainty_calculation():
    """Test that uncertainty is properly calculated and used."""
    rng = random.Random(789)

    def signal(x: float) -> float:
        return 1.0 + 0.5 * math.sin(2 * math.pi * x)

    scan = ScanBatch(x_min=0.0, x_max=1.0, truth_positions=[0.25, 0.75], signal=signal, meta={})

    repeats_df = pl.DataFrame({"repeat_id": [0], "active": [True]})
    locator = OnePeakGoldenLocator(max_evals=12)

    current_history = _empty_history()
    for _i in range(5):
        next_proposal = locator.propose_next(current_history, repeats_df, scan)
        x = next_proposal.get_column("x")[0]
        y_noisy = signal(x) + rng.gauss(0, 0.1)
        current_history = _append_measurement(current_history, x, y_noisy)

        stop_df = locator.should_stop(current_history, repeats_df, scan)
        repeats_df = stop_df.select(["repeat_id", "stop"]).with_columns(pl.lit(True).alias("active"))

    assert current_history.height > 0

    result = locator.finalize(current_history, repeats_df, scan)
    assert "uncert" in result.columns


def test_locator_comparison():
    """Test that different locators produce different behaviors."""
    rng = random.Random(999)

    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.4) / 0.05) ** 2)

    scan = ScanBatch(x_min=0.0, x_max=1.0, truth_positions=[0.4], signal=signal, meta={})

    grid_repeats = pl.DataFrame({"repeat_id": [0], "active": [True]})
    golden_repeats = pl.DataFrame({"repeat_id": [0], "active": [True]})

    grid_locator = OnePeakGridLocator(n_points=15)
    golden_locator = OnePeakGoldenLocator(max_evals=12)

    grid_history = _empty_history()
    golden_history = _empty_history()

    for _i in range(6):
        # Grid locator step
        grid_proposal = grid_locator.propose_next(grid_history, grid_repeats, scan)
        x_grid = grid_proposal.get_column("x")[0]
        y_grid = signal(x_grid) + rng.gauss(0, 0.02)
        grid_history = _append_measurement(grid_history, x_grid, y_grid)
        grid_stop = grid_locator.should_stop(grid_history, grid_repeats, scan)
        grid_repeats = grid_stop.select(["repeat_id", "stop"]).with_columns(pl.lit(True).alias("active"))

        # Golden locator step
        golden_proposal = golden_locator.propose_next(golden_history, golden_repeats, scan)
        x_golden = golden_proposal.get_column("x")[0]
        y_golden = signal(x_golden) + rng.gauss(0, 0.02)
        golden_history = _append_measurement(golden_history, x_golden, y_golden)
        golden_stop = golden_locator.should_stop(golden_history, golden_repeats, scan)
        golden_repeats = golden_stop.select(["repeat_id", "stop"]).with_columns(pl.lit(True).alias("active"))

    # Get final results
    grid_result = grid_locator.finalize(grid_history, grid_repeats, scan)
    golden_result = golden_locator.finalize(golden_history, golden_repeats, scan)

    # Verify results — check DataFrame columns
    assert grid_result.get_column("n_peaks")[0] == 1.0
    assert golden_result.get_column("n_peaks")[0] == 1.0
    assert 0.0 <= grid_result.get_column("x1_hat")[0] <= 1.0
    assert 0.0 <= golden_result.get_column("x1_hat")[0] <= 1.0
    assert grid_result.get_column("x1_hat")[0] != pytest.approx(golden_result.get_column("x1_hat")[0], abs=1e-6)
