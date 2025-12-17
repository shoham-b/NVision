"""Tests for the batched version of the NVCenterSequentialBayesianLocator."""

from __future__ import annotations

import numpy as np
import polars as pl

from nvision.sim.locs.base import ScanBatch
from nvision.sim.locs.nv_center._bayesian_adapter import NVCenterSequentialBayesianLocatorBatched


def build_batched_locator(**overrides: object) -> NVCenterSequentialBayesianLocatorBatched:
    """Create a batched locator with test-friendly defaults."""
    config: dict[str, object] = {
        "max_evals": 12,
        "prior_bounds": (2.7e9, 3.0e9),
        "grid_resolution": 64,
        "n_monte_carlo": 40,
    }
    config.update(overrides)
    return NVCenterSequentialBayesianLocatorBatched(**config)


def build_scan() -> ScanBatch:
    """Create a test scan with a single peak."""
    center = 2.85e9
    width = 5e6

    def signal(x: float) -> float:
        z = (x - center) / width
        return float(1.0 - 0.1 * np.exp(-0.5 * z * z))

    return ScanBatch(
        x_min=2.7e9,
        x_max=3.0e9,
        signal=signal,
        meta={"inference": {"peaks": [{"x": center, "y": 1.0, "width": width}]}},
        truth_positions=[center],
    )


def test_batched_locator_initialization():
    """Test that the batched locator initializes correctly."""
    locator = build_batched_locator()
    assert isinstance(locator, NVCenterSequentialBayesianLocatorBatched)
    assert hasattr(locator, "propose_next")
    assert hasattr(locator, "should_stop")
    assert hasattr(locator, "finalize")


def test_propose_next_handles_empty_history():
    """Test that propose_next works with empty history."""
    locator = build_batched_locator()
    scan = build_scan()

    # Create empty history and active repeats
    history = pl.DataFrame(
        {
            "repeat_id": pl.Series(dtype=pl.Int64),
            "step": pl.Series(dtype=pl.Int64),
            "x": pl.Series(dtype=pl.Float64),
            "signal_values": pl.Series(dtype=pl.Float64),
        }
    )

    repeats = pl.DataFrame(
        {
            "repeat_id": [1, 2],
            "active": [True, True],
        }
    )

    proposals = locator.propose_next(history, repeats, scan)

    # Should return one proposal per active repeat
    assert isinstance(proposals, pl.DataFrame)
    assert "repeat_id" in proposals.columns
    assert "x" in proposals.columns
    assert len(proposals) == 2
    assert all(scan.x_min <= x <= scan.x_max for x in proposals["x"].to_list())


def test_should_stop_handles_batch():
    """Test that should_stop works with batched input."""
    # Ensure n_warmup is smaller than max_evals
    locator = build_batched_locator(max_evals=3, n_warmup=1)
    scan = build_scan()

    # Create history with two repeats, one with enough points to trigger stop
    history = pl.DataFrame(
        {
            "repeat_id": [1, 1, 2, 2, 2, 2],  # repeat 1 has 2 points, repeat 2 has 4 points
            "step": [0, 1, 0, 1, 2, 3],
            "x": [
                scan.x_min,
                scan.x_max,
                scan.x_min,
                scan.x_max,
                scan.x_min + 1e8,
                scan.x_max - 1e8,
            ],
            "signal_values": [1.0, 0.9, 1.0, 0.9, 0.95, 0.92],
        }
    )

    repeats = pl.DataFrame(
        {
            "repeat_id": [1, 2],
            "active": [True, True],
        }
    )

    stop_flags = locator.should_stop(history, repeats, scan)

    # Should return stop flags for each repeat
    assert isinstance(stop_flags, pl.DataFrame)
    assert "repeat_id" in stop_flags.columns
    assert "stop" in stop_flags.columns
    assert len(stop_flags) == 2  # One row per repeat in the repeats DataFrame

    # Get the stop flags as a dictionary for easier testing
    stop_dict = {row["repeat_id"]: row["stop"] for row in stop_flags.to_dicts()}

    # Debug output to help diagnose issues
    print("Stop flags:", stop_flags)
    print("History:", history)

    # The locator should stop for repeats that have reached max_evals
    # Since we're using max_evals=3:
    # - repeat 1 (2 points) should continue
    # - repeat 2 (4 points) should stop
    assert stop_dict[1] is False, "Repeat 1 should continue (2 points < max_evals=3)"
    assert stop_dict[2] is True, "Repeat 2 should stop (4 points > max_evals=3)"


def test_finalize_handles_batch():
    """Test that finalize works with batched input."""
    locator = build_batched_locator()
    scan = build_scan()

    # Create history with two repeats
    history = pl.DataFrame(
        {
            "repeat_id": [1, 1, 2, 2],
            "step": [0, 1, 0, 1],
            "x": [2.8e9, 2.9e9, 2.8e9, 2.9e9],
            "signal_values": [0.95, 0.92, 0.96, 0.91],
        }
    )

    repeats = pl.DataFrame(
        {
            "repeat_id": [1, 2],
            "active": [True, True],
        }
    )

    results = locator.finalize(history, repeats, scan)

    # Should return results for each repeat
    assert isinstance(results, pl.DataFrame)
    assert "repeat_id" in results.columns
    assert "x1_hat" in results.columns
    assert "uncert" in results.columns
    assert "measurements" in results.columns
    assert len(results) == 2

    # Check that results are within expected bounds
    for result in results.to_dicts():
        assert scan.x_min <= result["x1_hat"] <= scan.x_max
        assert result["uncert"] >= 0
        assert result["measurements"] == 2  # 2 measurements per repeat


def test_locator_handles_mixed_active_states():
    """Test that the locator correctly handles a mix of active and inactive repeats."""
    locator = build_batched_locator()
    scan = build_scan()

    # Create history with three repeats, one inactive
    history = pl.DataFrame(
        {
            "repeat_id": [1, 1, 2, 2, 3, 3],
            "step": [0, 1, 0, 1, 0, 1],
            "x": [2.8e9, 2.9e9, 2.8e9, 2.9e9, 2.8e9, 2.9e9],
            "signal_values": [0.95, 0.92, 0.96, 0.91, 0.94, 0.93],
        }
    )

    # Only repeats 1 and 3 are active
    repeats = pl.DataFrame(
        {
            "repeat_id": [1, 2, 3],
            "active": [True, False, True],
        }
    )

    # Test propose_next with mixed active states
    proposals = locator.propose_next(history, repeats, scan)
    assert len(proposals) == 2  # Only active repeats 1 and 3
    assert set(proposals["repeat_id"].to_list()) == {1, 3}

    # Test should_stop with mixed active states
    stop_flags = locator.should_stop(history, repeats, scan)
    # Should return results for all repeats, but only active ones will be processed
    assert len(stop_flags) == 3  # One row per repeat, regardless of active state
    assert set(stop_flags["repeat_id"].to_list()) == {1, 2, 3}

    # Test finalize with mixed active states
    results = locator.finalize(history, repeats, scan)
    # Should return results for all repeats, but only active ones will have meaningful results
    assert len(results) == 3  # One row per repeat, regardless of active state
    assert set(results["repeat_id"].to_list()) == {1, 2, 3}


def test_sampling_loop_generates_adaptive_points():
    """Integration test that verifies the batched locator adapts to the signal."""
    # This replaces the similar test in test_sequential_bayesian_locator.py
    locator = build_batched_locator(max_evals=10, n_warmup=2, grid_resolution=128, n_monte_carlo=100)
    scan = build_scan()

    # Initialize with proper schema
    schema = {
        "repeat_id": pl.Int64,
        "step": pl.Int64,
        "x": pl.Float64,
        "signal_values": pl.Float64,
    }
    history = pl.DataFrame(schema=schema)
    repeats = pl.DataFrame({"repeat_id": [1], "active": [True]})

    # Run the sampling loop
    for step in range(10):
        # check stopping condition
        stop_flags = locator.should_stop(history, repeats, scan)
        active_repeats = stop_flags.filter(pl.col("stop").not_()).select("repeat_id")  # noqa: F841

        # Update active repeats based on stop flags
        repeats = repeats.join(stop_flags.select(["repeat_id", "stop"]), on="repeat_id").select(
            pl.col("repeat_id"), (pl.col("active") & pl.col("stop").not_()).alias("active")
        )

        if repeats.filter(pl.col("active")).is_empty():
            break

        proposals = locator.propose_next(history, repeats, scan)

        new_rows = []
        for prop in proposals.iter_rows(named=True):
            rid = prop["repeat_id"]
            x = prop["x"]
            y = scan.signal(x)
            new_rows.append(
                {
                    "repeat_id": rid,
                    "step": step,
                    "x": x,
                    "signal_values": y,
                }
            )

        if new_rows:
            history = pl.concat([history, pl.DataFrame(new_rows, schema=schema)])

    # Check that we have the expected number of measurements
    assert len(history) == 10, f"Expected 10 measurements, got {len(history)}"

    # Check that we have measurements within the expected range
    x_values = history["x"].to_list()
    in_range = [2.8e9 <= x <= 2.9e9 for x in x_values]
    assert any(in_range), f"No measurements in expected range 2.8e9-2.9e9. Got: {x_values}"
