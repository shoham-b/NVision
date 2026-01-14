"""Test category-specific locators with uncertainty calculation."""

from __future__ import annotations

import math
import random

import polars as pl
import pytest

from nvision.sim.locs.base import ScanBatch
from nvision.sim.locs.nv_center.sweep_locator import NVCenterSweepLocator
from nvision.sim.locs.one_peak import OnePeakGoldenLocator, OnePeakGridLocator

from .locator_compat import LegacyLocatorShim


class UniformPrior:
    def get_probabilities(self, min_x: float, max_x: float, num_x_bins: int) -> list[float]:
        if num_x_bins <= 0:
            return []
        return [1.0 / num_x_bins] * num_x_bins


class IdentityObs:
    def get_uncertainty(self, posterior: list[float]) -> list[float]:
        return posterior

    def update_posterior(
        self,
        posterior: list[float],
        x_measured: float,
        y_measured: float,
        min_x: float,
        max_x: float,
        num_x_bins: int,
    ) -> list[float]:
        return posterior


# The following helper functions mirror simplified observer interfaces used in NVCenter
# Bayesian locator tests. They provide deterministic behaviour for the test scenario.


def test_one_peak_locators_basic():
    """Test that the one-peak locators operate and return uncertainty information."""
    rng = random.Random(123)

    # Create a simple peak signal
    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)

    scan = ScanBatch(
        x_min=0.0,
        x_max=1.0,
        truth_positions=[0.5],
        signal=signal,
        meta={"peak_width": 0.1},
    )

    grid_locator = LegacyLocatorShim(OnePeakGridLocator(n_points=11))
    sweep_locator = LegacyLocatorShim(OnePeakGoldenLocator(max_evals=10))

    # Create initial history and repeats dataframes
    history_grid = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})
    history_sweep = history_grid.clone()
    repeats_df = pl.DataFrame({"repeat_id": [0], "active": [True]})

    for locator, history in ((grid_locator, history_grid), (sweep_locator, history_sweep)):
        current_history = history
        for _i in range(3):
            # Pass repeats_df to propose_next
            next_proposal = locator.propose_next(current_history, repeats_df, scan)
            # Extract x from the returned DataFrame
            x = next_proposal.get_column("x")[0]
            assert 0.0 <= x <= 1.0
            y = signal(x) + rng.gauss(0, 0.05)
            # Update history with new measurement
            current_history = pl.concat([current_history, pl.DataFrame({"x": [x], "signal_values": [y]})])
            # Update repeats_df based on should_stop result
            stop_df = locator.should_stop(current_history, repeats_df, scan)
            repeats_df = stop_df.select(["repeat_id", "stop"]).with_columns(pl.lit(True).alias("active"))

        # Final check and result
        final_stop = locator.should_stop(current_history, repeats_df, scan)
        assert isinstance(final_stop, pl.DataFrame)
        assert "stop" in final_stop.columns

        result = locator.finalize(current_history, scan)
        assert "n_peaks" in result
        assert "x1_hat" in result
        assert "uncert" in result


def test_nv_center_locators_basic():
    """Test NV center locators return measurement counts and uncertainty."""
    rng = random.Random(456)

    # Create a simple peak signal
    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.3) / 0.08) ** 2)

    scan = ScanBatch(
        x_min=0.0,
        x_max=1.0,
        truth_positions=[0.3],
        signal=signal,
        meta={"peak_width": 0.08},
    )

    # Create initial history and repeats dataframes
    history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})
    repeats_df = pl.DataFrame({"repeat_id": [0], "active": [True]})

    # Create locator
    locator = LegacyLocatorShim(NVCenterSweepLocator(coarse_points=15, refine_points=5))

    current_history = history
    for _i in range(3):
        # Pass repeats_df to propose_next
        next_proposal = locator.propose_next(current_history, repeats_df, scan)
        # Extract x from the returned DataFrame
        x = next_proposal.get_column("x")[0]
        assert 0.0 <= x <= 1.0
        y = signal(x) + rng.gauss(0, 0.03)
        # Update history with new measurement
        current_history = pl.concat([current_history, pl.DataFrame({"x": [x], "signal_values": [y]})])
        # Update repeats_df based on should_stop result
        stop_df = locator.should_stop(current_history, repeats_df, scan)
        repeats_df = stop_df.select(["repeat_id", "stop"]).with_columns(pl.lit(True).alias("active"))

    # Final check and result
    final_stop = locator.should_stop(current_history, repeats_df, scan)
    assert isinstance(final_stop, pl.DataFrame)
    assert "stop" in final_stop.columns

    result = locator.finalize(current_history, scan)
    assert "n_peaks" in result
    assert "uncert" in result


def test_uncertainty_calculation():
    """Test that uncertainty is properly calculated and used."""
    rng = random.Random(789)

    # Create a signal with known properties
    def signal(x: float) -> float:
        return 1.0 + 0.5 * math.sin(2 * math.pi * x)

    scan = ScanBatch(x_min=0.0, x_max=1.0, truth_positions=[0.25, 0.75], signal=signal, meta={})

    # Create initial history and repeats dataframes
    history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})
    repeats_df = pl.DataFrame({"repeat_id": [0], "active": [True]})

    # Create locator
    locator = LegacyLocatorShim(OnePeakGoldenLocator(max_evals=12))

    # Simulate a measurement process
    current_history = history
    for _i in range(5):
        # Get next proposal with repeats_df
        next_proposal = locator.propose_next(current_history, repeats_df, scan)
        x = next_proposal.get_column("x")[0]
        y_clean = signal(x)
        y_noisy = y_clean + rng.gauss(0, 0.1)

        # Update history with new measurement
        current_history = pl.concat(
            [current_history, pl.DataFrame({"x": [x], "signal_values": [y_noisy]})], how="vertical"
        )

        # Update repeats_df based on should_stop result
        stop_df = locator.should_stop(current_history, repeats_df, scan)
        repeats_df = stop_df.select(["repeat_id", "stop"]).with_columns(pl.lit(True).alias("active"))

    # Check that we have measurements
    assert current_history.height > 0

    # Test that locators use uncertainty in their decisions
    result = locator.finalize(current_history, scan)
    assert "uncert" in result


def test_locator_comparison():
    """Test that different locators produce different behaviors."""
    rng = random.Random(999)

    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.4) / 0.05) ** 2)

    scan = ScanBatch(x_min=0.0, x_max=1.0, truth_positions=[0.4], signal=signal, meta={})

    # Create initial history and repeats dataframes for both locators
    grid_history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})
    golden_history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})
    grid_repeats = pl.DataFrame({"repeat_id": [0], "active": [True]})
    golden_repeats = pl.DataFrame({"repeat_id": [0], "active": [True]})

    # Create locators
    grid_locator = LegacyLocatorShim(OnePeakGridLocator(n_points=15))
    golden_locator = LegacyLocatorShim(OnePeakGoldenLocator(max_evals=12))

    for _i in range(6):
        # Grid locator step
        grid_proposal = grid_locator.propose_next(grid_history, grid_repeats, scan)
        x_grid = grid_proposal.get_column("x")[0]
        y_grid = signal(x_grid) + rng.gauss(0, 0.02)
        grid_history = pl.concat([grid_history, pl.DataFrame({"x": [x_grid], "signal_values": [y_grid]})])
        grid_stop = grid_locator.should_stop(grid_history, grid_repeats, scan)
        grid_repeats = grid_stop.select(["repeat_id", "stop"]).with_columns(pl.lit(True).alias("active"))

        # Golden locator step
        golden_proposal = golden_locator.propose_next(golden_history, golden_repeats, scan)
        x_golden = golden_proposal.get_column("x")[0]
        y_golden = signal(x_golden) + rng.gauss(0, 0.02)
        golden_history = pl.concat([golden_history, pl.DataFrame({"x": [x_golden], "signal_values": [y_golden]})])
        golden_stop = golden_locator.should_stop(golden_history, golden_repeats, scan)
        golden_repeats = golden_stop.select(["repeat_id", "stop"]).with_columns(pl.lit(True).alias("active"))

    # Get final results
    grid_result = grid_locator.finalize(grid_history, scan)
    golden_result = golden_locator.finalize(golden_history, scan)

    # Verify results
    assert grid_result["n_peaks"] == 1.0
    assert golden_result["n_peaks"] == 1.0
    assert 0.0 <= grid_result["x1_hat"] <= 1.0
    assert 0.0 <= golden_result["x1_hat"] <= 1.0
    assert grid_result["x1_hat"] != pytest.approx(golden_result["x1_hat"], abs=1e-6)
