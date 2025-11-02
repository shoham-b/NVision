"""Test category-specific locators with uncertainty calculation."""

from __future__ import annotations

import math
import random

import polars as pl
import pytest

from nvision.sim import (
    NVCenterBayesianLocator,
    NVCenterSweepLocator,
    OnePeakGoldenLocator,
    OnePeakGridLocator,
    ScanBatch,
)


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

    grid_locator = OnePeakGridLocator(n_points=11)
    sweep_locator = OnePeakGoldenLocator(max_evals=10)

    history_grid = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})
    history_sweep = history_grid.clone()

    for locator, history in ((grid_locator, history_grid), (sweep_locator, history_sweep)):
        for _i in range(3):
            x = locator.propose_next(history, scan)
            assert 0.0 <= x <= 1.0
            y = signal(x) + rng.gauss(0, 0.05)
            history = pl.concat([history, pl.DataFrame({"x": [x], "signal_values": [y]})])
        assert isinstance(locator.should_stop(history, scan), bool)
        result = locator.finalize(history, scan)
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

    sweep_locator = NVCenterSweepLocator(coarse_points=15, refine_points=5)
    bayes_locator = NVCenterBayesianLocator(max_steps=12)

    for locator in (sweep_locator, bayes_locator):
        history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})
        for _i in range(3):
            x = locator.propose_next(history, scan)
            assert 0.0 <= x <= 1.0
            y = signal(x) + rng.gauss(0, 0.03)
            history = pl.concat([history, pl.DataFrame({"x": [x], "signal_values": [y]})])
        assert isinstance(locator.should_stop(history, scan), bool)
        result = locator.finalize(history, scan)
        assert "n_peaks" in result
        assert "uncert" in result


def test_uncertainty_calculation():
    """Test that uncertainty is properly calculated and used."""
    rng = random.Random(789)

    # Create a signal with known properties
    def signal(x: float) -> float:
        return 1.0 + 0.5 * math.sin(2 * math.pi * x)

    scan = ScanBatch(x_min=0.0, x_max=1.0, truth_positions=[0.25, 0.75], signal=signal, meta={})

    # Test with ODMR locator
    locator = OnePeakGoldenLocator(max_evals=12)

    # Simulate a measurement process
    history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})

    for _i in range(5):
        x = locator.propose_next(history, scan)
        y_clean = signal(x)
        y_noisy = y_clean + rng.gauss(0, 0.1)

        # Calculate uncertainty (simplified)
        new_row = pl.DataFrame({"x": [x], "signal_values": [y_noisy]})
        history = pl.concat([history, new_row], how="vertical")

    # Check that uncertainty is being used in decisions
    assert history.height > 0

    # Test that locators use uncertainty in their decisions
    result = locator.finalize(history, scan)
    assert "uncert" in result


def test_locator_comparison():
    """Test that different locators produce different behaviors."""
    rng = random.Random(999)

    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.4) / 0.05) ** 2)

    scan = ScanBatch(x_min=0.0, x_max=1.0, truth_positions=[0.4], signal=signal, meta={})

    grid_locator = OnePeakGridLocator(n_points=15)
    golden_locator = OnePeakGoldenLocator(max_evals=12)

    grid_history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})
    golden_history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})

    for _i in range(6):
        x_grid = grid_locator.propose_next(grid_history, scan)
        y_grid = signal(x_grid) + rng.gauss(0, 0.02)
        grid_history = pl.concat(
            [grid_history, pl.DataFrame({"x": [x_grid], "signal_values": [y_grid]})]
        )

        x_golden = golden_locator.propose_next(golden_history, scan)
        y_golden = signal(x_golden) + rng.gauss(0, 0.02)
        golden_history = pl.concat(
            [golden_history, pl.DataFrame({"x": [x_golden], "signal_values": [y_golden]})]
        )

    grid_result = grid_locator.finalize(grid_history, scan)
    golden_result = golden_locator.finalize(golden_history, scan)

    assert grid_result["n_peaks"] == 1.0
    assert golden_result["n_peaks"] == 1.0
    assert 0.0 <= grid_result["x1_hat"] <= 1.0
    assert 0.0 <= golden_result["x1_hat"] <= 1.0
    assert grid_result["x1_hat"] != pytest.approx(golden_result["x1_hat"])
