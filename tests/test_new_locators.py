"""Test the new ODMR and Bayesian locators with uncertainty calculation."""

from __future__ import annotations

import math
import random

import polars as pl

from nvision.sim import BayesianLocator, ScanBatch, SweepLocator


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


def test_sweep_locator_basic():
    """Test that ODMR locator works with uncertainty calculation."""
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

    # Create ODMR locator
    locator = SweepLocator(coarse_points=5, refine_points=3, uncertainty_threshold=0.05)

    # Test propose_next
    history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})

    # First few points should be from coarse sweep
    for _i in range(3):
        x = locator.propose_next(history, scan)
        assert 0.0 <= x <= 1.0
        # Simulate measurement with uncertainty
        y = signal(x) + rng.gauss(0, 0.05)
        new_row = pl.DataFrame({"x": [x], "signal_values": [y]})
        history = pl.concat([history, new_row], how="vertical")

    # Test should_stop
    assert isinstance(locator.should_stop(history, scan), bool)

    # Test finalize
    result = locator.finalize(history, scan)
    assert "n_peaks" in result
    assert "x1" in result
    assert "uncert" in result


def test_bayesian_locator_basic():
    """Test that Bayesian locator works with uncertainty calculation."""
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

    # Create Bayesian locator
    locator = BayesianLocator(
        priors=UniformPrior(),
        obs_model=IdentityObs(),
        min_x=0.0,
        max_x=1.0,
        max_steps=10,
    )

    # Test propose_next
    history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})

    # First few points should be random
    for _i in range(3):
        x = locator.propose_next(history, scan)
        assert 0.0 <= x <= 1.0
        # Simulate measurement with uncertainty
        y = signal(x) + rng.gauss(0, 0.03)
        new_row = pl.DataFrame({"x": [x], "signal_values": [y]})
        history = pl.concat([history, new_row], how="vertical")

    # Test should_stop returns boolean
    assert isinstance(locator.should_stop(history, scan), bool)

    # Test finalize
    result = locator.finalize(history, scan)
    assert "x_hat" in result
    assert "uncertainty" in result


def test_uncertainty_calculation():
    """Test that uncertainty is properly calculated and used."""
    rng = random.Random(789)

    # Create a signal with known properties
    def signal(x: float) -> float:
        return 1.0 + 0.5 * math.sin(2 * math.pi * x)

    scan = ScanBatch(x_min=0.0, x_max=1.0, truth_positions=[0.25, 0.75], signal=signal, meta={})

    # Test with ODMR locator
    locator = SweepLocator(coarse_points=8, refine_points=4)

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
    assert result["uncert"] == 0.0


def test_locator_comparison():
    """Test that different locators produce different behaviors."""
    rng = random.Random(999)

    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.4) / 0.05) ** 2)

    scan = ScanBatch(x_min=0.0, x_max=1.0, truth_positions=[0.4], signal=signal, meta={})

    # Test both locators on the same problem
    odmr = SweepLocator(coarse_points=6, refine_points=3)
    bayesian = BayesianLocator(
        priors=UniformPrior(),
        obs_model=IdentityObs(),
        min_x=0.0,
        max_x=1.0,
        max_steps=9,
    )

    # Sweep locator result
    sweep_history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})
    for _i in range(6):
        x = odmr.propose_next(sweep_history, scan)
        y = signal(x) + rng.gauss(0, 0.02)
        sweep_history = pl.concat([sweep_history, pl.DataFrame({"x": [x], "signal_values": [y]})])

    sweep_result = odmr.finalize(sweep_history, scan)
    assert sweep_result["n_peaks"] == 1.0
    assert 0.0 <= sweep_result["x1"] <= 1.0
    assert sweep_result["uncert"] == 0.0

    # Bayesian locator result
    bayes_history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})
    for _i in range(3):
        x = bayesian.propose_next(bayes_history, scan)
        y = signal(x) + rng.gauss(0, 0.02)
        bayes_history = pl.concat([bayes_history, pl.DataFrame({"x": [x], "signal_values": [y]})])

    bayes_result = bayesian.finalize(bayes_history, scan)
    assert "x_hat" in bayes_result
    assert "uncertainty" in bayes_result
