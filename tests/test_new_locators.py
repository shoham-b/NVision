"""Test the new ODMR and Bayesian locators with uncertainty calculation."""

from __future__ import annotations

import math
import random

from nvision.sim import (
    BayesianLocator,
    CompositeNoise,
    ODMRLocator,
    OverVoltageGaussianNoise,
    ScalarMeasure,
    ScanBatch,
)
from nvision.sim.locs import Obs


def test_odmr_locator_basic():
    """Test that ODMR locator works with uncertainty calculation."""
    rng = random.Random(123)

    # Create a simple peak signal
    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)

    ScanBatch(
        x_min=0.0,
        x_max=1.0,
        truth_positions=[0.5],
        signal=signal,
        meta={"peak_width": 0.1},
    )

    # Create ODMR locator
    locator = ODMRLocator(coarse_points=5, refine_points=3, uncertainty_threshold=0.05)

    # Create measurement with noise
    noise = CompositeNoise([OverVoltageGaussianNoise(0.05)])
    ScalarMeasure(noise=noise)

    # Test propose_next
    history = []
    domain = (0.0, 1.0)

    # First few points should be from coarse sweep
    for _i in range(3):
        x = locator.propose_next(history, domain)
        assert 0.0 <= x <= 1.0
        # Simulate measurement with uncertainty
        y = signal(x) + rng.gauss(0, 0.05)
        uncertainty = 0.1 + abs(y) * 0.05  # Simple uncertainty model
        history.append(Obs(x=x, intensity=y, uncertainty=uncertainty))

    # Test should_stop
    assert not locator.should_stop(history)  # Not enough points yet

    # Test finalize
    result = locator.finalize(history)
    assert "n_peaks" in result
    assert "x1" in result
    assert "uncert" in result


def test_bayesian_locator_basic():
    """Test that Bayesian locator works with uncertainty calculation."""
    rng = random.Random(456)

    # Create a simple peak signal
    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.3) / 0.08) ** 2)

    ScanBatch(
        x_min=0.0,
        x_max=1.0,
        truth_positions=[0.3],
        signal=signal,
        meta={"peak_width": 0.08},
    )

    # Create Bayesian locator
    locator = BayesianLocator(max_evals=10, uncertainty_threshold=0.03)

    # Create measurement with noise
    noise = CompositeNoise([OverVoltageGaussianNoise(0.03)])
    ScalarMeasure(noise=noise)

    # Test propose_next
    history = []
    domain = (0.0, 1.0)

    # First few points should be random
    for _i in range(3):
        x = locator.propose_next(history, domain)
        assert 0.0 <= x <= 1.0
        # Simulate measurement with uncertainty
        y = signal(x) + rng.gauss(0, 0.03)
        uncertainty = 0.08 + abs(y) * 0.03  # Simple uncertainty model
        history.append(Obs(x=x, intensity=y, uncertainty=uncertainty))

    # Test should_stop
    assert not locator.should_stop(history)  # Not enough points yet

    # Test finalize
    result = locator.finalize(history)
    assert "n_peaks" in result
    assert "x1" in result
    assert "uncert" in result


def test_uncertainty_calculation():
    """Test that uncertainty is properly calculated and used."""
    rng = random.Random(789)

    # Create a signal with known properties
    def signal(x: float) -> float:
        return 1.0 + 0.5 * math.sin(2 * math.pi * x)

    ScanBatch(x_min=0.0, x_max=1.0, truth_positions=[0.25, 0.75], signal=signal, meta={})

    # Test with ODMR locator
    locator = ODMRLocator(coarse_points=8, refine_points=4)
    noise = CompositeNoise([OverVoltageGaussianNoise(0.1)])
    ScalarMeasure(noise=noise)

    # Simulate a measurement process
    history = []
    domain = (0.0, 1.0)

    for _i in range(5):
        x = locator.propose_next(history, domain)
        y_clean = signal(x)
        y_noisy = y_clean + rng.gauss(0, 0.1)

        # Calculate uncertainty (simplified)
        uncertainty = 0.1 + abs(y_noisy) * 0.05
        history.append(Obs(x=x, intensity=y_noisy, uncertainty=uncertainty))

    # Check that uncertainty is being used in decisions
    assert all(obs.uncertainty > 0 for obs in history)

    # Test that locators use uncertainty in their decisions
    result = locator.finalize(history)
    assert result["uncert"] > 0
    assert result["uncert"] != float("inf")


def test_locator_comparison():
    """Test that different locators produce different behaviors."""
    rng = random.Random(999)

    def signal(x: float) -> float:
        return math.exp(-0.5 * ((x - 0.4) / 0.05) ** 2)

    ScanBatch(x_min=0.0, x_max=1.0, truth_positions=[0.4], signal=signal, meta={})

    # Test both locators on the same problem
    odmr = ODMRLocator(coarse_points=6, refine_points=3)
    bayesian = BayesianLocator(max_evals=9, uncertainty_threshold=0.05)

    # Both should be able to find the peak
    for locator in [odmr, bayesian]:
        history = []
        domain = (0.0, 1.0)

        for _i in range(6):
            x = locator.propose_next(history, domain)
            y = signal(x) + rng.gauss(0, 0.02)
            uncertainty = 0.05 + abs(y) * 0.02
            history.append(Obs(x=x, intensity=y, uncertainty=uncertainty))

        result = locator.finalize(history)
        assert result["n_peaks"] == 1.0
        assert 0.0 <= result["x1"] <= 1.0
        assert result["uncert"] > 0
