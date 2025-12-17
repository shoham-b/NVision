"""Integration tests for the single-version NVCenterSequentialBayesianLocator."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nvision.sim.locs.base import ScanBatch
from nvision.sim.locs.nv_center.sequential_bayesian_locator import (
    NVCenterSequentialBayesianLocator,
)


def build_locator(**overrides: object) -> NVCenterSequentialBayesianLocator:
    """Create a locator instance with test-friendly defaults."""
    config: dict[str, object] = {
        "max_evals": 12,
        "prior_bounds": (2.7e9, 3.0e9),
        "grid_resolution": 64,
        "n_monte_carlo": 40,
        "noise_model": "gaussian",
    }
    config.update(overrides)
    return NVCenterSequentialBayesianLocator(**config)


def build_scan() -> ScanBatch:
    center = 2.85e9
    width = 5e6

    def signal(x: float) -> float:
        z = (x - center) / width
        return float(1.0 - 0.1 * np.exp(-0.5 * z * z))

    return ScanBatch(
        x_min=2.7e9,
        x_max=3.0e9,
        signal=signal,
        meta={"inference": {"peaks": []}},
        truth_positions=[center],
    )


def test_initial_state_uniform():
    locator = build_locator()
    assert locator.freq_grid.shape == (locator.grid_resolution,)
    assert np.isclose(locator.freq_posterior.sum(), 1.0)


def test_likelihood_supports_noise_models():
    locator = build_locator()
    measurement = {"x": 2.86e9, "signal_values": 0.92}
    params = {"frequency": 2.86e9, "linewidth": 5e6, "amplitude": 0.1, "background": 1.0}

    gaussian_ll = locator.likelihood(measurement, params)
    locator.noise_model = "poisson"
    poisson_ll = locator.likelihood(measurement, params)

    assert isinstance(gaussian_ll, float)
    assert isinstance(poisson_ll, float)


def test_update_posterior_modifies_distribution():
    locator = build_locator()
    prior = locator.freq_posterior.copy()
    locator.update_posterior({"x": 2.84e9, "signal_values": 0.91})
    assert not np.allclose(prior, locator.freq_posterior)


def test_expected_information_gain_is_non_negative():
    locator = build_locator()
    locator.update_posterior({"x": 2.84e9, "signal_values": 0.91})
    eig = locator.expected_information_gain(2.86e9)
    assert eig >= 0.0


def test_reset_posterior_returns_uniform_prior():
    locator = build_locator()
    locator.update_posterior({"x": 2.84e9, "signal_values": 0.91})
    locator.reset_posterior()
    assert np.allclose(locator.freq_posterior, 1.0 / locator.grid_resolution)


def test_unknown_noise_model_raises():
    locator = build_locator(noise_model="unknown")
    measurement = {"x": 2.86e9, "signal_values": 0.92}
    params = {"frequency": 2.86e9, "linewidth": 5e6, "amplitude": 0.1, "background": 1.0}
    with pytest.raises(ValueError, match="Unknown noise model"):
        locator.likelihood(measurement, params)


def test_should_stop_when_utility_stalls():
    """Test that should_stop returns True when utility stalls."""
    locator = build_locator(utility_history_window=2, min_uncertainty_reduction=1e-8)
    scan = build_scan()

    # Set up utility history to simulate stalled optimization
    locator.utility_history = [1e-9, 1e-10]  # Below min_uncertainty_reduction
    assert locator.should_stop(pl.DataFrame(), scan)


def test_odmr_model_handles_distributions():
    locator = build_locator()

    params = locator.current_estimates.copy()

    # Override for test clarity
    params["amplitude"] = 0.1
    params["background"] = 1.0
    params["linewidth"] = 5e6
    params["gaussian_width"] = 5e6
    params["split"] = 30e6

    # Test lorentzian
    locator.distribution = "lorentzian"
    val_l_peak = locator.odmr_model(params["frequency"], params)
    assert val_l_peak == pytest.approx(params["background"] - params["amplitude"])
    val_l_far = locator.odmr_model(params["frequency"] + 20 * params["linewidth"], params)
    assert val_l_far == pytest.approx(params["background"], abs=1e-3)

    # Test voigt
    locator.distribution = "voigt"
    val_v_peak = locator.odmr_model(params["frequency"], params)
    assert val_v_peak == pytest.approx(params["background"] - params["amplitude"])
    val_v_far = locator.odmr_model(params["frequency"] + 20 * params["linewidth"], params)
    assert val_v_far == pytest.approx(params["background"], abs=1e-3)

    # Test voigt-zeeman
    locator.distribution = "voigt-zeeman"
    val_vz_center = locator.odmr_model(params["frequency"], params)
    peak_freq_1 = params["frequency"] - params["split"] / 2
    val_vz_peak1 = locator.odmr_model(peak_freq_1, params)
    assert val_vz_peak1 > val_vz_center

    # Test unknown distribution
    locator.distribution = "unknown"
    with pytest.raises(ValueError, match="Unknown distribution"):
        locator.odmr_model(params["frequency"], params)
