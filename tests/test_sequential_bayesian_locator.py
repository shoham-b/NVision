"""Integration tests for SequentialBayesianLocator aligned with current API."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nvision.sim import NVCenterSequentialBayesianLocator, ScanBatch


def build_locator(**overrides: object) -> NVCenterSequentialBayesianLocator:
    config: dict[str, object] = {
        "max_evals": 12,
        "prior_bounds": (2.7e9, 3.0e9),
        "grid_resolution": 64,
        "n_monte_carlo": 40,
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


def test_propose_next_consumes_polars_history():
    locator = build_locator()
    history = pl.DataFrame({"x": [2.76e9, 2.88e9], "signal_values": [0.98, 0.90]})
    scan = build_scan()
    proposal = locator.propose_next(history, scan)
    assert scan.x_min <= proposal <= scan.x_max


def test_should_stop_respects_max_evaluations():
    locator = build_locator(max_evals=2)
    scan = build_scan()
    history = pl.DataFrame({"x": [scan.x_min, scan.x_max], "signal_values": [1.0, 0.9]})
    assert locator.should_stop(history, scan)


def test_finalize_includes_required_fields():
    locator = build_locator()
    scan = build_scan()
    history = pl.DataFrame({"x": [2.84e9, 2.86e9], "signal_values": [0.95, 0.9]})
    result = locator.finalize(history, scan)
    assert {"n_peaks", "x1_hat", "uncert"} <= set(result)


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
    locator = build_locator()
    locator.utility_history = [0.0, 0.0, 0.0]
    scan = build_scan()
    history = pl.DataFrame({"x": [2.85e9], "signal_values": [0.9]})
    assert locator.should_stop(history, scan)
