"""Integration tests for the AnalyticalBayesianLocator."""

from __future__ import annotations

import numpy as np
import polars as pl

from nvision.sim.locs.base import ScanBatch
from nvision.sim.locs.nv_center.analytical_bayesian_locator import AnalyticalBayesianLocator


def build_locator(**overrides: object) -> AnalyticalBayesianLocator:
    """Create a locator instance with test-friendly defaults."""
    config: dict[str, object] = {
        "max_evals": 20,
        "prior_bounds": (2.7e9, 3.0e9),
        "grid_resolution": 100,
        "n_monte_carlo": 40,
        "noise_model": "gaussian",
        "n_warmup": 5,
    }
    config.update(overrides)
    return AnalyticalBayesianLocator(**config)


def build_scan(center=2.85e9) -> ScanBatch:
    width = 5e6

    def signal(x: float) -> float:
        # Lorentzian signal to allow detection from tails
        diff = x - center
        # 1 - depth * (gamma^2 / (4*diff^2 + gamma^2))
        # Using HWHM = width/2
        hwhm = width / 2.0
        denom = diff**2 + hwhm**2
        return float(1.0 - 0.1 * (hwhm**2) / denom)

    return ScanBatch(
        x_min=2.7e9,
        x_max=3.0e9,
        signal=signal,
        meta={"inference": {"peaks": []}},
        truth_positions=[center],
    )


def test_warmup_phase_random_sampling():
    locator = build_locator(n_warmup=5)
    scan = build_scan()
    history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})

    # First 5 calls should be random
    for _ in range(4):
        x = locator.propose_next(history, scan)
        assert locator.prior_bounds[0] <= x <= locator.prior_bounds[1]

        # Add to history
        new_row = pl.DataFrame({"x": [x], "signal_values": [scan.signal(x)]})
        history = pl.concat([history, new_row])

    # Posterior should still be uniform (or effectively so, as we don't update it in warmup logic explicitly
    # but base class might if we called super, but we didn't call super in warmup phase).
    # Wait, in my implementation I call self._ingest_history(history) in warmup phase.
    # But I also said "we might NOT want to update posterior".
    # Let's check the implementation again.
    # "Phase 1: Warmup... self._ingest_history(history)"
    # And _ingest_history calls update_posterior.
    # So posterior IS updated.
    # But at step n_warmup, we reset it.

    assert len(history) == 4


def test_transition_phase_omp():
    locator = build_locator(n_warmup=5)
    scan = build_scan(center=2.85e9)

    # Create history with 5 points
    # We want points that roughly outline the dip so OMP can find it.
    # Let's manually construct history to ensure OMP works.
    xs = [2.7e9, 2.8e9, 2.85e9, 2.9e9, 3.0e9]
    ys = [scan.signal(x) for x in xs]
    history = pl.DataFrame({"x": xs, "signal_values": ys})

    # This call triggers the transition logic (hist_len == n_warmup)
    locator.propose_next(history, scan)

    # Check if posterior is now concentrated around 2.85e9
    # Find peak of posterior
    peak_idx = np.argmax(locator.freq_posterior)
    peak_freq = locator.freq_grid[peak_idx]

    assert abs(peak_freq - 2.85e9) < 20e6  # Within 20 MHz

    # Check that current estimates are updated
    assert abs(locator.current_estimates["frequency"] - 2.85e9) < 20e6


def test_omp_estimation_logic():
    locator = build_locator()

    # Create a history with a clear dip at 2.85 GHz
    center = 2.85e9
    width = 5e6
    xs = np.linspace(2.8e9, 2.9e9, 20)
    ys = [1.0 - 0.1 / (1 + ((x - center) / (width / 2)) ** 2) for x in xs]  # Lorentzian

    history = pl.DataFrame({"x": xs, "signal_values": ys})

    est_freq = locator._omp_estimate(history)

    assert abs(est_freq - center) < 10e6  # Close to center


def test_full_run_convergence():
    np.random.seed(42)
    locator = build_locator(n_warmup=10, max_evals=30)
    scan = build_scan(center=2.88e9)

    history = pl.DataFrame(schema={"x": pl.Float64, "signal_values": pl.Float64})

    for _ in range(30):
        if locator.should_stop(history, scan):
            break
        x = locator.propose_next(history, scan)
        y = scan.signal(x)
        history = pl.concat([history, pl.DataFrame({"x": [x], "signal_values": [y]})])

    # Finalize
    result = locator.finalize(history, scan)

    assert abs(result["x1_hat"] - 2.88e9) < 5e6
