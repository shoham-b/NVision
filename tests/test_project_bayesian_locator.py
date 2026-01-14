import numpy as np

from nvision.sim.locs.base import ScanBatch
from nvision.sim.locs.nv_center.project_bayesian_locator import ProjectBayesianLocator


def build_locator(**overrides):
    config = {
        "max_evals": 10,
        "prior_bounds": (2.8e9, 2.9e9),
        "linewidth_prior": (5e6, 20e6),
        "grid_resolution": 64,
        "n_monte_carlo": 10,
        "n_warmup": 0,
        "distribution": "lorentzian",
    }
    config.update(overrides)
    return ProjectBayesianLocator(**config)


def build_scan():
    return ScanBatch(
        x_min=2.8e9,
        x_max=2.9e9,
        signal=lambda x: 1.0,
        meta={"inference": {"peaks": []}},
        truth_positions=[2.85e9],
    )


def test_project_locator_initialization():
    locator = build_locator()
    locator._init_bayes_optimizer()  # Usually called in post_init but we want to ensure state
    locator.reset_posterior()

    assert isinstance(locator, ProjectBayesianLocator)
    # Check 2D posterior initialization
    assert hasattr(locator, "posterior_2d")
    assert locator.posterior_2d.shape == (locator.grid_resolution, locator.linewidth_resolution)
    assert np.isclose(np.sum(locator.posterior_2d), 1.0)


def test_project_locator_update():
    locator = build_locator(grid_resolution=50, n_monte_carlo=50, pickiness=5.0)
    locator.reset_posterior()

    # Propose next
    # proposal = locator.propose_next([], build_scan()) # requires real history or scan
    # Direct optimize call
    freq, _ = locator._optimize_acquisition((2.8e9, 2.9e9))
    assert 2.8e9 <= freq <= 2.9e9

    # Simulate measurement
    true_f0 = 2.85e9
    true_gamma = 10e6
    hwhm = true_gamma / 2.0

    def lorentzian(f):
        return 1.0 - 0.1 * hwhm**2 / ((f - true_f0) ** 2 + hwhm**2)

    signal = lorentzian(freq)
    measurement = {"x": freq, "signal_values": signal, "uncertainty": 0.05}

    # Update
    locator.update_posterior(measurement)

    # Check posterior sum
    assert np.isclose(np.sum(locator.posterior_2d), 1.0)

    # Check estimates
    est_freq = locator.current_estimates["frequency"]
    est_gamma = locator.current_estimates["linewidth"]

    assert 2.8e9 <= est_freq <= 2.9e9
    assert 5e6 <= est_gamma <= 20e6
    assert not np.isnan(est_freq)
    assert not np.isnan(est_gamma)


def test_project_locator_propose_next():
    locator = build_locator()
    scan = build_scan()
    # Testing propose_next interface
    proposal = locator.propose_next([], scan)
    assert isinstance(proposal, float)
    assert 2.8e9 <= proposal <= 2.9e9
