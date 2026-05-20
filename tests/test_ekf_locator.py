"""Tests for the EKF belief and EKF locator."""

from __future__ import annotations

import math
import random

import numpy as np

from nvision.belief.abstract_marginal import ParameterValues
from nvision.models.experiment import CoreExperiment
from nvision.runner.executor import run_loop
from nvision.sim.locs.ekf.belief import EKFBelief
from nvision.sim.locs.ekf.ekf_locator import EKFAOptimalLocator, EKFDOptimalLocator
from nvision.spectra.nv_center import NVCenterLorentzianModel
from nvision.spectra.signal import TrueSignal


def _nv_center_experiment() -> CoreExperiment:
    from nvision.spectra.nv_center import NVCenterLorentzianSpectrum

    model = NVCenterLorentzianModel()
    typed_params = NVCenterLorentzianSpectrum(
        frequency=2.87e9,
        linewidth=200e3,
        split=2.1e6,
        k_np=3.0,
        dip_depth=0.5,
    )
    bounds = {
        "frequency": (2.6e9, 3.1e9),
        "linewidth": (50e3, 400e3),
        "split": (2.0e6, 3.5e6),
        "k_np": (2.0, 4.0),
        "dip_depth": (0.1, 1.0),
    }
    true_signal = TrueSignal(model=model, typed_parameters=typed_params, bounds=bounds)
    return CoreExperiment(true_signal=true_signal, noise=None, x_min=2.6e9, x_max=3.1e9)


def test_ekf_belief_methods():
    model = NVCenterLorentzianModel()
    theta_initial = np.array([2870.0, 2.1, 0.05, 5.0, 1.0])
    P_initial = np.eye(5) * 0.1  # noqa: N806
    bounds = {
        "frequency": (2.6e9, 3.1e9),
        "linewidth": (50e3, 400e3),
        "split": (2.0e6, 3.5e6),
        "k_np": (2.0, 4.0),
        "dip_depth": (0.1, 1.0),
    }
    belief = EKFBelief(
        model=model,
        theta_initial=theta_initial,
        P_initial=P_initial,
        R=0.05**2,
        parameter_bounds=bounds,
    )

    # 1. Estimates
    est = belief.estimates()
    assert isinstance(est, dict)
    assert all(name in est for name in model.parameter_names())
    assert est["frequency"] == 2870.0 * 1e6
    assert est["split"] == 2.1 * 1e6

    # 2. Uncertainty
    unc = belief.uncertainty()
    assert isinstance(unc, ParameterValues)
    assert all(name in unc for name in model.parameter_names())
    assert all(unc[name] >= 0 for name in model.parameter_names())

    # 3. Entropy
    ent = belief.entropy()
    assert isinstance(ent, float)

    # 4. Copy
    copied = belief.copy()
    assert isinstance(copied, EKFBelief)
    assert np.allclose(copied.theta_hat, belief.theta_hat)
    assert np.allclose(copied.P, belief.P)

    # 5. Sample
    samples = belief.sample(10)
    assert isinstance(samples, ParameterValues)
    assert all(samples[name].shape == (10,) for name in model.parameter_names())

    # 6. PDF and CDF
    pdf_vals = belief.marginal_pdf("frequency", np.array([2.87e9]))
    assert pdf_vals.shape == (1,)
    cdf_vals = belief.marginal_cdf("frequency", np.array([2.87e9]))
    assert cdf_vals.shape == (1,)
    assert 0.0 <= cdf_vals[0] <= 1.0

    # 7. Predicted signal call
    pred = belief(2.87e9)
    assert isinstance(pred, float)


def test_ekf_doptimal_locator_create_and_run():
    exp = _nv_center_experiment()
    rng = random.Random(42)

    # Build the locator
    locator = EKFDOptimalLocator.create(
        max_steps=10,
        parameter_bounds=exp.true_signal.bounds,
        noise_std=0.05,
    )

    assert isinstance(locator, EKFDOptimalLocator)
    assert isinstance(locator.belief, EKFBelief)

    # Run the measurement loop
    steps = list(run_loop(EKFDOptimalLocator, exp, rng, max_steps=15, noise_std=0.05))
    assert len(steps) > 0
    assert len(steps) <= 20

    last_loc = steps[-1]
    assert isinstance(last_loc, EKFDOptimalLocator)
    assert last_loc.belief.last_obs is not None

    # Check finite estimates
    est = last_loc.result()
    assert all(math.isfinite(v) for v in est.values())


def test_ekf_aoptimal_locator_create_and_run():
    exp = _nv_center_experiment()
    rng = random.Random(42)

    # Build the locator
    locator = EKFAOptimalLocator.create(
        max_steps=10,
        parameter_bounds=exp.true_signal.bounds,
        noise_std=0.05,
    )

    assert isinstance(locator, EKFAOptimalLocator)
    assert isinstance(locator.belief, EKFBelief)

    # Run the measurement loop
    steps = list(run_loop(EKFAOptimalLocator, exp, rng, max_steps=15, noise_std=0.05))
    assert len(steps) > 0
    assert len(steps) <= 20

    last_loc = steps[-1]
    assert isinstance(last_loc, EKFAOptimalLocator)
    assert last_loc.belief.last_obs is not None

    # Check finite estimates
    est = last_loc.result()
    assert all(math.isfinite(v) for v in est.values())


def test_ekf_locator_with_priors():
    exp = _nv_center_experiment()
    bounds = dict(exp.true_signal.bounds)

    # Set up priors
    bounds["_priors"] = {
        "frequency": (2.87e9, 10e6),
        "split": (2.1e6, 0.1e6),
        "linewidth": (200e3, 10e3),
        "k_np": (3.0, 0.1),
        "dip_depth": (0.5, 0.05),
    }

    locator = EKFDOptimalLocator.create(
        max_steps=10,
        parameter_bounds=bounds,
        noise_std=0.05,
    )

    # Verify that prior means and variances map correctly
    theta = locator.belief.theta_hat
    P = locator.belief.P

    # frequency: always flat prior (midpoint 2850 MHz, uniform variance over bounds)
    assert np.allclose(theta[0], 2850.0)
    assert np.allclose(P[0, 0], 500.0**2 / 12.0)

    # split: 2.1 MHz
    assert np.allclose(theta[1], 2.1)
    assert np.allclose(P[1, 1], (0.1) ** 2)

    # linewidth (Omega): 0.2 MHz
    assert np.allclose(theta[3], 0.2)
    assert np.allclose(P[3, 3], (0.01) ** 2)

    # k_NP = 1 / k_np = 1/3
    assert np.allclose(theta[4], 1.0 / 3.0)
    # Delta method variance for k_NP = (knp_std / knp_val^2)^2 = (0.1 / 9)^2
    assert np.allclose(P[4, 4], (0.1 / 9.0) ** 2)

    # dip_depth (a): 0.5 * k_NP * Omega^2 = 0.5 * (1/3) * (0.2^2)
    a_expected = 0.5 * (1.0 / 3.0) * (0.2**2)
    assert np.allclose(theta[2], a_expected)
    # Delta method variance: (dd_std * k_NP * Omega^2)^2 = (0.05 * (1/3) * 0.04)^2
    # Clamped to minimum covariance floor of 1e-6 for stability
    assert np.allclose(P[2, 2], max(1e-6, (0.05 * (1.0 / 3.0) * 0.04) ** 2))


def test_ekf_posterior_animation_extraction():
    from nvision.runner.plots import _posterior_animation_inputs, _posterior_animation_inputs_all_params

    # Set up EKF belief
    model = NVCenterLorentzianModel()
    theta_initial = np.array([2870.0, 2.1, 0.05, 5.0, 1.0])
    P_initial = np.eye(5) * 0.1
    bounds = {
        "frequency": (2.6e9, 3.1e9),
        "linewidth": (50e3, 400e3),
        "split": (2.0e6, 3.5e6),
        "k_np": (2.0, 4.0),
        "dip_depth": (0.1, 1.0),
    }
    belief = EKFBelief(
        model=model,
        theta_initial=theta_initial,
        P_initial=P_initial,
        R=0.05**2,
        parameter_bounds=bounds,
    )

    class MockSnapshot:
        def __init__(self, b):
            self.belief = b

    snapshots = [MockSnapshot(belief)]

    # Create RunResult
    class MockRunResult:
        def __init__(self, snaps):
            self.snapshots = snaps

    run_result = MockRunResult(snapshots)

    # 1. 1D posterior animation inputs
    out_1d = _posterior_animation_inputs(run_result, "frequency")
    assert out_1d is not None
    hist, grid = out_1d
    assert len(hist) == 1
    assert hist[0].shape == (250,)
    assert grid.shape == (250,)
    assert np.all(hist[0] >= 0)

    # 2. Multi-parameter posterior animation inputs
    out_multi = _posterior_animation_inputs_all_params(run_result)
    assert out_multi is not None
    assert "frequency" in out_multi
    assert "linewidth" in out_multi
    assert "split" in out_multi

    hist_f, grid_f = out_multi["frequency"]
    assert len(hist_f) == 1
    assert hist_f[0].shape == (250,)
    assert grid_f.shape == (250,)


def test_ekf_particle_frequency_posterior_animation_extraction():
    from nvision.runner.plots import _posterior_animation_inputs, _posterior_animation_inputs_all_params
    from nvision.sim.locs.ekf.belief import EKFParticleFrequencyBelief

    model = NVCenterLorentzianModel()
    theta_initial = np.array([2870.0, 2.1, 0.05, 5.0, 1.0])
    P_initial = np.eye(5) * 0.1
    bounds = {
        "frequency": (2.6e9, 3.1e9),
        "linewidth": (50e3, 400e3),
        "split": (2.0e6, 3.5e6),
        "k_np": (2.0, 4.0),
        "dip_depth": (0.1, 1.0),
    }
    belief = EKFParticleFrequencyBelief(
        model=model,
        theta_initial=theta_initial,
        P_initial=P_initial,
        R=0.05**2,
        parameter_bounds=bounds,
        num_particles=100,
    )

    class MockSnapshot:
        def __init__(self, b):
            self.belief = b

    snapshots = [MockSnapshot(belief)]

    class MockRunResult:
        def __init__(self, snaps):
            self.snapshots = snaps

    run_result = MockRunResult(snapshots)

    # 1. 1D frequency extraction -> should be particles
    out_1d_freq = _posterior_animation_inputs(run_result, "frequency")
    assert out_1d_freq is not None
    hist, grid = out_1d_freq
    assert len(hist) == 1
    assert hist[0].shape == (100, 2)
    assert grid.shape == (2,)

    # 2. 1D linewidth extraction -> should be continuous (250 grid points)
    out_1d_lw = _posterior_animation_inputs(run_result, "linewidth")
    assert out_1d_lw is not None
    hist_lw, grid_lw = out_1d_lw
    assert len(hist_lw) == 1
    assert hist_lw[0].shape == (250,)
    assert grid_lw.shape == (250,)

    # 3. Multi-parameter extraction
    out_multi = _posterior_animation_inputs_all_params(run_result)
    assert out_multi is not None
    assert "frequency" in out_multi
    assert "linewidth" in out_multi
    assert "split" in out_multi

    hist_f, grid_f = out_multi["frequency"]
    assert len(hist_f) == 1
    assert hist_f[0].shape == (100, 2)

    hist_lw, grid_lw = out_multi["linewidth"]
    assert len(hist_lw) == 1
    assert hist_lw[0].shape == (250,)

