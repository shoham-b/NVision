"""Tests for Hybrid Noise-Aware Signal Modeling."""

import numpy as np

from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.models.observation import Observation
from nvision.spectra.noise_model import (
    CompositeNoiseSignalModel,
    DriftNoiseSignalModel,
    GaussianNoiseSignalModel,
    PoissonNoiseSignalModel,
)
from nvision.spectra.nv_center import NVCenterLorentzianModel
from nvision.spectra.unit_cube import UnitCubeSignalModel


def test_gaussian_noise_likelihood():
    """Verify Gaussian noise likelihood with epistemic broadening."""
    model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.1)})

    # residuals (y_obs - mu)
    residuals = np.array([0.0, 0.1])
    # noise_sigma particles
    noise_params = [np.array([0.05, 0.05])]
    predicted = np.array([1.0, 1.0])

    # 1. No epistemic uncertainty
    ll_pure = model.composite_log_likelihood(predicted, residuals, noise_params, sigma_epistemic=0.0)
    # ll = -0.5*(r/s)^2 - log(s)
    # r=0: -log(0.05) ~= 2.99
    # r=0.1: -0.5*(0.1/0.05)^2 - log(0.05) = -2 + 2.99 = 0.99
    assert ll_pure[0] > ll_pure[1]

    # 2. Large epistemic uncertainty (should broaden/flatten the likelihood)
    # Crossover for 0.05 -> 0.5 is at r ~= 0.107. Use r=0.2 to see broadening.
    residuals_large = np.array([0.2, 0.2])
    ll_pure_large = model.composite_log_likelihood(predicted, residuals_large, noise_params, sigma_epistemic=0.0)
    ll_broad_large = model.composite_log_likelihood(predicted, residuals_large, noise_params, sigma_epistemic=0.5)
    assert ll_broad_large[1] > ll_pure_large[1]  # Large residuals become "less unlikely"

def test_poisson_noise_likelihood():
    """Verify Poisson scale-based likelihood."""
    model = PoissonNoiseSignalModel(prior_bounds={"poisson_scale": (10, 1000)})

    # mu=1.0, scale=100 -> lambda=100
    # obs_y = 1.1 (residuals = 0.1) -> k = 110
    predicted = np.array([1.0, 1.0])
    residuals = np.array([0.1, 0.1])
    noise_params = [np.array([100.0, 100.0])]

    ll = model.composite_log_likelihood(predicted, residuals, noise_params, sigma_epistemic=0.0)
    assert not np.isnan(ll).any()
    assert ll[0] == ll[1]

def test_composite_noise_model():
    """Verify combining multiple noise sources."""
    g_model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.1)})
    d_model = DriftNoiseSignalModel(prior_bounds={"drift_rate": (0.0, 0.1)})

    comp = CompositeNoiseSignalModel([g_model, d_model])
    assert "noise_sigma" in comp.spec.names
    assert "drift_rate" in comp.spec.names
    assert comp.spec.dim == 2

    predicted = np.array([1.0])
    residuals = np.array([0.05])
    noise_params = [np.array([0.05]), np.array([0.01])]

    ll = comp.composite_log_likelihood(predicted, residuals, noise_params, sigma_epistemic=0.0)
    assert len(ll) == 1

def test_smc_joint_parameter_tracking():
    """Integration test: Verify SMC tracks joint parameters correctly."""
    from nvision.spectra.nv_center import nv_center_lorentzian_bounds_for_domain
    sig_model = NVCenterLorentzianModel()
    noise_model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.1)})

    # Get bounds
    bounds = nv_center_lorentzian_bounds_for_domain(2.8e9, 2.9e9)
    bounds.update(noise_model.spec.bounds)

    # Unit cube wrapper
    wrapped = UnitCubeSignalModel(sig_model, bounds, x_bounds_phys=(2.8e9, 2.9e9))

    belief = UnitCubeSMCMarginalDistribution(
        model=wrapped,
        noise_model=noise_model,
        parameter_bounds={name: (0.0, 1.0) for name in bounds},
        physical_param_bounds=bounds,
        num_particles=100
    )

    assert "noise_sigma" in belief._param_names
    assert belief._noise_param_slice is not None
    # noise_sigma should be the last parameter
    assert belief._param_names[-1] == "noise_sigma"
    assert belief._noise_param_slice.start == len(belief._param_names) - 1

    # Perform a dummy update
    obs = Observation(x=0.5, signal_value=0.9, noise_std=0.05)
    belief.update(obs)

    # Check that particles evolved
    assert belief._particles.shape == (100, len(belief._param_names))
    assert not np.allclose(belief._particles, 0.5) # Assuming they jittered
