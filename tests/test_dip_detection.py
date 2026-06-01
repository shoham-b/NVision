"""Unit tests for robust dip detection and noise estimation."""

from __future__ import annotations

import numpy as np
import pytest

from nvision import nv_center_smc_belief
from nvision.sim.locs.bayesian.dip_detection import identify_dip_candidates


def test_identify_dip_candidates_empty():
    # Empty inputs
    assert identify_dip_candidates(np.array([]), np.array([]), 0.02, 5e6) == []


def test_identify_dip_candidates_no_dips():
    # Signals all around background level (1.0), with noise std of 0.02, min cluster count 2.
    obs_xs = np.linspace(2.8e9, 2.9e9, 10)
    obs_ys = np.ones(10) + np.random.uniform(-0.01, 0.01, 10)
    # The threshold will be background - max(3 * 0.02, 0.01) = 1.0 - 0.06 = 0.94.
    # All signals are ~1.0, so no dips are found.
    assert identify_dip_candidates(obs_xs, obs_ys, 0.02, 5e6, n_sigma=3.0, min_cluster_count=2) == []


def test_identify_dip_candidates_single_isolated_spike():
    # A single isolated point below threshold.
    # With min_cluster_count=2, it should be filtered out.
    obs_xs = np.linspace(2.8e9, 2.9e9, 10)
    obs_ys = np.ones(10)
    obs_ys[5] = 0.8  # Deep dip at index 5, but only 1 point.
    # Threshold background is ~1.0. dip_thresh = 1.0 - 0.06 = 0.94.
    # The point at index 5 is 0.8 < 0.94, but it is alone.
    assert identify_dip_candidates(obs_xs, obs_ys, 0.02, 5e6, n_sigma=3.0, min_cluster_count=2) == []


def test_identify_dip_candidates_cluster_detection():
    # Two points close to each other below threshold.
    # Should be detected as a cluster, and return its signal-depth-weighted centroid.
    obs_xs = np.array([2.8e9, 2.81e9, 2.82e9, 2.822e9, 2.824e9, 2.83e9, 2.84e9, 2.85e9, 2.86e9, 2.87e9])
    obs_ys = np.array([1.0, 1.0, 0.8, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0])
    # Points at 2.82e9, 2.822e9, 2.824e9 are below 0.94.
    # They are within 3 * 5 MHz = 15 MHz of each other, so they form a single cluster.
    # Depth weights: background = 1.0.
    # weights: 1.0 - 0.8 = 0.2, 1.0 - 0.7 = 0.3, 1.0 - 0.8 = 0.2.
    # centroid = (2.82e9 * 0.2 + 2.822e9 * 0.3 + 2.824e9 * 0.2) / 0.7 = 2.822e9.
    centroids = identify_dip_candidates(obs_xs, obs_ys, 0.02, 5e6, n_sigma=3.0, min_cluster_count=2)
    assert len(centroids) == 1
    assert np.isclose(centroids[0].centroid_hz, 2.822e9)


def test_identify_dip_candidates_per_particle_voting():
    obs_xs = np.array([2.8e9, 2.802e9, 2.9e9, 2.91e9, 2.92e9, 2.93e9])
    obs_ys = np.array([0.7, 0.7, 1.0, 1.0, 1.0, 1.0])
    
    # 2 particles:
    # Particle 0: sigma = 0.01, weight = 0.8
    # Particle 1: sigma = 0.15, weight = 0.2
    per_particle_sigmas = np.array([0.01, 0.15])
    particle_weights = np.array([0.8, 0.2])
    
    candidates = identify_dip_candidates(
        obs_xs, obs_ys, noise_std=0.05, max_linewidth_hz=5e6,
        per_particle_sigmas=per_particle_sigmas,
        particle_weights=particle_weights,
        min_cluster_count=2,
        confidence_threshold=0.0,
    )
    assert len(candidates) == 1
    assert np.isclose(candidates[0].significance, 1.6)


def test_identify_dip_candidates_binomial_confidence_gate():
    obs_xs = np.array([2.8e9, 2.802e9, 2.9e9, 2.91e9, 2.92e9, 2.93e9, 2.94e9, 2.95e9, 2.96e9, 2.97e9])
    obs_ys = np.array([0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # 2 particles:
    # Particle 0: sigma = 0.01, weight = 0.1
    # Particle 1: sigma = 0.04, weight = 0.9
    per_particle_sigmas = np.array([0.01, 0.04])
    particle_weights = np.array([0.1, 0.9])
    
    cands_gated = identify_dip_candidates(
        obs_xs, obs_ys, noise_std=0.03, max_linewidth_hz=5e6,
        per_particle_sigmas=per_particle_sigmas,
        particle_weights=particle_weights,
        min_cluster_count=2,
        confidence_threshold=0.99,
    )
    assert cands_gated == []
    
    cands_ungated = identify_dip_candidates(
        obs_xs, obs_ys, noise_std=0.03, max_linewidth_hz=5e6,
        per_particle_sigmas=per_particle_sigmas,
        particle_weights=particle_weights,
        min_cluster_count=2,
        confidence_threshold=0.0,
    )
    assert len(cands_ungated) == 1


def test_identify_dip_candidates_single_dip_prior():
    obs_xs = np.array([2.82e9, 2.822e9, 2.92e9, 2.922e9, 2.85e9, 2.86e9, 2.87e9, 2.88e9])
    obs_ys = np.array([0.7, 0.7, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0])
    
    per_particle_sigmas = np.array([0.01, 0.05])
    particle_weights = np.array([0.8, 0.2])
    
    cands_gated = identify_dip_candidates(
        obs_xs, obs_ys, noise_std=0.03, max_linewidth_hz=5e6,
        per_particle_sigmas=per_particle_sigmas,
        particle_weights=particle_weights,
        min_cluster_count=2,
        confidence_threshold=0.0,
        max_split_hz=10e6,
    )
    assert len(cands_gated) == 1
    assert np.isclose(cands_gated[0].centroid_hz, 2.821e9)
    
    cands_ungated = identify_dip_candidates(
        obs_xs, obs_ys, noise_std=0.03, max_linewidth_hz=5e6,
        per_particle_sigmas=per_particle_sigmas,
        particle_weights=particle_weights,
        min_cluster_count=2,
        confidence_threshold=0.0,
        max_split_hz=150e6,
    )
    assert len(cands_ungated) == 2


def test_identify_dip_candidates_uncertainty_gating():
    # Setup observations with a valid dip cluster
    obs_xs = np.array([2.8e9, 2.81e9, 2.82e9, 2.822e9, 2.824e9, 2.83e9, 2.84e9, 2.85e9, 2.86e9, 2.87e9])
    obs_ys = np.array([1.0, 1.0, 0.8, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Case A: High relative uncertainty (noise_std_unc = 0.01, noise_std = 0.05 => 20% relative unc).
    # Since 20% >= 15% (default threshold), it should gate the dip and return []
    centroids_gated = identify_dip_candidates(
        obs_xs, obs_ys, noise_std=0.05, max_linewidth_hz=5e6, noise_std_unc=0.01
    )
    assert centroids_gated == []

    # Case B: Low relative uncertainty (noise_std_unc = 0.002, noise_std = 0.05 => 4% relative unc).
    # Since 4% < 15%, it should successfully return the cluster centroid
    centroids_ungated = identify_dip_candidates(
        obs_xs, obs_ys, noise_std=0.05, max_linewidth_hz=5e6, noise_std_unc=0.002
    )
    assert len(centroids_ungated) == 1
    assert np.isclose(centroids_ungated[0].centroid_hz, 2.822e9)



def test_noise_std_uncertainty_unconfigured_raises_error():
    # Fresh SMC belief without noise model should raise ValueError
    b = nv_center_smc_belief(num_particles=200)
    with pytest.raises(ValueError, match="no active noise model"):
        b.noise_std_uncertainty()


def test_noise_std_uncertainty_configured():
    # SMC belief with noise model should return a valid float
    from nvision.spectra.noise_model import GaussianNoiseSignalModel
    noise_model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.1)})
    b = nv_center_smc_belief(num_particles=200, noise_model=noise_model)
    b_unc = b.noise_std_uncertainty()
    assert isinstance(b_unc, float)
    assert b_unc >= 0.0


def test_noise_sigma_updates_every_update_but_dip_detection_only_upon_resampling():
    from nvision.spectra.noise_model import GaussianNoiseSignalModel
    from nvision.models.observation import Observation
    noise_model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.1)})
    b = nv_center_smc_belief(num_particles=200, noise_model=noise_model)
    b.auto_resample = False
    
    initial_noise_std = b.estimated_noise_std()
    
    # 1. Update with a high-noise observation.
    # The noise estimate MUST change immediately because noise is updated on every update().
    # However, dip centers must remain empty because dip detection only runs upon resampling.
    obs = Observation(x=0.5, signal_value=0.5)
    b.update(obs)
    
    post_update_noise_std = b.estimated_noise_std()
    assert not np.isclose(initial_noise_std, post_update_noise_std)
    assert getattr(b, "_dip_centers", []) == []


def test_estimated_noise_std_unconfigured_raises_error():
    # Fresh SMC belief without noise model should raise ValueError
    b = nv_center_smc_belief(num_particles=200)
    with pytest.raises(ValueError, match="no active noise model"):
        b.estimated_noise_std()


def test_estimated_noise_std_configured():
    # SMC belief with noise model should return a valid float
    from nvision.spectra.noise_model import GaussianNoiseSignalModel
    noise_model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.1)})
    b = nv_center_smc_belief(num_particles=200, noise_model=noise_model)
    noise_std = b.estimated_noise_std()
    assert isinstance(noise_std, float)
    assert 0.01 <= noise_std <= 0.1


def test_estimated_noise_std_is_90th_percentile():

    from nvision.spectra.noise_model import GaussianNoiseSignalModel
    noise_model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.1)})
    b = nv_center_smc_belief(num_particles=10, noise_model=noise_model)
    
    if getattr(b, "_use_rao_blackwell_noise", False):
        b._noise_alphas = np.full(10, 9.5, dtype=np.float32)
        sigmas = np.linspace(0.01, 0.10, 10)
        b._noise_betas = (sigmas ** 2 * 10.0).astype(np.float32)
        b._weights = np.full(10, 0.10, dtype=np.float32)
        
        est = b.estimated_noise_std()
        assert np.isclose(est, 0.09)




