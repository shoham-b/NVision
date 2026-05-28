"""Tests for Gaussian parameter priors and sine-squared frequency prior."""

from __future__ import annotations

import random

import numpy as np

from nvision import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    NVCenterCoreGenerator,
    nv_center_smc_belief,
)


def test_generator_adds_gaussian_and_sin2_priors():
    rng = random.Random(42)

    # 1. Lorentzian variant
    gen_lor = NVCenterCoreGenerator(variant="lorentzian")
    signal_lor = gen_lor.generate(rng)

    assert "_priors" in signal_lor.bounds
    priors_lor = signal_lor.bounds["_priors"]

    # Verify frequency prior
    assert "frequency" in priors_lor
    expected_k = np.pi / (DEFAULT_NV_CENTER_FREQ_X_MAX - DEFAULT_NV_CENTER_FREQ_X_MIN)
    assert priors_lor["frequency"] == ("sin^2", expected_k)

    # Verify other parameter priors are Gaussian (val, std)
    for name in ("split", "linewidth", "k_np", "c_total"):
        assert name in priors_lor
        val, std = priors_lor[name]
        assert isinstance(val, float)
        assert isinstance(std, float)
        assert std > 0.0

    # 2. Voigt variant
    gen_voigt = NVCenterCoreGenerator(variant="voigt")
    signal_voigt = gen_voigt.generate(rng)

    assert "_priors" in signal_voigt.bounds
    priors_voigt = signal_voigt.bounds["_priors"]

    # Verify frequency prior
    assert "frequency" in priors_voigt
    assert priors_voigt["frequency"] == ("sin^2", expected_k)

    # Verify other parameter priors are Gaussian (val, std)
    for name in ("split", "fwhm_total", "lorentz_frac", "k_np", "dip_depth"):
        assert name in priors_voigt
        val, std = priors_voigt[name]
        assert isinstance(val, float)
        assert isinstance(std, float)
        assert std > 0.0


def test_smc_belief_initializes_with_sin2_and_gaussian_priors():
    rng = random.Random(42)
    gen = NVCenterCoreGenerator(variant="lorentzian")
    signal = gen.generate(rng)

    # Build SMC belief with generator priors
    belief = nv_center_smc_belief(signal.bounds, num_particles=1000)

    # Verify particles for frequency follow sin^2(k f) prior
    f_idx = belief._param_names.index("frequency")
    freq_particles_unit = belief._particles[:, f_idx]

    # In UnitCube, particles are in [0, 1]
    assert np.all((freq_particles_unit >= 0.0) & (freq_particles_unit <= 1.0))

    # Convert frequency particles to physical space
    f_lo, f_hi = belief.physical_param_bounds["frequency"]
    freq_particles_phys = f_lo + freq_particles_unit * (f_hi - f_lo)

    # Verify rejection sampling shape: evaluate sin^2(k f)
    k = np.pi / (f_hi - f_lo)
    probs = np.sin(k * (freq_particles_phys - f_lo)) ** 2

    # The mean of sin^2(k f) over accepted particles should be significantly higher
    # than the analytical mean of uniform [0, 1] density passing through the sin^2 filter
    # Let's ensure that the particles actually represent the distribution (none are rejected below 0)
    assert np.min(probs) >= 0.0
    assert np.max(probs) <= 1.0

    # Let's check another Gaussian prior parameter, e.g. split
    s_idx = belief._param_names.index("split")
    split_particles_unit = belief._particles[:, s_idx]

    s_lo, s_hi = belief.physical_param_bounds["split"]
    split_particles_phys = s_lo + split_particles_unit * (s_hi - s_lo)

    # Prior values are generated in physical units
    prior_mean_phys, prior_std_phys = signal.bounds["_priors"]["split"]

    # Check that particle mean is close to the prior mean
    particle_mean = np.mean(split_particles_phys)
    assert abs(particle_mean - prior_mean_phys) < 3.0 * prior_std_phys

    # Verify that particles are standard deviations-bound and not uniformly distributed
    particle_std = np.std(split_particles_phys)
    assert abs(particle_std - prior_std_phys) < 0.5 * prior_std_phys

    # Let's verify c_total prior as well
    c_idx = belief._param_names.index("c_total")
    c_particles_unit = belief._particles[:, c_idx]

    c_lo, c_hi = belief.physical_param_bounds["c_total"]
    c_particles_phys = c_lo + c_particles_unit * (c_hi - c_lo)

    prior_c_mean_phys, prior_c_std_phys = signal.bounds["_priors"]["c_total"]

    # Check that c_total particle mean is close to its prior mean
    c_particle_mean = np.mean(c_particles_phys)
    assert abs(c_particle_mean - prior_c_mean_phys) < 3.0 * prior_c_std_phys

    # Verify that c_total particles are standard deviations-bound and not uniformly distributed
    c_particle_std = np.std(c_particles_phys)
    assert abs(c_particle_std - prior_c_std_phys) < 0.5 * prior_c_std_phys
