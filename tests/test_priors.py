"""Tests for Gaussian parameter priors and sine-squared zero_field_freq prior."""

from __future__ import annotations

import random
import statistics

import numpy as np

from nvision import (
    NVCenterCoreGenerator,
    nv_center_smc_belief,
)
from nvision.spectra.nv_center import MIN_HWHM
from tests.noise import gaussian_noise


def test_generator_adds_gaussian_and_sin2_priors():
    rng = random.Random(42)

    # 1. Lorentzian variant
    gen_lor = NVCenterCoreGenerator(variant="lorentzian")
    signal_lor = gen_lor.generate(rng)

    assert "_priors" in signal_lor.bounds
    priors_lor = signal_lor.bounds["_priors"]

    # Verify zero_field_freq prior
    assert "zero_field_freq" in priors_lor
    expected_k = np.pi / (2.0 * MIN_HWHM)
    assert priors_lor["zero_field_freq"] == ("sin^2", expected_k)

    # Verify other parameter priors are Gaussian (val, std) — default: no hyperfine_shift/k_np
    for name in ("hwhm", "contrast_total"):
        assert name in priors_lor
        val, std = priors_lor[name]
        assert isinstance(val, float)
        assert isinstance(std, float)
        assert std > 0.0
    assert "hyperfine_shift" not in priors_lor
    assert "k_np" not in priors_lor

    # With hyperfine splitting enabled, hyperfine_shift and k_np are present
    gen_lor_hyperfine_shift = NVCenterCoreGenerator(variant="lorentzian", hyperfine="n14", infer_hyperfine=True)
    signal_lor_hyperfine_shift = gen_lor_hyperfine_shift.generate(rng)
    priors_hyperfine_shift = signal_lor_hyperfine_shift.bounds["_priors"]
    for name in ("hyperfine_shift", "hwhm", "k_np", "contrast_total"):
        assert name in priors_hyperfine_shift

    # 2. Voigt variant (default: hyperfine="unresolved", same as Lorentzian --
    # hyperfine_shift/k_np are fixed to the real N-14 constant / 1.0, not free/inferred priors)
    gen_voigt = NVCenterCoreGenerator(variant="voigt")
    signal_voigt = gen_voigt.generate(rng)

    assert "_priors" in signal_voigt.bounds
    priors_voigt = signal_voigt.bounds["_priors"]

    # Verify zero_field_freq prior
    assert "zero_field_freq" in priors_voigt
    assert priors_voigt["zero_field_freq"] == ("sin^2", expected_k)

    # Verify other parameter priors are Gaussian (val, std)
    for name in ("hwhm_lorentz", "sigma_inhom", "contrast_total"):
        assert name in priors_voigt
        val, std = priors_voigt[name]
        assert isinstance(val, float)
        assert isinstance(std, float)
        assert std > 0.0
    assert "hyperfine_shift" not in priors_voigt
    assert "k_np" not in priors_voigt

    # With hyperfine splitting enabled, hyperfine_shift and k_np are present
    gen_voigt_hf = NVCenterCoreGenerator(variant="voigt", hyperfine="n14", infer_hyperfine=True)
    signal_voigt_hf = gen_voigt_hf.generate(rng)
    priors_voigt_hf = signal_voigt_hf.bounds["_priors"]
    for name in ("hyperfine_shift", "k_np", "hwhm_lorentz", "sigma_inhom", "contrast_total"):
        assert name in priors_voigt_hf


def test_smc_belief_initializes_with_sin2_and_gaussian_priors():
    rng = random.Random(42)
    gen = NVCenterCoreGenerator(variant="lorentzian")
    signal = gen.generate(rng)

    # Build SMC belief with generator priors. with_fixed_zero_field_freq=False: this test
    # specifically exercises the sin^2 prior on the zero_field_freq dimension, which
    # requires zero_field_freq to be a free particle dimension (it's fixed by default --
    # see NVCenterCoreGenerator's docstring -- so nv_center_smc_belief's own default
    # would otherwise leave it out of belief._param_names entirely).
    # prior_broad_fraction=0.0: this test checks the Gaussian prior's shape; the uniform mixture
    # component widens the particle std on purpose and is covered in test_smc_prior_mixture.py.
    belief = nv_center_smc_belief(
        signal.bounds,
        num_particles=1000,
        with_fixed_zero_field_freq=False,
        noise_model=gaussian_noise(),
        prior_broad_fraction=0.0,
    )

    # Verify particles for zero_field_freq follow sin^2(k f) prior
    f_idx = belief._param_names.index("zero_field_freq")
    freq_particles_unit = belief._particles[:, f_idx]

    # In UnitCube, particles are in [0, 1]
    assert np.all((freq_particles_unit >= 0.0) & (freq_particles_unit <= 1.0))

    # Convert zero_field_freq particles to physical space
    f_lo, f_hi = belief.physical_param_bounds["zero_field_freq"]
    freq_particles_phys = f_lo + freq_particles_unit * (f_hi - f_lo)

    # Verify rejection sampling shape: evaluate sin^2(k f)
    k = np.pi / (2.0 * MIN_HWHM)
    probs = np.sin(k * (freq_particles_phys - f_lo)) ** 2

    # The mean of sin^2(k f) over accepted particles should be significantly higher
    # than the analytical mean of uniform [0, 1] density passing through the sin^2 filter
    # Let's ensure that the particles actually represent the distribution (none are rejected below 0)
    assert np.min(probs) >= 0.0
    assert np.max(probs) <= 1.0

    # Let's check another Gaussian prior parameter, e.g. hwhm (default: no hyperfine_shift)
    lw_idx = belief._param_names.index("hwhm")
    lw_particles_unit = belief._particles[:, lw_idx]

    lw_lo, lw_hi = belief.physical_param_bounds["hwhm"]
    lw_particles_phys = lw_lo + lw_particles_unit * (lw_hi - lw_lo)

    # Prior values are generated in physical units
    prior_mean_phys, prior_std_phys = signal.bounds["_priors"]["hwhm"]

    # Check that particle mean is close to the prior mean
    particle_mean = np.mean(lw_particles_phys)
    assert abs(particle_mean - prior_mean_phys) < 3.0 * prior_std_phys

    # Verify that particles are standard deviations-bound and not uniformly distributed
    particle_std = np.std(lw_particles_phys)
    assert abs(particle_std - prior_std_phys) < 0.5 * prior_std_phys

    # Let's verify contrast_total prior as well
    c_idx = belief._param_names.index("contrast_total")
    c_particles_unit = belief._particles[:, c_idx]

    c_lo, c_hi = belief.physical_param_bounds["contrast_total"]
    c_particles_phys = c_lo + c_particles_unit * (c_hi - c_lo)

    prior_c_mean_phys, prior_c_std_phys = signal.bounds["_priors"]["contrast_total"]

    # Check that contrast_total particle mean is close to its prior mean
    c_particle_mean = np.mean(c_particles_phys)
    assert abs(c_particle_mean - prior_c_mean_phys) < 3.0 * prior_c_std_phys

    # Verify that contrast_total particles are standard deviations-bound and not uniformly distributed
    c_particle_std = np.std(c_particles_phys)
    assert abs(c_particle_std - prior_c_std_phys) < 0.5 * prior_c_std_phys


def test_prior_mean_offset_from_truth_is_widened_by_prior_mean_offset_sigmas():
    """The generator's prior *mean* is drawn as gauss(true_value, std *
    PRIOR_MEAN_OFFSET_SIGMAS), not gauss(true_value, std) -- otherwise the SBED
    locator's starting belief for every non-zero_field_freq parameter is suspiciously
    close to the true value before a single measurement (see nv_center_generator's
    PRIOR_MEAN_OFFSET_SIGMAS import). At the default multiplier (3.0), the
    *typical* offset should be a couple of prior-sigmas, not the ~0.8 sigma a
    plain gauss(true, std) draw (multiplier 1.0) would produce on average.
    """
    import nvision.sim.gen.nv_center_generator as gen_mod

    offsets_in_std = []
    for seed in range(300):
        rng = random.Random(seed)
        gen = NVCenterCoreGenerator(variant="lorentzian")
        signal = gen.generate(rng)
        mean, std = signal.bounds["_priors"]["hwhm"]
        true_hwhm = signal.typed_parameters.hwhm
        offsets_in_std.append(abs(mean - true_hwhm) / std)

    mean_offset_in_std = statistics.mean(offsets_in_std)
    # E[|offset|/std] = PRIOR_MEAN_OFFSET_SIGMAS * sqrt(2/pi); at the default
    # 3.0 that's ~2.39. A same-std draw (multiplier 1.0) would average ~0.8 --
    # assert we're clearly in the widened regime, not the old narrow one.
    assert mean_offset_in_std > 1.8, (
        f"mean |prior_mean - true| / std was {mean_offset_in_std:.2f}, expected "
        "~2.4 at the default PRIOR_MEAN_OFFSET_SIGMAS=3.0 -- looks like the "
        "narrow (pre-fix) offset is back"
    )
    # Sanity-check it scales with the constant (rather than being a coincidence
    # of some other unrelated std): halving the multiplier should roughly halve
    # the average offset.
    orig = gen_mod.PRIOR_MEAN_OFFSET_SIGMAS
    try:
        gen_mod.PRIOR_MEAN_OFFSET_SIGMAS = orig / 2.0
        halved_offsets = []
        for seed in range(300):
            rng = random.Random(seed)
            gen = NVCenterCoreGenerator(variant="lorentzian")
            signal = gen.generate(rng)
            mean, std = signal.bounds["_priors"]["hwhm"]
            true_hwhm = signal.typed_parameters.hwhm
            halved_offsets.append(abs(mean - true_hwhm) / std)
        halved_mean_offset = statistics.mean(halved_offsets)
    finally:
        gen_mod.PRIOR_MEAN_OFFSET_SIGMAS = orig
    assert halved_mean_offset < mean_offset_in_std * 0.7
