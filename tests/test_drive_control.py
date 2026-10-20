"""Drive-controlled NV models (CW and pulsed), drive plumbing, and the drive-choosing SBED locator."""

from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest

from nvision.models.experiment import CoreExperiment
from nvision.models.observation import Observation, aggregate_shots
from nvision.sim import presets
from nvision.sim.combinations import CombinationGrid, _parse_noise
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.sbed_drive_locator import DriveChoosingSBEDLocator, dominant_mode
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.spectra.drive import DRIVE_AMP_MIN_UNIT, DriveSetting, compute_vectorized_many_driven
from nvision.spectra.nv_center import (
    NV_ZERO_FIELD_SPLITTING_HZ,
    NVCenterSaturationVoigtModel,
    NVCenterSaturationVoigtZeemanSpectrum,
)
from nvision.spectra.nv_center_drive import (
    NVCenterCWDriveModel,
    NVCenterCWDriveZeemanSpectrum,
    NVCenterPulsedDriveModel,
    NVCenterPulsedZeemanSpectrum,
    pi_pulse_duration_s,
    rabi_transition_probability,
)
from tests.noise import gaussian_noise

D = NV_ZERO_FIELD_SPLITTING_HZ


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_cw_drive_matches_saturation_voigt_at_equivalent_saturation():
    """At amplitude a, the CW model equals saturation-Voigt with s = (a / drive_amp_sat)**2."""
    cw = NVCenterCWDriveModel()
    sv = NVCenterSaturationVoigtModel(with_zeeman_splitting=True)
    params = NVCenterCWDriveZeemanSpectrum(zero_field_freq=D, drive_amp_sat=0.2, sigma_inhom=1e5, zeeman_shift=30e6)
    a = 0.4
    sv_params = NVCenterSaturationVoigtZeemanSpectrum(
        zero_field_freq=D, saturation=(a / 0.2) ** 2, sigma_inhom=1e5, zeeman_shift=30e6
    )
    view = cw.at_drive(DriveSetting(a))
    for x in (D + 29e6, D + 30e6, D + 31e6, D + 50e6):
        assert view.compute(x, params) == pytest.approx(sv.compute(x, sv_params), rel=1e-6)


def test_cw_higher_drive_is_wider_and_deeper():
    """The search lever: more amplitude power-broadens and deepens the dip."""
    cw = NVCenterCWDriveModel()
    params = NVCenterCWDriveZeemanSpectrum(zero_field_freq=D, drive_amp_sat=0.1, sigma_inhom=0.0, zeeman_shift=30e6)
    xs = np.linspace(D + 25e6, D + 35e6, 2001)
    weak = cw.at_drive(DriveSetting(0.05)).compute_many_float64(xs, params)
    strong = cw.at_drive(DriveSetting(1.0)).compute_many_float64(xs, params)
    assert strong.min() < weak.min()  # deeper

    def fwhm(ys):
        half = 1.0 - 0.5 * (1.0 - ys.min())
        below = xs[ys < half]
        return below.max() - below.min()

    assert fwhm(strong) > 3.0 * fwhm(weak)  # much wider


def test_pi_pulse_resonant_flip_and_width_scales_with_rabi():
    rabi = 2e6
    t = pi_pulse_duration_s(rabi)
    assert rabi_transition_probability(0.0, rabi, t) == pytest.approx(1.0)
    # Half-maximum detuning ~0.8 * Omega for a pi pulse, so the width scales with Omega.
    for omega in (1e6, 4e6):
        tt = pi_pulse_duration_s(omega)
        assert rabi_transition_probability(0.7988 * omega, omega, tt) == pytest.approx(0.5, abs=0.01)


def test_pulsed_calibration_aliases_on_resonance_but_not_off_resonance():
    """Ratios r and 2 - r of true/estimated Rabi give the same depth on resonance, different off it."""
    t = pi_pulse_duration_s(2e6)  # calibrated for 2 MHz
    r_lo, r_hi = 0.7, 1.3
    on_lo = rabi_transition_probability(0.0, r_lo * 2e6, t)
    on_hi = rabi_transition_probability(0.0, r_hi * 2e6, t)
    assert on_lo == pytest.approx(on_hi, rel=1e-9)
    off_lo = rabi_transition_probability(2e6, r_lo * 2e6, t)
    off_hi = rabi_transition_probability(2e6, r_hi * 2e6, t)
    assert abs(off_lo - off_hi) > 0.05


def test_pulsed_vectorized_matches_scalar():
    model = NVCenterPulsedDriveModel()
    drive = DriveSetting(0.5, pi_pulse_duration_s(3e6 * 0.5))
    view = model.at_drive(drive)
    params = NVCenterPulsedZeemanSpectrum(
        zero_field_freq=D, rabi_per_amp=3e6, contrast_total=0.2, sigma_inhom=2e5, zeeman_shift=30e6
    )
    xs = np.linspace(D + 25e6, D + 35e6, 11)
    samples = [np.full(3, v, dtype=np.float32) for v in (3e6, 0.2, 2e5, 30e6)]
    many = view.compute_vectorized_many(xs, samples)
    assert many.shape == (11, 3)
    scalar = [view.compute(float(x), params) for x in xs]
    np.testing.assert_allclose(many[:, 0], scalar, rtol=1e-5)
    # Each Zeeman group carries half of contrast_total (same convention as the CW kernels): depth 0.1 here.
    assert min(scalar) < 1.0 - 0.09


def test_drive_controlled_model_refuses_direct_evaluation():
    model = NVCenterCWDriveModel()
    params = NVCenterCWDriveZeemanSpectrum(zero_field_freq=D, drive_amp_sat=0.2, sigma_inhom=0.0, zeeman_shift=30e6)
    with pytest.raises(ValueError, match="drive-controlled"):
        model.compute(D + 30e6, params)
    with pytest.raises(ValueError, match="pulse_duration_s"):
        NVCenterPulsedDriveModel().at_drive(DriveSetting(0.5))
    with pytest.raises(ValueError, match="CW"):
        model.at_drive(DriveSetting(0.5, 1e-7))


def test_driven_evaluation_groups_by_drive():
    model = NVCenterCWDriveModel()
    xs = np.array([D + 30e6, D + 30e6, D + 31e6])
    samples = [np.array([0.2, 0.1], dtype=np.float32), np.zeros(2, dtype=np.float32), np.full(2, 30e6, np.float32)]
    drives = [DriveSetting(0.2), DriveSetting(1.0), DriveSetting(0.2)]
    out = compute_vectorized_many_driven(model, xs, samples, drives)
    assert out.shape == (3, 2)
    expected_row1 = model.at_drive(DriveSetting(1.0)).compute_vectorized_many(xs[1:2], samples)
    np.testing.assert_allclose(out[1], expected_row1[0])
    with pytest.raises(ValueError, match="no drive"):
        compute_vectorized_many_driven(model, xs, samples, [None, None, None])


def test_drive_setting_validates_range():
    with pytest.raises(ValueError, match="drive_amp_unit"):
        DriveSetting(0.0)
    with pytest.raises(ValueError, match="drive_amp_unit"):
        DriveSetting(1.5)
    with pytest.raises(ValueError, match="pulse_duration_s"):
        DriveSetting(0.5, -1.0)
    assert DriveSetting(DRIVE_AMP_MIN_UNIT).drive_amp_unit == DRIVE_AMP_MIN_UNIT


# ---------------------------------------------------------------------------
# Measurement plumbing
# ---------------------------------------------------------------------------


def _experiment(gen_name: str, seed: int = 0) -> CoreExperiment:
    gen = dict(presets.drive_controlled_generators())[gen_name]
    ts = gen.generate(random.Random(seed))
    noise = _parse_noise("Gauss(0.01)")
    nm = noise.to_noise_signal_model()
    ts = dataclasses.replace(ts, noise_model=nm, noise_bounds=nm.spec.bounds)
    return CoreExperiment(ts, noise, gen.drive_freq_min_phys, gen.drive_freq_max_phys)


@pytest.mark.parametrize("gen_name", ["NVCenter-cw-drive", "NVCenter-pulsed-drive"])
def test_measure_stamps_the_drive_actually_used(gen_name):
    exp = _experiment(gen_name)
    rng = random.Random(1)
    baseline = exp.true_signal.baseline_drive
    assert baseline is not None
    assert (baseline.pulse_duration_s is not None) == ("pulsed" in gen_name)
    obs = exp.measure(0.3, rng)
    assert obs.drive == baseline
    chosen = DriveSetting(0.8, None if "cw" in gen_name else 1e-7)
    obs2 = exp.measure(0.3, rng, drive=chosen)
    assert obs2.drive == chosen
    assert obs2.drive_amp_unit == 0.8


def test_measure_rejects_drive_without_drive_control():
    from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator

    gen = NVCenterCoreGenerator()
    ts = gen.generate(random.Random(0))
    exp = CoreExperiment(ts, None, gen.drive_freq_min_phys, gen.drive_freq_max_phys)
    assert exp.measure(0.5, random.Random(0)).drive is None
    with pytest.raises(ValueError, match="not drive-controlled"):
        exp.measure(0.5, random.Random(0), drive=DriveSetting(0.5))


def test_observation_carries_drive_through_aggregation():
    obs = aggregate_shots(0.5, np.array([0.9, 0.95]), 0.01, drive=DriveSetting(0.4))
    assert obs.drive == DriveSetting(0.4)
    assert Observation(0.5, 1.0).drive_amp_unit is None


# ---------------------------------------------------------------------------
# Belief and locator
# ---------------------------------------------------------------------------


def _belief(lineshape: str, fixed_drive=None):
    return nv_center_smc_belief(
        noise_model=gaussian_noise(), lineshape=lineshape, num_particles=300, seed=0, fixed_drive=fixed_drive
    )


def test_fixed_drive_belief_rejects_observation_at_other_drive():
    belief = _belief("cw_drive", fixed_drive=DriveSetting(0.3))
    belief.update(Observation(0.4, 0.98, drive=DriveSetting(0.3)))
    with pytest.raises(ValueError, match="fixed at"):
        belief.update(Observation(0.4, 0.98, drive=DriveSetting(0.9)))


def test_driven_belief_needs_each_observation_drive():
    belief = _belief("cw_drive")
    belief.update(Observation(0.4, 0.98, drive=DriveSetting(0.9)))
    with pytest.raises(ValueError, match="no drive"):
        belief.update(Observation(0.4, 0.98))


def test_driven_eig_grid_shape_and_selection():
    belief = _belief("cw_drive")
    freqs = belief.get_candidate_drive_freq_phys()[:50]
    drives = (DriveSetting(0.1), DriveSetting(0.5), DriveSetting(1.0))
    eig = belief.expected_information_gain_driven(freqs, drives)
    assert eig.shape == (3, freqs.shape[0])
    assert np.all(np.isfinite(eig))
    assert np.all(eig >= 0)
    f, drive = belief.select_max_information_gain_driven(freqs, drives)
    assert f in freqs
    assert drive in drives


def test_dominant_mode_picks_heavier_mode_not_mean():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(2e6, 5e4, 700), rng.normal(6e6, 5e4, 300)])
    weights = np.ones_like(values)
    mode = dominant_mode(values, weights)
    assert abs(mode - 2e6) < 2e5  # the mean (~3.2 MHz) would sit between the modes


def test_drive_locator_rejects_non_drive_model():
    belief = nv_center_smc_belief(noise_model=gaussian_noise(), num_particles=200, seed=0)
    with pytest.raises(ValueError, match="not drive-controlled"):
        DriveChoosingSBEDLocator(belief, max_steps=10)


def _run(locator_cls, gen_name: str, max_steps: int, seed: int = 0):
    exp = _experiment(gen_name, seed)
    ts = exp.true_signal
    bounds = {k: v for k, v in ts.bounds.items() if not k.startswith("_") or k == "_priors"}
    kw = {} if locator_cls.CHOOSES_DRIVE else {"fixed_drive": ts.baseline_drive}
    loc = locator_cls.create(
        builder=nv_center_smc_belief,
        max_steps=max_steps,
        parameter_bounds=bounds,
        noise_model=ts.noise_model,
        lineshape="cw_drive" if "cw" in gen_name else "pulsed_drive",
        num_particles=500,
        seed=seed,
        **kw,
    )
    rng = random.Random(seed)
    drives = []
    while not loc.done():
        f = loc.next_drive_freq_unit()
        obs = exp.measure(f, rng, drive=loc.next_drive())
        drives.append(obs.drive)
        loc.observe(obs)
    return loc, drives


@pytest.mark.parametrize("gen_name", ["NVCenter-cw-drive", "NVCenter-pulsed-drive"])
def test_drive_locator_runs_and_chooses_varied_in_range_drives(gen_name):
    loc, drives = _run(DriveChoosingSBEDLocator, gen_name, max_steps=30)
    assert len(drives) >= 5
    amps = [d.drive_amp_unit for d in drives]
    assert all(DRIVE_AMP_MIN_UNIT <= a <= 1.0 for a in amps)
    assert len(set(amps)) > 1
    if "pulsed" in gen_name:
        assert all(d.pulse_duration_s is not None for d in drives)
        assert loc.rabi_per_amp_calibration is not None
    assert "zeeman_shift" in loc.result()


def test_fixed_drive_sbed_measures_only_at_baseline():
    _loc, drives = _run(SequentialBayesianExperimentDesignLocator, "NVCenter-pulsed-drive", max_steps=15)
    assert len(set(drives)) == 1


def test_pulsed_drive_amplitude_falls_from_search_to_precision():
    """Seeded diagnostic (verified seed): high amplitude while searching, lower once the dip is found."""
    _, drives = _run(DriveChoosingSBEDLocator, "NVCenter-pulsed-drive", max_steps=250, seed=3)
    amps = np.array([d.drive_amp_unit for d in drives])
    k = max(1, len(amps) // 4)
    assert amps[:k].mean() > amps[-k:].mean()


def test_registration_and_run_group():
    from nvision.sim.run_groups import get_run_group

    grid = CombinationGrid()
    strategies = dict(grid.strategies_for("NVCenter-cw-drive"))
    assert strategies["Bayesian-SBED-Drive"]["class"] is DriveChoosingSBEDLocator
    assert strategies["Bayesian-SBED"]["config"]["lineshape"] == "cw_drive"
    assert "Bayesian-SBED-Drive" not in dict(grid.strategies_for("NVCenter-lorentzian"))
    assert grid.resolve("NVCenter-pulsed-drive", "Gauss(0.01)", "Bayesian-SBED-Drive") is not None
    group = get_run_group("drive-control")
    assert set(group.generator_names) == {"NVCenter-cw-drive", "NVCenter-pulsed-drive"}


def test_noise_recompute_after_resample_uses_each_observations_drive():
    """Resampling replays the noise posterior over the whole history, at every observation's own drive."""
    belief = _belief("cw_drive")
    drives = [DriveSetting(0.2), DriveSetting(1.0), DriveSetting(0.5)]
    for i, drive in enumerate(drives):
        belief.update(Observation(0.1 + 0.2 * i, 0.97, drive=drive))
    assert [o.drive for o in belief._observations] == drives
    belief._recompute_noise_betas()
    assert np.all(np.isfinite(belief._noise_betas))
    assert belief.copy()._obs_drives == drives
