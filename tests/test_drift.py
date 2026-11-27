import dataclasses
import math
import random

import numpy as np
import pytest

from nvision.models.experiment import CoreExperiment
from nvision.models.observation import Observation
from nvision.noises.drift import (
    NV_D_TEMPERATURE_COEFF_HZ_PER_K,
    NV_GYROMAGNETIC_HZ_PER_MG,
    DriftProcess,
    DriftSpec,
    attach_drift_for_repeat,
)
from nvision.sim import defaults as sim_defaults
from nvision.sim import presets
from nvision.sim.combinations import (
    CombinationGrid,
    _parse_noise,
    parse_drift_label,
    parse_gauss_sigma,
    parse_snr,
)
from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from tests.noise import gaussian_noise


def _signal(seed: int = 3):
    return NVCenterCoreGenerator(drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="lorentzian").generate(
        random.Random(seed), repeat_idx=0, n_repeats=1
    )


def _experiment(noise, seed: int = 3) -> CoreExperiment:
    return CoreExperiment(true_signal=_signal(seed), noise=noise, drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9)


def test_every_component_starts_at_zero():
    spec = presets.drift_scenario("stress")
    traj = spec.realize(seed=11, base_zeeman_shift=30e6)
    assert traj.center_offset_hz(0) == 0.0
    assert traj.zeeman_shift_offset_hz(0) == 0.0


def test_warmup_maps_temperature_and_field_to_physical_shifts():
    spec = DriftSpec(
        label="t",
        shot_duration_s=1.0,
        temperature_k=DriftProcess(warmup_amplitude=2.0, warmup_tau_s=10.0),
        field_mg=DriftProcess(warmup_amplitude=10.0, warmup_tau_s=10.0),
        magnet_tempco_per_k=-0.001,
    )
    traj = spec.realize(seed=0, base_zeeman_shift=20e6)
    frac = 1.0 - math.exp(-5.0 / 10.0)
    assert traj.center_offset_hz(5) == pytest.approx(NV_D_TEMPERATURE_COEFF_HZ_PER_K * 2.0 * frac)
    expected_zeeman_shift_offset = NV_GYROMAGNETIC_HZ_PER_MG * 10.0 * frac + (-0.001) * 2.0 * frac * 20e6
    assert traj.zeeman_shift_offset_hz(5) == pytest.approx(expected_zeeman_shift_offset)


def test_realization_is_a_pure_function_of_shot_index():
    spec = presets.drift_scenario("lab")
    sequential = spec.realize(seed=42, base_zeeman_shift=30e6)
    seq_values = [sequential.zeeman_shift_offset_hz(i) for i in range(9000)]
    jumped = spec.realize(seed=42, base_zeeman_shift=30e6)
    assert jumped.zeeman_shift_offset_hz(8999) == seq_values[8999]
    assert jumped.zeeman_shift_offset_hz(17) == seq_values[17]
    other = spec.realize(seed=43, base_zeeman_shift=30e6)
    assert other.zeeman_shift_offset_hz(8999) != seq_values[8999]


def test_ou_wander_reaches_its_stationary_std():
    spec = DriftSpec(label="ou", shot_duration_s=1.0, temperature_k=DriftProcess(ou_sigma=0.5, ou_tau_s=20.0))
    traj = spec.realize(seed=5, base_zeeman_shift=None)
    temps = np.array([traj.center_offset_hz(i) for i in range(200, 60000)]) / NV_D_TEMPERATURE_COEFF_HZ_PER_K
    assert temps.std() == pytest.approx(0.5, rel=0.1)
    assert abs(temps.mean()) < 0.1


def test_process_validation_rejects_nonphysical_parameters():
    with pytest.raises(ValueError, match="ou_tau_s"):
        DriftProcess(ou_tau_s=0.0)
    with pytest.raises(ValueError, match="ou_sigma"):
        DriftProcess(ou_sigma=-1.0)
    with pytest.raises(ValueError, match="shot_duration_s"):
        DriftSpec(label="x", shot_duration_s=0.0)


def test_non_drifting_measure_is_unchanged():
    exp = _experiment(presets.gauss_with_drift(0.01, None))
    drive_freq_unit = 0.4
    rng_a, rng_b = random.Random(9), random.Random(9)
    obs = exp.measure(drive_freq_unit, rng_a, shot_index=123)
    expected = exp.true_signal(2.6e9 + drive_freq_unit * 0.5e9) + rng_b.gauss(0.0, 0.01)
    assert obs.signal_value == expected


def test_drifting_measure_needs_realization_and_shot_index():
    noise = presets.gauss_with_drift(0.0, presets.drift_scenario("warmup"))
    exp = _experiment(noise)
    with pytest.raises(ValueError, match="no drift realization"):
        exp.measure(0.5, random.Random(0), shot_index=0)
    attached = attach_drift_for_repeat(exp, seed=1, generator_name="g", repeat_idx=0)
    with pytest.raises(ValueError, match="shot_index"):
        attached.measure(0.5, random.Random(0))


def test_drifting_measure_evaluates_the_truth_at_that_shot():
    spec = DriftSpec(
        label="big",
        shot_duration_s=1.0,
        temperature_k=DriftProcess(warmup_amplitude=20.0, warmup_tau_s=1.0),
        field_mg=DriftProcess(warmup_amplitude=300.0, warmup_tau_s=1.0),
    )
    exp = attach_drift_for_repeat(_experiment(presets.gauss_with_drift(0.0, spec)), 1, "g", 0)
    base = exp.true_signal.typed_parameters
    for shot in (0, 50):
        truth = exp.drift.apply(base, shot)
        drive_freq_unit = (truth.zero_field_freq - truth.zeeman_shift - 2.6e9) / 0.5e9  # on the moving left dip
        obs = exp.measure(drive_freq_unit, random.Random(0), shot_index=shot)
        assert obs.signal_value == pytest.approx(exp.true_signal.model.compute(2.6e9 + drive_freq_unit * 0.5e9, truth))
    moved = exp.drift.apply(base, 50)
    assert moved.zero_field_freq == pytest.approx(
        base.zero_field_freq + 20.0 * NV_D_TEMPERATURE_COEFF_HZ_PER_K, rel=1e-6
    )
    assert moved.zeeman_shift == pytest.approx(base.zeeman_shift + 300.0 * NV_GYROMAGNETIC_HZ_PER_MG, rel=1e-6)


def test_shots_in_a_batch_are_taken_at_successive_times():
    spec = DriftSpec(label="w", shot_duration_s=1.0, field_mg=DriftProcess(warmup_amplitude=500.0, warmup_tau_s=2.0))
    exp = attach_drift_for_repeat(_experiment(presets.gauss_with_drift(0.0, spec)), 1, "g", 0)
    base = exp.true_signal.typed_parameters
    drive_freq_phys = base.zero_field_freq - base.zeeman_shift
    drive_freq_unit = (drive_freq_phys - 2.6e9) / 0.5e9
    batch = exp.measure(drive_freq_unit, random.Random(0), n_shots=3, shot_index=4)
    singles = [exp.measure(drive_freq_unit, random.Random(0), shot_index=4 + i).signal_value for i in range(3)]
    assert batch.signal_value == pytest.approx(np.mean(singles))
    assert batch.sample_var is not None
    assert batch.sample_var > 0


def test_attach_is_shared_across_strategies_and_noop_without_drift():
    plain = _experiment(presets.gauss_with_drift(0.01, None))
    assert attach_drift_for_repeat(plain, 1, "g", 0) is plain
    noise = presets.gauss_with_drift(0.01, presets.drift_scenario("lab"))
    a = attach_drift_for_repeat(_experiment(noise), 7, "gen", 2)
    b = attach_drift_for_repeat(_experiment(noise), 7, "gen", 2)
    c = attach_drift_for_repeat(_experiment(noise), 7, "gen", 3)
    assert attach_drift_for_repeat(a, 7, "gen", 2) is a
    assert a.drift.zeeman_shift_offset_hz(500) == b.drift.zeeman_shift_offset_hz(500)
    assert a.drift.zeeman_shift_offset_hz(500) != c.drift.zeeman_shift_offset_hz(500)


def test_truth_summary_reports_end_and_mean():
    spec = DriftSpec(label="w", shot_duration_s=1.0, temperature_k=DriftProcess(warmup_amplitude=1.0, warmup_tau_s=5.0))
    exp = attach_drift_for_repeat(_experiment(presets.gauss_with_drift(0.0, spec)), 1, "g", 0)
    base = exp.true_signal.typed_parameters
    summary = exp.drift.truth_summary(base, 10)
    path = [base.zero_field_freq + exp.drift.center_offset_hz(i) for i in range(10)]
    assert summary["drift_true_zero_field_freq_start"] == base.zero_field_freq
    assert summary["drift_true_zero_field_freq_end"] == pytest.approx(path[-1])
    assert summary["drift_true_zero_field_freq_mean"] == pytest.approx(np.mean(path))
    assert "drift_true_zeeman_shift_end" not in summary


def test_splitting_drift_on_a_single_dip_signal_fails_loudly():
    single = NVCenterCoreGenerator(
        drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="lorentzian", with_zeeman_splitting=False
    ).generate(random.Random(0), repeat_idx=0, n_repeats=1)
    spec = DriftSpec(label="f", shot_duration_s=1.0, field_mg=DriftProcess(warmup_amplitude=1.0))
    exp = CoreExperiment(
        true_signal=single,
        noise=presets.gauss_with_drift(0.0, spec),
        drive_freq_min_phys=2.6e9,
        drive_freq_max_phys=3.1e9,
    )
    exp = attach_drift_for_repeat(exp, 1, "g", 0)
    assert not hasattr(exp.true_signal.typed_parameters, "zeeman_shift")
    with pytest.raises(ValueError, match="zeeman_shift"):
        exp.measure(0.5, random.Random(0), shot_index=1)


def _jittered_experiment(step_hz: float, noise=None) -> CoreExperiment:
    generator = NVCenterCoreGenerator(
        drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="lorentzian", line_walk_step_phys=step_hz
    )
    return CoreExperiment(
        true_signal=generator.generate(random.Random(3), repeat_idx=0, n_repeats=1),
        noise=noise,
        drive_freq_min_phys=2.6e9,
        drive_freq_max_phys=3.1e9,
    )


def test_line_jitter_is_a_shared_random_walk_of_the_stated_step():
    exp = _jittered_experiment(600.0)
    a = attach_drift_for_repeat(exp, 1, "g", 0)
    b = attach_drift_for_repeat(_jittered_experiment(600.0, presets.gauss_with_drift(0.01, None)), 1, "g", 0)
    path = np.array([a.drift.zeeman_shift_offset_hz(i) for i in range(20000)])
    assert path[0] == 0.0
    assert np.std(np.diff(path)) == pytest.approx(600.0, rel=0.03)
    # The same path for every noise level and strategy of the repeat, in any query order.
    assert b.drift.zeeman_shift_offset_hz(15000) == path[15000]
    assert attach_drift_for_repeat(exp, 1, "g", 1).drift.zeeman_shift_offset_hz(500) != path[500]


def test_jittering_signal_cannot_be_measured_as_if_still():
    exp = _jittered_experiment(600.0)
    with pytest.raises(ValueError, match="no drift realization"):
        exp.measure(0.5, random.Random(0), shot_index=0)


def test_truth_traces_follow_the_walk_per_measurement():
    exp = attach_drift_for_repeat(_jittered_experiment(600.0), 1, "g", 0)
    base = exp.true_signal.typed_parameters
    traces = exp.drift.truth_traces(base, 5, n_shots=2)
    expected = [base.zeeman_shift + exp.drift.zeeman_shift_offset_hz(2 * i + 1) for i in range(5)]
    assert list(traces) == ["zeeman_shift"]
    assert traces["zeeman_shift"] == pytest.approx(expected)


def test_run_group_signals_jitter_and_the_bound_levels_off():
    from nvision.metrics.detection_aligned import moving_line_bound
    from nvision.spectra.nv_center import NV_LINE_JITTER_STEP_FWHM

    for name, gen in presets.tracking_generators() + presets.lab_mirror_generators():
        assert gen.line_walk_step_phys > 0, name
    for name, gen in presets.voigt_snr_window_param_grid_generators():
        assert gen.line_walk_step_phys == pytest.approx(
            NV_LINE_JITTER_STEP_FWHM * sim_defaults.NVISION_SBED_FWHM, rel=0.01
        ), name
    counts = np.array([1.0, 2.0, 100000.0])
    j0, i_max, q = np.array([2.0]), 3.0, 0.1
    still = moving_line_bound(j0, counts, i_max, 0.0, first_step_moves=False)[0]
    assert still == pytest.approx(1.0 / (2.0 + counts * i_max))
    moving = moving_line_bound(j0, counts, i_max, q, first_step_moves=False)[0]
    assert moving[0] == pytest.approx(1.0 / 5.0)  # no step before the first measurement
    assert moving[1] == pytest.approx(1.0 / (1.0 / (1.0 / 5.0 + q**2) + i_max))
    # Steady state: P solves 1/P = 1/(P + q^2) + I.
    steady = (-(q**2) + math.sqrt(q**4 + 4.0 * q**2 / i_max)) / 2.0
    assert moving[2] == pytest.approx(steady)


def _walking_belief(step_phys: float):
    return nv_center_smc_belief(noise_model=gaussian_noise(), num_particles=4000, seed=3, line_walk_step_phys=step_phys)


def _shift_unit(belief) -> np.ndarray:
    return belief._particles[:, belief.parameter_names.index("zeeman_shift")].astype(np.float64)


def test_walking_belief_spreads_by_one_step_between_observations(monkeypatch):
    """With an uninformative likelihood only the motion acts: no step before the first observation, then one each."""
    lo, hi = _walking_belief(0.0).physical_param_bounds["zeeman_shift"]
    step_unit = 0.002
    belief = _walking_belief(step_unit * (hi - lo))
    column = belief.parameter_names.index("zeeman_shift")
    belief._particles[:, column] = 0.5  # one point, far from the bounds
    belief.auto_resample = False
    monkeypatch.setattr(belief, "_log_likelihood_and_update_noise", lambda obs, predicted: np.zeros_like(predicted))
    belief.update(Observation(0.5, 1.0))
    assert np.std(_shift_unit(belief)) == 0.0
    for _ in range(25):
        belief.update(Observation(0.5, 1.0))
    assert np.std(_shift_unit(belief)) == pytest.approx(step_unit * math.sqrt(25), rel=0.05)


def test_walking_belief_stays_in_bounds_and_a_batch_matches_single_updates():
    lo, hi = _walking_belief(0.0).physical_param_bounds["zeeman_shift"]
    one, batch = (_walking_belief(0.2 * (hi - lo)) for _ in range(2))  # a huge step: the walk must reflect
    for belief in (one, batch):
        belief.auto_resample = False
    observations = [Observation(0.1 * i, 0.99) for i in range(1, 6)]
    for obs in observations:
        one.update(obs)
    batch.batch_update(observations)
    np.testing.assert_array_equal(one._particles, batch._particles)
    np.testing.assert_allclose(one._weights, batch._weights, rtol=1e-4)
    assert _shift_unit(one).min() >= 0.0
    assert _shift_unit(one).max() <= 1.0
    assert one.copy().line_walk_step_phys == one.line_walk_step_phys


def test_walking_belief_resample_and_eig_follow_the_moved_particles():
    lo, hi = _walking_belief(0.0).physical_param_bounds["zeeman_shift"]
    belief = _walking_belief(0.001 * (hi - lo))
    belief.auto_resample = False
    candidates = belief.get_candidate_drive_freq_phys()
    belief.update(Observation(0.3, 0.99))
    belief.expected_information_gain(candidates)
    subset = belief._eig_cache[3]
    predictions = belief._eig_cache[1].copy()
    belief.update(Observation(0.3, 0.99))
    belief.expected_information_gain(candidates)
    assert belief._eig_cache[3] is subset  # same epoch, same particle subset
    assert not np.array_equal(belief._eig_cache[1], predictions)  # re-predicted after the walk
    parents = belief._noise_betas.copy()
    belief._resample()
    assert np.isin(belief._noise_betas, parents).all()  # carried from parents, not replayed


def test_only_tracking_tells_the_locators_the_step():
    for name, gen in presets.tracking_generators():
        signal = gen.generate(random.Random(0), repeat_idx=0, n_repeats=1)
        std_phys = gen.tracking_prior_std_fwhm * sim_defaults.NVISION_SBED_FWHM
        expected = std_phys / math.sqrt(sim_defaults.NVISION_MAX_MEASUREMENTS)
        assert signal.line_walk_step_phys == pytest.approx(expected, rel=0.01), name
        assert signal.locator_walk_step_phys == signal.line_walk_step_phys, name
    for name, gen in presets.lab_mirror_generators():
        signal = gen.generate(random.Random(0), repeat_idx=0, n_repeats=1)
        assert signal.line_walk_step_phys > 0, name
        assert signal.locator_walk_step_phys == 0.0, name


def test_drift_noise_names_round_trip():
    names = [name for name, _ in presets.drift_study_noises()]
    assert len(names) == 1 + len(presets.drift_scenarios())
    grid = CombinationGrid()
    generator = next(iter(presets.voigt_snr_window_param_grid_generators()))[0]
    for name, noise in presets.drift_study_noises():
        assert parse_snr(name) == sim_defaults.NVISION_DRIFT_SNR
        assert parse_gauss_sigma(name) is None
        assert parse_drift_label(name) == (None if noise.drift is None else noise.drift.label)
        combo = grid.resolve(generator, name, "Bayesian-SBED")
        assert combo is not None
        assert combo.noise.drift == noise.drift
        assert combo.noise.over_frequency_noise._parts[0].std == pytest.approx(
            presets.sigma_for_contrast_snr(sim_defaults.NVISION_SBED_SNR_CONTRAST, sim_defaults.NVISION_DRIFT_SNR)
        )
    # The older Gauss(sigma) names still resolve.
    for gauss in ("Gauss(0.006)+Drift(warmup)", "Gauss(0.006)"):
        parsed = _parse_noise(gauss)
        assert parsed is not None
        assert parse_drift_label(gauss) == (None if "Drift" not in gauss else "warmup")
        assert parsed.drift == (None if "Drift" not in gauss else presets.drift_scenario("warmup"))
    assert _parse_noise("Gauss(0.01)+Drift(no-such-scenario)") is None
    assert parse_gauss_sigma("Gauss(0.01)") == 0.01


def test_scenario_labels_are_unique_and_sized_as_documented():
    specs = presets.drift_scenarios()
    assert len({s.label for s in specs}) == len(specs)
    warmup = presets.drift_scenario("warmup")
    full = dataclasses.replace(
        warmup, temperature_k=DriftProcess(warmup_amplitude=4.0, warmup_tau_s=300.0), field_mg=DriftProcess()
    )
    traj = full.realize(0, base_zeeman_shift=30e6)
    assert traj.center_offset_hz(100000) == pytest.approx(-296.8e3, rel=1e-3)
    assert traj.zeeman_shift_offset_hz(100000) == pytest.approx(-144e3, rel=1e-3)
