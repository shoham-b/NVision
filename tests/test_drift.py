import dataclasses
import math
import random

import numpy as np
import pytest

from nvision.models.experiment import CoreExperiment
from nvision.noises.drift import (
    NV_D_TEMPERATURE_COEFF_HZ_PER_K,
    NV_GYROMAGNETIC_HZ_PER_MG,
    DriftProcess,
    DriftSpec,
    attach_drift_for_repeat,
)
from nvision.sim import presets
from nvision.sim.combinations import CombinationGrid, _parse_noise, parse_drift_label, parse_gauss_sigma
from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator


def _signal(seed: int = 3):
    return NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian").generate(random.Random(seed))


def _experiment(noise, seed: int = 3) -> CoreExperiment:
    return CoreExperiment(true_signal=_signal(seed), noise=noise, x_min=2.6e9, x_max=3.1e9)


def test_every_component_starts_at_zero():
    spec = presets.drift_scenario("stress")
    traj = spec.realize(seed=11, base_zeeman_split=30e6)
    assert traj.center_offset_hz(0) == 0.0
    assert traj.split_offset_hz(0) == 0.0


def test_warmup_maps_temperature_and_field_to_physical_shifts():
    spec = DriftSpec(
        label="t",
        shot_duration_s=1.0,
        temperature_k=DriftProcess(warmup_amplitude=2.0, warmup_tau_s=10.0),
        field_mg=DriftProcess(warmup_amplitude=10.0, warmup_tau_s=10.0),
        magnet_tempco_per_k=-0.001,
    )
    traj = spec.realize(seed=0, base_zeeman_split=20e6)
    frac = 1.0 - math.exp(-5.0 / 10.0)
    assert traj.center_offset_hz(5) == pytest.approx(NV_D_TEMPERATURE_COEFF_HZ_PER_K * 2.0 * frac)
    expected_split = NV_GYROMAGNETIC_HZ_PER_MG * 10.0 * frac + (-0.001) * 2.0 * frac * 20e6
    assert traj.split_offset_hz(5) == pytest.approx(expected_split)


def test_realization_is_a_pure_function_of_shot_index():
    spec = presets.drift_scenario("lab")
    sequential = spec.realize(seed=42, base_zeeman_split=30e6)
    seq_values = [sequential.split_offset_hz(i) for i in range(9000)]
    jumped = spec.realize(seed=42, base_zeeman_split=30e6)
    assert jumped.split_offset_hz(8999) == seq_values[8999]
    assert jumped.split_offset_hz(17) == seq_values[17]
    other = spec.realize(seed=43, base_zeeman_split=30e6)
    assert other.split_offset_hz(8999) != seq_values[8999]


def test_ou_wander_reaches_its_stationary_std():
    spec = DriftSpec(label="ou", shot_duration_s=1.0, temperature_k=DriftProcess(ou_sigma=0.5, ou_tau_s=20.0))
    traj = spec.realize(seed=5, base_zeeman_split=None)
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
    x = 0.4
    rng_a, rng_b = random.Random(9), random.Random(9)
    obs = exp.measure(x, rng_a, shot_index=123)
    expected = exp.true_signal(2.6e9 + x * 0.5e9) + rng_b.gauss(0.0, 0.01)
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
        x = (truth.frequency - truth.zeeman_split - 2.6e9) / 0.5e9  # on the moving left dip
        obs = exp.measure(x, random.Random(0), shot_index=shot)
        assert obs.signal_value == pytest.approx(exp.true_signal.model.compute(2.6e9 + x * 0.5e9, truth))
    moved = exp.drift.apply(base, 50)
    assert moved.frequency == pytest.approx(base.frequency + 20.0 * NV_D_TEMPERATURE_COEFF_HZ_PER_K, rel=1e-6)
    assert moved.zeeman_split == pytest.approx(base.zeeman_split + 300.0 * NV_GYROMAGNETIC_HZ_PER_MG, rel=1e-6)


def test_shots_in_a_batch_are_taken_at_successive_times():
    spec = DriftSpec(label="w", shot_duration_s=1.0, field_mg=DriftProcess(warmup_amplitude=500.0, warmup_tau_s=2.0))
    exp = attach_drift_for_repeat(_experiment(presets.gauss_with_drift(0.0, spec)), 1, "g", 0)
    base = exp.true_signal.typed_parameters
    x_phys = base.frequency - base.zeeman_split
    x = (x_phys - 2.6e9) / 0.5e9
    batch = exp.measure(x, random.Random(0), n_shots=3, shot_index=4)
    singles = [exp.measure(x, random.Random(0), shot_index=4 + i).signal_value for i in range(3)]
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
    assert a.drift.split_offset_hz(500) == b.drift.split_offset_hz(500)
    assert a.drift.split_offset_hz(500) != c.drift.split_offset_hz(500)


def test_truth_summary_reports_end_and_mean():
    spec = DriftSpec(label="w", shot_duration_s=1.0, temperature_k=DriftProcess(warmup_amplitude=1.0, warmup_tau_s=5.0))
    exp = attach_drift_for_repeat(_experiment(presets.gauss_with_drift(0.0, spec)), 1, "g", 0)
    base = exp.true_signal.typed_parameters
    summary = exp.drift.truth_summary(base, 10)
    path = [base.frequency + exp.drift.center_offset_hz(i) for i in range(10)]
    assert summary["drift_true_frequency_start"] == base.frequency
    assert summary["drift_true_frequency_end"] == pytest.approx(path[-1])
    assert summary["drift_true_frequency_mean"] == pytest.approx(np.mean(path))
    assert "drift_true_zeeman_split_end" not in summary


def test_splitting_drift_on_a_single_dip_signal_fails_loudly():
    single = NVCenterCoreGenerator(
        x_min=2.6e9, x_max=3.1e9, variant="lorentzian", with_zeeman_splitting=False
    ).generate(random.Random(0))
    spec = DriftSpec(label="f", shot_duration_s=1.0, field_mg=DriftProcess(warmup_amplitude=1.0))
    exp = CoreExperiment(true_signal=single, noise=presets.gauss_with_drift(0.0, spec), x_min=2.6e9, x_max=3.1e9)
    exp = attach_drift_for_repeat(exp, 1, "g", 0)
    assert not hasattr(exp.true_signal.typed_parameters, "zeeman_split")
    with pytest.raises(ValueError, match="zeeman_split"):
        exp.measure(0.5, random.Random(0), shot_index=1)


def test_drift_noise_names_round_trip():
    names = [name for name, _ in presets.drift_study_noises()]
    assert len(names) == 1 + len(presets.drift_scenarios())
    for name, noise in presets.drift_study_noises():
        parsed = _parse_noise(name)
        assert parsed is not None
        assert parse_gauss_sigma(name) == noise.over_frequency_noise._parts[0].std
        assert parse_drift_label(name) == (None if noise.drift is None else noise.drift.label)
        assert parsed.drift == noise.drift
    assert _parse_noise("Gauss(0.01)+Drift(no-such-scenario)") is None
    assert parse_gauss_sigma("Gauss(0.01)") == 0.01


def test_drift_group_resolves_every_combination():
    from nvision.sim.run_groups import get_run_group

    group = get_run_group("lorentzian-plain-drift")
    grid = CombinationGrid(extra_generators=group.extra_generators)
    gen = group.generator_names[0]
    for noise_name in group.noise_names:
        for strat in group.strategy_names:
            combo = grid.resolve(gen, noise_name, strat)
            assert combo is not None
            assert (combo.noise.drift is None) == ("+Drift(" not in noise_name)


def test_scenario_labels_are_unique_and_sized_as_documented():
    specs = presets.drift_scenarios()
    assert len({s.label for s in specs}) == len(specs)
    warmup = presets.drift_scenario("warmup")
    full = dataclasses.replace(
        warmup, temperature_k=DriftProcess(warmup_amplitude=4.0, warmup_tau_s=300.0), field_mg=DriftProcess()
    )
    traj = full.realize(0, base_zeeman_split=30e6)
    assert traj.center_offset_hz(100000) == pytest.approx(-296.8e3, rel=1e-3)
    assert traj.split_offset_hz(100000) == pytest.approx(-144e3, rel=1e-3)
