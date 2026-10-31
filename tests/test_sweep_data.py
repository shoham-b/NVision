"""The reference sweep's persisted data: what is stored, and how the smart searches replay it."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from nvision.models.observation import Observation
from nvision.runner.executor import run_loop
from nvision.runner.sweep_data import (
    MissingSweepDataError,
    SweepReplay,
    load_sweep_data,
    reference_sweep_strategy,
    sweep_data_from_run,
)
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief, nv_lineshape_for_model
from nvision.sim.locs.bayesian.sbed_locator import SmartSBEDLocator, SmartSweepLocator
from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator
from nvision.sim.locs.search_grid import SWEEP_POINTS_PER_FWHM
from tests.noise import gaussian_noise
from tests.test_smart_locators import NOISE_SIGMA, _experiment


def _reference_sweep(exp, bounds, n_points: int) -> list[Observation]:
    """The first ``n_points`` measurements of a plain sweep at SWEEP_POINTS_PER_FWHM, as the executor takes them."""
    belief = nv_center_smc_belief(
        bounds,
        noise_model=gaussian_noise(),
        lineshape=nv_lineshape_for_model(exp.true_signal.model),
        fixed_drive=exp.true_signal.baseline_drive,
    )
    sweep = GenericSweepLocator(
        belief=belief,
        signal_model=exp.true_signal.baseline_model,
        max_steps=400,
        noise_std=NOISE_SIGMA,
        domain_lo=exp.drive_freq_min_phys,
        domain_hi=exp.drive_freq_max_phys,
    )
    rng = random.Random(5)
    observations: list[Observation] = []
    for _ in range(n_points):
        obs = exp.measure(sweep.next_drive_freq_unit(), rng)
        sweep.observe(obs)
        observations.append(obs)
    return observations


def _stored(observations: list[Observation]) -> dict:
    snapshots = [SimpleNamespace(obs=o) for o in observations]
    return sweep_data_from_run(SimpleNamespace(snapshots=snapshots, fit_mode_estimates={"zero_field_freq": 2.87e9}))


def test_sweep_data_round_trips_through_json():
    exp, bounds = _experiment()
    data = _stored(_reference_sweep(exp, bounds, 5))
    assert json.loads(json.dumps(data)) == data
    assert len(data["drive_freqs_unit"]) == len(data["signal_values"]) == 5
    assert data["mode_estimates"] == {"zero_field_freq": 2.87e9}


def test_sweep_without_a_fit_stores_no_curve():
    exp, bounds = _experiment()
    snapshots = [SimpleNamespace(obs=o) for o in _reference_sweep(exp, bounds, 3)]
    assert sweep_data_from_run(SimpleNamespace(snapshots=snapshots, fit_mode_estimates=None))["mode_estimates"] is None


def test_smart_search_grid_is_the_reference_sweep_grid():
    exp, bounds = _experiment()
    sweep_xs = [o.drive_freq_unit for o in _reference_sweep(exp, bounds, 10)]
    for cls in (SmartSweepLocator,):
        loc = cls.create(
            builder=nv_center_smc_belief,
            max_steps=50,
            parameter_bounds=bounds,
            noise_model=gaussian_noise(),
            num_particles=200,
            seed=7,
            lineshape="lorentzian",
        )
        xs = []
        for _ in range(10):  # before the first dip moves the focus
            x = loc.next_drive_freq_unit()
            xs.append(x)
            loc.observe(exp.measure(x, random.Random(1)))
        np.testing.assert_allclose(xs, sweep_xs, atol=1e-9)


def test_replay_hands_out_each_point_once_and_rejects_off_grid_points():
    exp, bounds = _experiment()
    observations = _reference_sweep(exp, bounds, 6)
    replay = SweepReplay(_stored(observations))
    first = replay.take(exp, observations[2].drive_freq_unit)
    assert first.signal_value == observations[2].signal_value
    assert first.noise_std == observations[2].noise_std
    assert first.frequency_noise_model == observations[2].frequency_noise_model
    assert replay.take(exp, observations[2].drive_freq_unit) is None  # already used: measure afresh
    with pytest.raises(ValueError, match="not on the persisted sweep grid"):
        replay.take(exp, 0.5 * (observations[0].drive_freq_unit + observations[1].drive_freq_unit))


@pytest.mark.parametrize("cls", [SmartSweepLocator, SmartSBEDLocator])
def test_smart_search_measurements_come_from_the_persisted_sweep(cls, monkeypatch):
    exp, bounds = _experiment()
    observations = _reference_sweep(exp, bounds, 96)  # the whole grid: 95 spacings of FWHM/3 over this axis
    replay = SweepReplay(_stored(observations))
    drawn: list[float] = []
    real_measure = exp.measure

    def counting_measure(x, rng, **kw):
        drawn.append(x)
        return real_measure(x, rng, **kw)

    monkeypatch.setattr(exp, "measure", counting_measure)
    steps = 0
    for _ in run_loop(
        cls,
        exp,
        random.Random(11),
        None,
        search_replay=replay,
        builder=nv_center_smc_belief,
        max_steps=40,
        parameter_bounds=bounds,
        noise_model=gaussian_noise(),
        num_particles=300,
        seed=7,
        lineshape="lorentzian",
    ):
        steps += 1
    from_sweep = int(replay._used.sum())
    assert from_sweep > 0, "the search phase must have used the persisted sweep"
    assert len(drawn) == steps - from_sweep, "only the non-search measurements are drawn from the experiment"


class _Repo:
    def __init__(self, stored):
        self.stored = stored
        self.calls = []

    def load_scan_field(self, **kw):
        self.calls.append(kw)
        return self.stored


_COMBO = {"generator": "g", "noise": "n", "strategy": "Smart-SBED", "seed": 1, "max_steps": 10, "timeout_s": 5}


def test_load_reads_the_reference_sweeps_combination():
    repo = _Repo({"x": 1})
    assert load_sweep_data(repo, _COMBO, 3) == {"x": 1}
    assert repo.calls[0]["strategy"] == reference_sweep_strategy() == f"SimpleSweep-{SWEEP_POINTS_PER_FWHM}perFWHM"
    assert repo.calls[0]["repeat_idx"] == 3
    assert repo.calls[0]["field"] == "sweep_data"


def test_missing_sweep_fails_fast():
    with pytest.raises(MissingSweepDataError, match="run that strategy first"):
        load_sweep_data(_Repo(None), _COMBO, 0)
