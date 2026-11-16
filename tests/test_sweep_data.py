"""The reference sweep's persisted data: what is stored, and and its strategy name."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from nvision import CoreExperiment, NVCenterCoreGenerator
from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
from nvision.models.observation import Observation
from nvision.noises import OverFrequencyGaussianNoise
from nvision.runner.sweep_data import (
    MissingSweepDataError,
    load_sweep_data,
    reference_sweep_strategy,
    sweep_data_from_run,
)
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief, nv_lineshape_for_model
from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator
from nvision.sim.locs.search_grid import SWEEP_POINTS_PER_FWHM
from tests.noise import gaussian_noise

NOISE_SIGMA = 0.01


def _experiment(seed: int = 3, hwhm: float = 2.75e6, contrast: float = 0.4):
    gen = NVCenterCoreGenerator(variant="lorentzian", hwhm=hwhm, contrast_total=contrast)
    true_signal = gen.generate(random.Random(seed), repeat_idx=0, n_repeats=1)
    noise = CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(NOISE_SIGMA)]))
    exp = CoreExperiment(
        true_signal=true_signal,
        noise=noise,
        drive_freq_min_phys=gen.drive_freq_min_phys,
        drive_freq_max_phys=gen.drive_freq_max_phys,
    )
    return exp, true_signal.bounds


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


class _Repo:
    def __init__(self, stored):
        self.stored = stored
        self.calls = []

    def load_scan_field(self, **kw):
        self.calls.append(kw)
        return self.stored


_COMBO = {"generator": "g", "noise": "n", "strategy": "Bayesian-SBED", "seed": 1, "max_steps": 10, "timeout_s": 5}


def test_load_reads_the_reference_sweeps_combination():
    repo = _Repo({"x": 1})
    assert load_sweep_data(repo, _COMBO, 3) == {"x": 1}
    assert repo.calls[0]["strategy"] == reference_sweep_strategy() == f"SimpleSweep-{SWEEP_POINTS_PER_FWHM}perFWHM"
    assert repo.calls[0]["repeat_idx"] == 3
    assert repo.calls[0]["field"] == "sweep_data"


def test_missing_sweep_fails_fast():
    with pytest.raises(MissingSweepDataError, match="run that strategy first"):
        load_sweep_data(_Repo(None), _COMBO, 0)
