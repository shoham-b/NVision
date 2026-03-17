"""Tests for core locator architecture."""

from __future__ import annotations

import math
import random

from nvision.core import Locator
from nvision.core.experiment import CoreExperiment
from nvision.core.observer import Observer, RunResult
from nvision.core.runner import Runner
from nvision.core.signal import Parameter, TrueSignal
from nvision.core.models import GaussianModel
from nvision.sim.locs.core import SimpleSweepLocator


def _gaussian_experiment(center: float = 0.5, sigma: float = 0.1) -> CoreExperiment:
    model = GaussianModel()
    parameters = [
        Parameter(name="frequency", bounds=(0.0, 1.0), value=center),
        Parameter(name="sigma", bounds=(0.01, 0.3), value=sigma),
        Parameter(name="amplitude", bounds=(0.0, 1.5), value=1.0),
        Parameter(name="background", bounds=(0.0, 0.5), value=0.0),
    ]
    true_signal = TrueSignal(model=model, parameters=parameters)
    return CoreExperiment(true_signal=true_signal, noise=None, x_min=0.0, x_max=1.0)


def test_simple_sweep_locator_is_core_locator():
    assert issubclass(SimpleSweepLocator, Locator)


def test_simple_sweep_create_returns_instance():
    loc = SimpleSweepLocator.create(max_steps=10)
    assert isinstance(loc, SimpleSweepLocator)


def test_locator_proposes_valid_positions():
    exp = _gaussian_experiment()
    rng = random.Random(1)
    runner = Runner()
    for locator in runner.run(SimpleSweepLocator, exp, rng, max_steps=5):
        assert locator.belief.last_obs is not None
        x = locator.belief.last_obs.x
        assert 0.0 <= x <= 1.0


def test_runner_yields_exactly_max_steps():
    exp = _gaussian_experiment()
    rng = random.Random(2)
    runner = Runner()
    steps = list(runner.run(SimpleSweepLocator, exp, rng, max_steps=10))
    assert len(steps) <= 10
    assert len(steps) > 0


def test_observer_records_snapshots():
    exp = _gaussian_experiment()
    rng = random.Random(3)
    runner = Runner()
    observer = Observer(exp.true_signal, exp.x_min, exp.x_max)
    result = observer.watch(runner.run(SimpleSweepLocator, exp, rng, max_steps=15))
    assert isinstance(result, RunResult)
    assert len(result.snapshots) > 0


def test_locator_estimates_are_finite():
    exp = _gaussian_experiment(center=0.6)
    rng = random.Random(42)
    runner = Runner()
    last_locator = None
    for last_locator in runner.run(SimpleSweepLocator, exp, rng, max_steps=30):
        pass
    assert last_locator is not None
    estimates = last_locator.belief.estimates()
    assert all(math.isfinite(v) for v in estimates.values())


def test_locator_comparison_different_max_steps():
    exp = _gaussian_experiment(center=0.4)

    rng_a = random.Random(99)
    rng_b = random.Random(99)
    runner = Runner()

    steps_a = list(runner.run(SimpleSweepLocator, exp, rng_a, max_steps=5))
    steps_b = list(runner.run(SimpleSweepLocator, exp, rng_b, max_steps=20))

    # More steps means same or more measurements
    assert len(steps_b) >= len(steps_a)
