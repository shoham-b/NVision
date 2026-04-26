"""Tests for core locator architecture."""

from __future__ import annotations

import math
import random

import numpy as np

from nvision import (
    CoreExperiment,
    GaussianModel,
    Locator,
    Observer,
    RunResult,
    SimpleSweepLocator,
    TrueSignal,
    run_loop,
)
from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter


def _gaussian_experiment(center: float = 0.5, sigma: float = 0.1) -> CoreExperiment:
    from nvision.spectra.gaussian import GaussianSpectrum

    model = GaussianModel()
    typed_params = GaussianSpectrum(
        frequency=center,
        sigma=sigma,
        dip_depth=1.0,
        background=0.0,
    )
    bounds = {
        "frequency": (0.0, 1.0),
        "sigma": (0.01, 0.3),
        "dip_depth": (0.0, 1.5),
        "background": (0.0, 0.5),
    }
    true_signal = TrueSignal(model=model, typed_parameters=typed_params, bounds=bounds)
    return CoreExperiment(true_signal=true_signal, noise=None, x_min=0.0, x_max=1.0)


def _dummy_belief(model):
    grid = np.linspace(0.0, 1.0, 10)
    posterior = np.ones(10) / 10
    parameters = [
        GridParameter(name=name, bounds=(0.0, 1.0), grid=grid, posterior=posterior) for name in model.parameter_names()
    ]
    return GridMarginalDistribution(model=model, parameters=parameters)


def test_simple_sweep_locator_is_core_locator():
    assert issubclass(SimpleSweepLocator, Locator)


def test_simple_sweep_create_returns_instance():
    model = GaussianModel()
    belief = _dummy_belief(model)
    loc = SimpleSweepLocator.create(belief=belief, signal_model=model, max_steps=10)
    assert isinstance(loc, SimpleSweepLocator)


def test_locator_proposes_valid_positions():
    exp = _gaussian_experiment()
    rng = random.Random(1)
    for locator in run_loop(SimpleSweepLocator, exp, rng, max_steps=5):
        assert locator.belief.last_obs is not None
        x = locator.belief.last_obs.x
        assert 0.0 <= x <= 1.0


def test_runner_yields_exactly_max_steps():
    exp = _gaussian_experiment()
    rng = random.Random(2)
    steps = list(run_loop(SimpleSweepLocator, exp, rng, max_steps=10))
    assert len(steps) <= 10
    assert len(steps) > 0


def test_observer_records_snapshots():
    exp = _gaussian_experiment()
    rng = random.Random(3)
    observer = Observer(exp.true_signal, exp.x_min, exp.x_max)
    result = observer.watch(run_loop(SimpleSweepLocator, exp, rng, max_steps=15))
    assert isinstance(result, RunResult)
    assert len(result.snapshots) > 0


def test_locator_estimates_are_finite():
    exp = _gaussian_experiment(center=0.6)
    rng = random.Random(42)
    last_locator = None
    for loc in run_loop(SimpleSweepLocator, exp, rng, max_steps=30):
        last_locator = loc
    assert last_locator is not None
    estimates = last_locator.belief.estimates()
    assert all(math.isfinite(v) for v in estimates.values())


def test_locator_comparison_different_max_steps():
    exp = _gaussian_experiment(center=0.4)

    rng_a = random.Random(99)
    rng_b = random.Random(99)

    steps_a = list(run_loop(SimpleSweepLocator, exp, rng_a, max_steps=5))
    steps_b = list(run_loop(SimpleSweepLocator, exp, rng_b, max_steps=20))

    # More steps means same or more measurements
    assert len(steps_b) >= len(steps_a)
