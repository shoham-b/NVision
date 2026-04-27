"""Tests for Gaussian Process and Laplace locators."""

from __future__ import annotations

import numpy as np

from nvision import GaussianModel
from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter
from nvision.models.observation import Observation
from nvision.sim.locs.bayesian.gaussian_process_locator import GaussianProcessLocator
from nvision.sim.locs.bayesian.laplace_locator import LaplaceLocator


def _dummy_belief(model):
    grid = np.linspace(0.0, 1.0, 10)
    posterior = np.ones(10) / 10
    parameters = [
        GridParameter(name=name, bounds=(0.0, 1.0), grid=grid, posterior=posterior) for name in model.parameter_names()
    ]
    return GridMarginalDistribution(model=model, parameters=parameters)

def test_gaussian_process_locator_create():
    model = GaussianModel()
    belief = _dummy_belief(model)
    loc = GaussianProcessLocator.create(belief=belief, signal_model=model, max_steps=10)
    assert isinstance(loc, GaussianProcessLocator)
    assert loc.max_steps == 10
    assert loc.USES_SWEEP_MAX_STEPS is True

def test_laplace_locator_create():
    model = GaussianModel()
    belief = _dummy_belief(model)
    loc = LaplaceLocator.create(belief=belief, signal_model=model, max_steps=10, initial_sweep_steps=5)
    assert isinstance(loc, LaplaceLocator)
    assert loc.max_steps == 10
    assert loc.initial_sweep_steps == 5

def test_gaussian_process_locator_basic_flow():
    model = GaussianModel()
    belief = _dummy_belief(model)
    loc = GaussianProcessLocator.create(belief=belief, signal_model=model, max_steps=5)

    for _i in range(5):
        assert not loc.done()
        x = loc.next()
        loc.observe(Observation(x, 0.5))

    assert loc.done()
    result = loc.result()
    assert isinstance(result, dict)
    assert len(result) > 0

def test_laplace_locator_basic_flow():
    model = GaussianModel()
    belief = _dummy_belief(model)
    loc = LaplaceLocator.create(belief=belief, signal_model=model, max_steps=6, initial_sweep_steps=4)

    # Sweep phase
    for _i in range(4):
        assert not loc.done()
        x = loc.next()
        loc.observe(Observation(x, 0.5))

    # Inference phase
    for _i in range(2):
        assert not loc.done()
        x = loc.next()
        loc.observe(Observation(x, 0.5))

    assert loc.done()
    result = loc.result()
    assert isinstance(result, dict)
    assert len(result) > 0
