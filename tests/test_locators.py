from __future__ import annotations

import random

import numpy as np

from nvision import (
    CompositeNoise,
    CompositeOverFrequencyNoise,
    CompositeOverProbeNoise,
    CoreExperiment,
    GaussianModel,
    Locator,
    NVCenterCoreGenerator,
    OnePeakCoreGenerator,
    OverFrequencyGaussianNoise,
    OverFrequencyOutlierSpikes,
    OverProbeDriftNoise,
    SimpleSweepLocator,
    TwoPeakCoreGenerator,
    run_loop,
)
from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter


def _make_experiment(generator, rng: random.Random, noise=None) -> CoreExperiment:
    true_signal = generator.generate(rng)
    x_min, x_max = None, None
    for name in true_signal.parameter_names:
        if "frequency" in name:
            x_min, x_max = true_signal.get_param_bounds(name)
            break
    assert x_min is not None
    return CoreExperiment(true_signal=true_signal, noise=noise, x_min=x_min, x_max=x_max)


def test_simple_sweep_locator_is_core_locator():
    assert issubclass(SimpleSweepLocator, Locator)


def _dummy_belief(model):
    grid = np.linspace(0.0, 1.0, 10)
    posterior = np.ones(10) / 10
    parameters = [
        GridParameter(name=name, bounds=(0.0, 1.0), grid=grid, posterior=posterior) for name in model.parameter_names()
    ]
    return GridMarginalDistribution(model=model, parameters=parameters)


def test_simple_sweep_create_classmethod():
    model = GaussianModel()
    belief = _dummy_belief(model)
    loc = SimpleSweepLocator.create(belief=belief, signal_model=model, max_steps=10)
    assert isinstance(loc, SimpleSweepLocator)


def test_locator_runs_on_one_peak():
    rng = random.Random(123)
    gen = OnePeakCoreGenerator(x_min=0.0, x_max=1.0)
    exp = _make_experiment(gen, rng)
    steps = list(run_loop(SimpleSweepLocator, exp, rng, max_steps=30))
    assert len(steps) > 0
    assert len(steps) <= 30


def test_locator_runs_on_two_peak():
    rng = random.Random(7)
    gen = TwoPeakCoreGenerator(x_min=0.0, x_max=1.0)
    exp = _make_experiment(gen, rng)
    steps = list(run_loop(SimpleSweepLocator, exp, rng, max_steps=30))
    assert len(steps) > 0


def test_locator_runs_on_nv_center():
    rng = random.Random(99)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    exp = _make_experiment(gen, rng)
    steps = list(run_loop(SimpleSweepLocator, exp, rng, max_steps=30))
    assert len(steps) > 0


def test_locator_with_gaussian_noise():
    rng = random.Random(5)
    noise = CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.05)]))
    gen = OnePeakCoreGenerator(x_min=0.0, x_max=1.0)
    exp = _make_experiment(gen, rng, noise=noise)
    steps = list(run_loop(SimpleSweepLocator, exp, rng, max_steps=30))
    assert len(steps) > 0


def test_locator_with_heavy_noise():
    rng = random.Random(3)
    noise = CompositeNoise(
        over_frequency_noise=CompositeOverFrequencyNoise(
            [OverFrequencyGaussianNoise(0.1), OverFrequencyOutlierSpikes(0.02, 0.5)]
        ),
        over_probe_noise=CompositeOverProbeNoise([OverProbeDriftNoise(0.05)]),
    )
    gen = OnePeakCoreGenerator(x_min=0.0, x_max=1.0)
    exp = _make_experiment(gen, rng, noise=noise)
    steps = list(run_loop(SimpleSweepLocator, exp, rng, max_steps=30))
    assert len(steps) > 0
