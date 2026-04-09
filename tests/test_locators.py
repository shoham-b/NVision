from __future__ import annotations

import random

from nvision import (
    CompositeNoise,
    CompositeOverFrequencyNoise,
    CompositeOverProbeNoise,
)
from nvision import CoreExperiment
from nvision import Locator
from nvision import (
    NVCenterCoreGenerator,
    OnePeakCoreGenerator,
    TwoPeakCoreGenerator,
)
from nvision import (
    OverFrequencyGaussianNoise,
    OverFrequencyOutlierSpikes,
    OverProbeDriftNoise,
)
from nvision import SimpleSweepLocator
from nvision import run_loop


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


def test_simple_sweep_create_classmethod():
    loc = SimpleSweepLocator.create(max_steps=10)
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
