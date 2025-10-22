from __future__ import annotations

import math
import random

from nvcenter.sim import (
    CompositeNoise,
    DataBatch,
    DriftNoise,
    ExperimentRunner,
    FluorescenceCount,
    GaussianNoise,
    OutlierSpikes,
    PoissonNoise,
    RabiEstimate,
    RabiGenerator,
    T1Estimate,
    T1Generator,
)


def test_noise_composition_deterministic_and_length():
    y = [0.1 * i for i in range(50)]
    t = list(range(len(y)))
    data = DataBatch(time_points=t, signal_values=y, meta={})
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    noise = CompositeNoise([GaussianNoise(0.1), DriftNoise(0.05), OutlierSpikes(0.1, 0.5)])
    d1 = noise.apply(data, rng1)
    d2 = noise.apply(data, rng2)
    assert len(d1.signal_values) == len(y)
    assert len(d2.signal_values) == len(y)
    assert d1.signal_values == d2.signal_values  # same seed -> same result


def test_poisson_noise_non_negative():
    y = [0.0, 0.1, 1.0, 2.5]
    t = list(range(len(y)))
    data = DataBatch(time_points=t, signal_values=y, meta={})
    rng = random.Random(123)
    p = PoissonNoise(scale=50.0)
    out = p.apply(data, rng)
    assert all(v >= 0 for v in out.signal_values)


def test_rabi_estimate_noiseless_close():
    gen = RabiGenerator(n_points=400, duration=10.0, amplitude=0.8, frequency=1.25, phase=0.3, offset=0.2)
    data = gen.generate(random.Random(7))
    est = RabiEstimate().estimate(data)
    assert math.isclose(est["offset"], 0.2, rel_tol=0.05, abs_tol=0.05)
    assert math.isclose(est["amplitude"], 0.8, rel_tol=0.1, abs_tol=0.1)
    assert math.isclose(est["frequency"], 1.25, rel_tol=0.15, abs_tol=0.2)


def test_t1_estimate_noiseless_close():
    gen = T1Generator(n_points=300, duration=6.0, A=1.0, tau=2.0, offset=0.1)
    data = gen.generate(random.Random(11))
    est = T1Estimate().estimate(data)
    assert math.isclose(est["offset"], 0.1, rel_tol=0.05, abs_tol=0.05)
    assert math.isclose(est["tau"], 2.0, rel_tol=0.2, abs_tol=0.2)
    assert math.isclose(est["A"], 1.0, rel_tol=0.2, abs_tol=0.2)


def test_runner_sweep_produces_dataframe():
    runner = ExperimentRunner(rng_seed=123)
    gen = RabiGenerator()
    noises = [
        None,
        CompositeNoise([GaussianNoise(0.05)]),
        CompositeNoise([GaussianNoise(0.1), DriftNoise(0.1), OutlierSpikes(0.02, 0.5)]),
    ]
    strategies = [FluorescenceCount(), RabiEstimate()]
    df = runner.sweep(gen, noises, strategies, repeats=3)
    # Expect one row per (noise, strategy)
    assert df.height == len(noises) * len(strategies)
    # Check presence of rmse column
    assert "rmse" in df.columns
