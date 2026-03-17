from __future__ import annotations

import random

import polars as pl

from nvision.core import TrueSignal
from nvision.core.experiment import CoreExperiment
from nvision.core.runner import Runner
from nvision.sim.gen.core_generators import NVCenterCoreGenerator, OnePeakCoreGenerator
from nvision.sim.locs.core import SimpleSweepLocator


def _run_batch(generator, repeats: int = 2, max_steps: int = 30) -> pl.DataFrame:
    """Run a small simulation batch and return a finalize DataFrame."""
    rng_seed = 42
    rows = []
    for i in range(repeats):
        rng = random.Random(rng_seed + i)
        true_signal = generator.generate(rng)
        x_min, x_max = true_signal.parameters[0].bounds  # Use frequency bounds
        for p in true_signal.parameters:
            if "frequency" in p.name:
                x_min, x_max = p.bounds
                break
        experiment = CoreExperiment(
            true_signal=true_signal,
            noise=None,
            x_min=x_min,
            x_max=x_max,
        )
        runner = Runner()
        steps = 0
        for _ in runner.run(SimpleSweepLocator, experiment, rng, max_steps=max_steps):
            steps += 1
        rows.append({"repeat": i, "steps": steps})
    return pl.DataFrame(rows)


def test_one_peak_run_completes():
    gen = OnePeakCoreGenerator(x_min=0.0, x_max=1.0, peak_type="gaussian")
    df = _run_batch(gen, repeats=2)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 2
    assert df["steps"].min() > 0


def test_nv_center_run_completes():
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    df = _run_batch(gen, repeats=2)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 2
    assert df["steps"].min() > 0


def test_generator_produces_true_signal_not_scan_batch():
    rng = random.Random(0)
    gen = OnePeakCoreGenerator()
    result = gen.generate(rng)
    assert isinstance(result, TrueSignal), f"Expected TrueSignal, got {type(result)}"
