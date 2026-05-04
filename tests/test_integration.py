from __future__ import annotations

import random

import polars as pl

from nvision import (
    CoreExperiment,
    NVCenterCoreGenerator,
    SimpleSweepLocator,
    run_loop,
)


def _run_batch(generator, repeats: int = 2, max_steps: int = 30) -> pl.DataFrame:
    """Run a small simulation batch and return a finalize DataFrame."""
    rng_seed = 42
    rows = []
    for i in range(repeats):
        rng = random.Random(rng_seed + i)
        true_signal = generator.generate(rng)
        # Find frequency parameter bounds for x_min/x_max
        x_min, x_max = true_signal.all_param_bounds()[true_signal.parameter_names[0]]
        for name in true_signal.parameter_names:
            if "frequency" in name:
                x_min, x_max = true_signal.get_param_bounds(name)
                break
        experiment = CoreExperiment(
            true_signal=true_signal,
            noise=None,
            x_min=x_min,
            x_max=x_max,
        )
        steps = 0
        for _ in run_loop(SimpleSweepLocator, experiment, rng, max_steps=max_steps):
            steps += 1
        rows.append({"repeat": i, "steps": steps})
    return pl.DataFrame(rows)


def test_nv_center_run_completes():
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    df = _run_batch(gen, repeats=2)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 2
    assert df["steps"].min() > 0
