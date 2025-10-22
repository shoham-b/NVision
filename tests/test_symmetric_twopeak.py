from __future__ import annotations

import math
import random

import polars as pl

from nvision.sim import (
    CompositeNoise,
    GaussianNoise,
    LocatorRunner,
    TwoPeakGreedy,
    GaussianManufacturer,
)
from nvision.sim.generators import SymmetricTwoPeakGenerator


def _argmax_two_lr(
    f, x_min: float, x_max: float, center: float, n: int = 2001,
) -> tuple[float, float]:
    step = (x_max - x_min) / (n - 1)
    best_left = (-float("inf"), x_min)
    best_right = (-float("inf"), x_max)
    for i in range(n):
        x = x_min + i * step
        y = float(f(x))
        if x <= center and y > best_left[0]:
            best_left = (y, x)
        if x >= center and y > best_right[0]:
            best_right = (y, x)
    return best_left[1], best_right[1]


def test_symmetric_twopeak_gaussian_centers_and_maxima():
    rng = random.Random(123)
    gen = SymmetricTwoPeakGenerator(
        x_min=0.0,
        x_max=1.0,
        center=0.5,
        sep_frac=0.2,
        base=0.0,
        manufacturers=(
            GaussianManufacturer(amplitude=1.0, sigma=0.06),
            GaussianManufacturer(amplitude=0.8, sigma=0.06),
        ),
    )
    scan = gen.generate(rng)
    width = gen.x_max - gen.x_min
    delta = 0.5 * gen.sep_frac * width
    expected = (gen.center - delta, gen.center + delta)
    assert tuple(sorted(scan.truth_positions)) == tuple(sorted(expected))

    x1, x2 = _argmax_two_lr(scan.signal, scan.x_min, scan.x_max, gen.center)
    assert math.isfinite(x1)
    assert math.isfinite(x2)
    # Maxima should be near expected centers within a few grid steps
    tol = (scan.x_max - scan.x_min) / 2000 + 1e-3
    ex1, ex2 = tuple(sorted(expected))
    m1, m2 = x1, x2
    assert abs(m1 - ex1) <= tol
    assert abs(m2 - ex2) <= tol


def test_symmetric_twopeak_runner_with_twopeak_greedy():
    rng_seed = 42
    runner = LocatorRunner(rng_seed=rng_seed)
    generators = [
        (
            "Sym2Gauss",
            SymmetricTwoPeakGenerator(
                x_min=0.0,
                x_max=1.0,
                center=0.5,
                sep_frac=0.2,
                base=0.0,
                manufacturers=(
                    GaussianManufacturer(amplitude=1.0, sigma=0.06),
                    GaussianManufacturer(amplitude=1.0, sigma=0.06),
                ),
            ),
        ),
    ]
    noises = [("NoNoise", None), ("Gauss", CompositeNoise([GaussianNoise(0.05)]))]
    strategies = [("TwoGreedy", TwoPeakGreedy(coarse_points=15, refine_points=5))]

    df = runner.sweep(generators, strategies, noises, repeats=2, max_steps=80)
    assert isinstance(df, pl.DataFrame)
    assert df.height == len(generators) * len(noises) * len(strategies)
    # pair_rmse should exist and be finite
    assert "pair_rmse" in df.columns
    assert df.select(pl.col("pair_rmse").is_finite().all()).item()


def test_symmetric_twopeak_invalid_params_raise():
    rng = random.Random(0)
    # sep too large for given center and domain
    gen = SymmetricTwoPeakGenerator(
        x_min=0.0,
        x_max=1.0,
        center=0.05,
        sep_frac=0.2,
        base=0.0,
        manufacturers=(
            GaussianManufacturer(amplitude=1.0, sigma=0.06),
            GaussianManufacturer(amplitude=1.0, sigma=0.06),
        ),
    )
    import pytest
    with pytest.raises(ValueError, match="sep_frac too large"):
        _ = gen.generate(rng)
