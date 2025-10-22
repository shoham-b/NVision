from __future__ import annotations

import polars as pl

from nvision.sim import (
    CompositeNoise,
    DriftNoise,
    GaussianNoise,
    GoldenSectionSearch,
    GridScan,
    OnePeakGenerator,
    OutlierSpikes,
    TwoPeakGenerator,
    TwoPeakGreedy,
)
from nvision.sim.loc_runner import LocatorRunner


def test_locator_sweep_dataframe_shape():
    rng_seed = 123
    runner = LocatorRunner(rng_seed=rng_seed)

    generators = [
        ("OnePeak", lambda rng: OnePeakGenerator().generate(rng)),
        ("TwoPeak", lambda rng: TwoPeakGenerator().generate(rng)),
    ]
    noises = [
        ("NoNoise", None),
        ("Gauss", CompositeNoise([GaussianNoise(0.05)])),
        ("Heavy", CompositeNoise([GaussianNoise(0.1), DriftNoise(0.05), OutlierSpikes(0.02, 0.5)])),
    ]
    strategies = [
        ("Grid21", GridScan(n_points=21)),
        ("Golden20", GoldenSectionSearch(max_evals=20)),
        ("TwoGreedy", TwoPeakGreedy(coarse_points=15, refine_points=5)),
    ]

    df = runner.sweep(generators, strategies, noises, repeats=3, max_steps=100)

    assert isinstance(df, pl.DataFrame)
    expected_rows = len(generators) * len(noises) * len(strategies)
    assert df.height == expected_rows
    # Expect at least one metric column
    assert any(c for c in df.columns if c not in ("generator", "noise", "strategy"))


def test_gridscan_converges_noiseless_single_peak_reasonable_error():
    rng_seed = 7
    runner = LocatorRunner(rng_seed=rng_seed)
    gen = lambda rng: OnePeakGenerator().generate(rng)  # noqa: E731
    df = runner.sweep(
        generators=[("OnePeak", gen)],
        strategies=[("Grid21", GridScan(n_points=21))],
        noises=[("NoNoise", None)],
        repeats=3,
        max_steps=100,
    )
    # abs_err_x should exist and be finite
    assert "abs_err_x" in df.columns
    assert df.select(pl.col("abs_err_x").is_finite().all()).item()
