from __future__ import annotations

import polars as pl

from nvision.sim.core import (
    CompositeNoise,
    CompositeOverFrequencyNoise,
    CompositeOverProbeNoise,
)
from nvision.sim.gen import (
    GaussianManufacturer,
    NVCenterGenerator,
    OnePeakGenerator,
    TwoPeakGenerator,
)
from nvision.sim.loc_runner import LocatorRunner
from nvision.sim.locs.nv_center.sweep_locator import NVCenterSweepLocator
from nvision.sim.locs.one_peak import OnePeakGridLocator, OnePeakSweepLocator
from nvision.sim.locs.two_peak import TwoPeakGridLocator, TwoPeakSweepLocator
from nvision.sim.noises import (
    OverFrequencyGaussianNoise,
    OverFrequencyOutlierSpikes,
    OverProbeDriftNoise,
)
from tests.locator_compat import LegacyLocatorShim


def test_locator_sweep_dataframe_shape():
    rng_seed = 123
    runner = LocatorRunner(rng_seed=rng_seed)

    generators = [
        ("OnePeak", OnePeakGenerator(manufacturer=GaussianManufacturer())),
        (
            "TwoPeak",
            TwoPeakGenerator(
                manufacturer_left=GaussianManufacturer(),
                manufacturer_right=GaussianManufacturer(),
            ),
        ),
        ("NVCenter", NVCenterGenerator(variant="zeeman")),
    ]
    noises = [
        ("NoNoise", None),
        (
            "Gauss",
            CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.05)])),
        ),
        (
            "Heavy",
            CompositeNoise(
                over_frequency_noise=CompositeOverFrequencyNoise(
                    [OverFrequencyGaussianNoise(0.1), OverFrequencyOutlierSpikes(0.02, 0.5)]
                ),
                over_probe_noise=CompositeOverProbeNoise([OverProbeDriftNoise(0.05)]),
            ),
        ),
    ]
    strategies = [
        ("OnePeak-Grid", LegacyLocatorShim(OnePeakGridLocator(n_points=21))),
        (
            "OnePeak-Sweep",
            LegacyLocatorShim(OnePeakSweepLocator(coarse_points=20, refine_points=10)),
        ),
        ("TwoPeak-Grid", LegacyLocatorShim(TwoPeakGridLocator(coarse_points=25))),
        (
            "TwoPeak-Sweep",
            LegacyLocatorShim(TwoPeakSweepLocator(coarse_points=25, refine_points=10)),
        ),
        (
            "NVCenter-Sweep",
            LegacyLocatorShim(NVCenterSweepLocator(coarse_points=30, refine_points=10)),
        ),
    ]

    df = runner.sweep(generators, strategies, noises, repeats=3, max_steps=100)

    assert isinstance(df, pl.DataFrame)
    expected_rows = len(generators) * len(noises) * len(strategies)
    assert df.height == expected_rows
    # Expect at least one metric column
    assert any(c for c in df.columns if c not in ("generator", "noise", "strategy"))
    assert "measurements" in df.columns
    assert "duration_ms" in df.columns


def test_gridscan_converges_noiseless_single_peak_reasonable_error():
    rng_seed = 7
    runner = LocatorRunner(rng_seed=rng_seed)
    gen = OnePeakGenerator(manufacturer=GaussianManufacturer())
    df = runner.sweep(
        generators=[("OnePeak", gen)],
        strategies=[("OnePeak-Grid", LegacyLocatorShim(OnePeakGridLocator(n_points=21)))],
        noises=[("NoNoise", None)],
        repeats=3,
        max_steps=100,
    )
    # abs_err_x should exist and be finite
    assert "abs_err_x" in df.columns
    assert df.select(pl.col("abs_err_x").is_finite().all()).item()
    assert df.select(pl.col("measurements").is_finite().all()).item()
    assert df.select(pl.col("duration_ms").is_finite().all()).item()
