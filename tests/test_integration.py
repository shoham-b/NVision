import os
import tempfile

import polars as pl

from nvision.sim.gen import (
    NVCenterGenerator,
)
from nvision.sim.runner_v2 import LocatorRunnerV2
from nvision.sim.locs.nv_center.sweep_locator_v2 import NVCenterSweepLocatorV2


def test_basic_run_no_artifacts():
    """Test a basic 1-repeat run with 2 steps limitation does not generate artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.environ["NVISION_CACHE_DIR"] = tmpdir
        try:
            os.chdir(tmpdir)
            rng_seed = 42
            runner = LocatorRunnerV2(rng_seed=rng_seed)
            gen = NVCenterGenerator(variant="zeeman")
            strat = NVCenterSweepLocatorV2(coarse_points=5, refine_points=2)
            df = runner.sweep(
                generators=[("NVCenter", gen)],
                strategies=[("Sweep", strat)],
                noises=[("NoNoise", None)],
                repeats=1,
                max_steps=2,
            )
            assert isinstance(df, pl.DataFrame)
            assert df.height == 1
            assert df["repeats"][0] == 1
            assert "measurements" in df.columns
            assert df["measurements"][0] <= 2
            files = [
                f
                for f in os.listdir(tmpdir)
                if not f.startswith("__")
                and not f.endswith(".db")
                and f not in ("coverage.xml", ".coverage", "htmlcov", ".pytest_cache")
            ]
            assert not files, f"Found unexpected artifacts: {files}"
        finally:
            os.chdir(original_cwd)
            if "NVISION_CACHE_DIR" in os.environ:
                del os.environ["NVISION_CACHE_DIR"]
