import os
import tempfile

import polars as pl

from nvision.sim.gen import (
    GaussianManufacturer,
    NVCenterGenerator,
)
from nvision.sim.loc_runner import LocatorRunner
from nvision.sim.locs.nv_center import NVCenterSequentialBayesianLocatorBatched


def test_basic_run_no_artifacts():
    """Test a basic 1-repeat run with 2 steps limitation does not generate artifacts."""
    # We will run this in a temporary CWD and set cache dir to it
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        # Set environment variable to ensure any global cache uses this temp dir
        # (Assuming NVISION_CACHE_DIR is the variable, or just as a precaution)
        os.environ["NVISION_CACHE_DIR"] = tmpdir
        try:
            os.chdir(tmpdir)

            rng_seed = 42
            runner = LocatorRunner(rng_seed=rng_seed)

            gen_name = "NVCenter"
            # Assuming GaussianManufacturer is separate or defaults are fine
            gen = NVCenterGenerator(manufacturer=GaussianManufacturer(), variant="zeeman")

            noise_name = "NoNoise"
            noise = None

            strat_name = "BatchedBayesian"

            # Using Batched Locator
            strat = NVCenterSequentialBayesianLocatorBatched(
                max_evals=10
            )  # Max evals higher than steps to test limiting by step count

            # Run sweep
            df = runner.sweep(
                generators=[(gen_name, gen)],
                strategies=[(strat_name, strat)],
                noises=[(noise_name, noise)],
                repeats=1,
                max_steps=2,
            )

            assert isinstance(df, pl.DataFrame)
            assert df.height == 1
            assert df["repeats"][0] == 1
            assert "measurements" in df.columns

            # NVCenterSequentialBayesianLocatorBatched has default warmup=10.
            # If max_steps=2, it should stop at 2.
            assert df["measurements"][0] <= 3

            # Verify no files created in the temp dir
            # This confirms that LocatorRunner itself is pure and doesn't write side artifacts
            files = os.listdir(tmpdir)
            # Filter out __pycache__ or similar if they appear
            files = [
                f
                for f in files
                if not f.startswith("__")
                and not f.endswith(".db")
                and f not in ("coverage.xml", ".coverage", "htmlcov", ".pytest_cache")
            ]
            assert not files, f"Found unexpected artifacts: {files}"

        finally:
            os.chdir(original_cwd)
            if "NVISION_CACHE_DIR" in os.environ:
                del os.environ["NVISION_CACHE_DIR"]
