import logging
from pathlib import Path
import polars as pl
from nvision.tools.artifacts import merge_locator_results_with_existing, locator_results_path


def test_merge_locator_results_schema_mismatch(tmp_path: Path):
    log = logging.getLogger("test_merge")

    # Create an existing CSV with Int64 for 'measurements'
    old_df = pl.DataFrame(
        {
            "generator": ["NVCenter-lorentzian"],
            "noise": ["Gauss(0.01)"],
            "strategy": ["Bayesian-SBED-NoSweep"],
            "max_steps": [150],
            "seed": [12345],
            "attempt": [1],
            "measurements": [45],
        }
    )

    out_dir = tmp_path
    csv_path = locator_results_path(out_dir)
    old_df.write_csv(csv_path)

    # Create a new df with Float64 for 'measurements' (or vice versa)
    new_df = pl.DataFrame(
        {
            "generator": ["NVCenter-lorentzian"],
            "noise": ["Gauss(0.01)"],
            "strategy": ["Bayesian-SBED-NoSweep"],
            "max_steps": [150],
            "seed": [12345],
            "attempt": [1],
            "measurements": [89.0],
        }
    )

    # Execute merge
    merged_df = merge_locator_results_with_existing(new_df, out_dir, log)

    # Verify that they merged successfully and measurements column is aligned (Float64)
    assert len(merged_df) == 1
    assert merged_df.schema["measurements"] == pl.Float64
    assert merged_df.get_column("measurements")[0] == 89.0
