import logging
from pathlib import Path

import polars as pl

from nvision.tools.artifacts import locator_results_path, merge_locator_results_with_existing


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

    # Verify that they merged successfully and measurements column is aligned.
    # 'measurements' is a known-integer count column, so mismatches are
    # normalized to Int64 rather than promoted to Float64.
    assert len(merged_df) == 1
    assert merged_df.schema["measurements"] == pl.Int64
    assert merged_df.get_column("measurements")[0] == 89


def test_merge_locator_results_keeps_other_combinations_even_with_no_cache(tmp_path: Path):
    log = logging.getLogger("test_merge")
    out_dir = tmp_path

    # Old results contains two different combinations:
    # 1. NVCenter-lorentzian, strategy Bayesian
    # 2. NV-Center-voigt, strategy Sweep
    old_df = pl.DataFrame(
        {
            "generator": ["NVCenter-lorentzian", "NVCenter-voigt"],
            "noise": ["Gauss(0.01)", "Gauss(0.01)"],
            "strategy": ["Bayesian-SBED-NoSweep", "Sweep"],
            "max_steps": [150, 100],
            "seed": [12345, 12345],
            "attempt": [1, 1],
            "measurements": [45, 30],
        }
    )
    old_df.write_csv(locator_results_path(out_dir))

    # New results updates ONLY combination 1 (NVCenter-lorentzian, strategy Bayesian) with a new repeat/attempt count
    new_df = pl.DataFrame(
        {
            "generator": ["NVCenter-lorentzian"],
            "noise": ["Gauss(0.01)"],
            "strategy": ["Bayesian-SBED-NoSweep"],
            "max_steps": [150],
            "seed": [12345],
            "attempt": [2],
            "measurements": [55],
        }
    )

    # Execute merge with no_cache=True (simulating the run --no-cache CLI command)
    merged_df = merge_locator_results_with_existing(new_df, out_dir, log, no_cache=True)

    # Verify that the old combination 2 (NVCenter-voigt, Sweep) is PRESERVED,
    # but the old combination 1 (Bayesian) is overwritten/discarded, even though the new attempt was 2 instead of 1.
    assert len(merged_df) == 2
    # The new one for Bayesian is present
    bayesian_rows = merged_df.filter(pl.col("strategy") == "Bayesian-SBED-NoSweep")
    assert len(bayesian_rows) == 1
    assert bayesian_rows.get_column("attempt")[0] == 2
    assert bayesian_rows.get_column("measurements")[0] == 55

    # The Sweep row is preserved
    sweep_rows = merged_df.filter(pl.col("strategy") == "Sweep")
    assert len(sweep_rows) == 1
    assert sweep_rows.get_column("attempt")[0] == 1
    assert sweep_rows.get_column("measurements")[0] == 30


def test_merge_plot_manifest_keeps_other_combinations_and_overrides_updated_completely(tmp_path: Path):
    log = logging.getLogger("test_merge")
    out_dir = tmp_path

    # Prepare some plot files on disk
    old_plot_1 = out_dir / "plot1.html"
    old_plot_2 = out_dir / "plot2.html"
    old_plot_1.touch()
    old_plot_2.touch()

    # Pre-populate plots_manifest.json with two different combinations:
    # 1. NVCenter-lorentzian, strategy Bayesian, repeat 1
    # 2. NVCenter-voigt, strategy Sweep, repeat 1
    import json

    from nvision.tools.artifacts import merge_run_plot_manifest_with_existing_on_disk, plots_manifest_path

    old_manifest = [
        {
            "type": "scan",
            "generator": "NVCenter-lorentzian",
            "noise": "Gauss(0.01)",
            "strategy": "Bayesian",
            "max_steps": 150,
            "seed": 12345,
            "repeat": 1,
            "path": "plot1.html",
        },
        {
            "type": "scan",
            "generator": "NVCenter-voigt",
            "noise": "Gauss(0.01)",
            "strategy": "Sweep",
            "max_steps": 100,
            "seed": 12345,
            "repeat": 1,
            "path": "plot2.html",
        },
    ]
    plots_manifest_path(out_dir).write_text(json.dumps(old_manifest), encoding="utf-8")

    # New plot_manifest updates ONLY combination 1 (Bayesian) with a different repeat/attempt number
    new_plot = out_dir / "new_plot.html"
    new_plot.touch()
    new_manifest = [
        {
            "type": "scan",
            "generator": "NVCenter-lorentzian",
            "noise": "Gauss(0.01)",
            "strategy": "Bayesian",
            "max_steps": 150,
            "seed": 12345,
            "repeat": 2,
            "path": "new_plot.html",
        }
    ]

    # Execute merge with no_cache=True (simulating the run --no-cache CLI command)
    merge_run_plot_manifest_with_existing_on_disk(new_manifest, out_dir, log, no_cache=True)

    # Verify that the Sweep plot is preserved, and the old Bayesian plot is removed from disk and manifest.
    assert len(new_manifest) == 2

    # Check that old Bayesian plot (plot1.html) was deleted from disk
    assert not old_plot_1.exists()
    # Check that Sweep plot (plot2.html) is still on disk
    assert old_plot_2.exists()

    # The Sweep entry is preserved in new_manifest
    sweep_entries = [e for e in new_manifest if e.get("strategy") == "Sweep"]
    assert len(sweep_entries) == 1
    assert sweep_entries[0]["repeat"] == 1

    # The old Bayesian entry is gone, replaced by the new Bayesian entry
    bayesian_entries = [e for e in new_manifest if e.get("strategy") == "Bayesian"]
    assert len(bayesian_entries) == 1
    assert bayesian_entries[0]["repeat"] == 2
    assert bayesian_entries[0]["path"] == "new_plot.html"
