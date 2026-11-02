"""Lab study payload: shape and calibration arithmetic on a tiny synthetic convergence_statistics folder."""

from pathlib import Path

import polars as pl
import pytest

from nvision.metrics.lab_study import lab_study_payload


def _write_window(root: Path, w: int) -> None:
    pl.DataFrame(
        {
            "converged": [1, 1, 1, 0],
            "finalMeanMHz": [100.0, 101.0, 99.0, 150.0],
            "finalStdMHz": [0.5, 0.5, 0.5, 0.5],
            "stepsToConvergence": [10, 20, 30, 40],
            "credibleIntervalMHz_1": [99.0, 100.0, 98.0, 149.0],
            "credibleIntervalMHz_2": [101.0, 102.0, 100.0, 151.0],
        }
    ).write_csv(root / f"window{w}MHz_statistics_perRun.csv")
    pl.DataFrame(
        {
            "measurementStep": [1, 2],
            "numContributingRuns": [4, 4],
            "medianPosteriorStdMHz": [5.0, 2.0],
            "q25PosteriorStdMHz": [4.0, 1.0],
            "q75PosteriorStdMHz": [6.0, 3.0],
        }
    ).write_csv(root / f"window{w}MHz_self_reported_std_vs_step.csv")
    pl.DataFrame({"measurementStep": [1, 2], "numContributingRuns": [4, 4], "acrossRunStdMHz": [3.0, 1.5]}).write_csv(
        root / f"window{w}MHz_resonance_std_vs_step_retained.csv"
    )


def test_payload_summarises_converged_runs_only(tmp_path):
    _write_window(tmp_path, 40)
    out = lab_study_payload(tmp_path)
    (row,) = out["windows"]
    assert out["reference_mhz"] == pytest.approx(100.0)
    assert (row["n_runs"], row["n_converged"]) == (4, 3)
    assert row["steps_median"] == 20
    assert row["across_run_std_mhz"] == pytest.approx(1.0)
    assert row["overconfidence"] == pytest.approx(2.0)
    assert row["within_1sigma"] == pytest.approx(1 / 3)  # |z|=0 for the run at the reference only
    assert out["curves"]["40"]["across_run_std"] == [3.0, 1.5]


def test_missing_files_raise(tmp_path):
    with pytest.raises(ValueError, match="No window"):
        lab_study_payload(tmp_path)
