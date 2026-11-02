"""Lab Bayesian-run processing: replayed stop for full-budget runs, kept stop for precision-stopped runs."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.io as sio

from nvision.metrics.lab_process import process_bayesian_runs


def _save_run(path: Path, *, no_stop: int, std: list[float], stop_reason: str, step_count: int) -> None:
    n = len(std)
    history = np.column_stack([np.full(n, 100.0), np.array(std), np.zeros(n)])
    state = {
        "history": history,
        "stepCount": step_count,
        "stopReason": stop_reason,
        "meanMHz": 100.0,
        "stdMHz": std[step_count - 1],
        "credibleIntervalMHz": np.array([99.0, 101.0]),
    }
    config = {"scanRangeMHz": np.array([80, 120]), "targetStdMHz": 1, "minimumStopSteps": 2}
    sio.savemat(path, {"myStruct": {"BayesianESR": {"config": config, "state": state, "noConvergenceStop": no_stop}}})


def test_full_budget_run_is_replayed(tmp_path):
    _save_run(tmp_path / "a.mat", no_stop=1, std=[5.0, 3.0, 0.9, 0.5], stop_reason="budget", step_count=4)
    out = process_bayesian_runs(tmp_path)
    row = pl.read_csv(out / "window40MHz_statistics_perRun.csv").row(0, named=True)
    assert (row["converged"], row["stepsToConvergence"]) == (1, 3)  # first step with std <= 1
    assert row["finalStdMHz"] == pytest.approx(0.9)
    assert pl.read_csv(out / "window40MHz_self_reported_std_vs_step.csv").height == 4


def test_precision_stopped_run_keeps_its_own_stop(tmp_path):
    _save_run(tmp_path / "a.mat", no_stop=0, std=[5.0, 0.8, 0.5], stop_reason="budget", step_count=3)
    row = pl.read_csv(process_bayesian_runs(tmp_path) / "window40MHz_statistics_perRun.csv").row(0, named=True)
    assert (row["converged"], row["stepsToConvergence"]) == (0, 3)  # saved stop reason wins over a replay


def test_lab_script_output_is_not_overwritten(tmp_path):
    (tmp_path / "convergence_statistics").mkdir()
    (tmp_path / "convergence_statistics" / "window_summary.csv").write_text("x\n")
    _save_run(tmp_path / "a.mat", no_stop=1, std=[1.0, 0.5], stop_reason="budget", step_count=2)
    with pytest.raises(ValueError, match="not overwriting"):
        process_bayesian_runs(tmp_path)
