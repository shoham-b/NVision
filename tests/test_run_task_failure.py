"""A failed task must not stop the successful tasks' results from being written (and shown)."""

from __future__ import annotations

import json

import polars as pl
from typer.testing import CliRunner

from nvision.cli.main import app
from nvision.runner import run_task as real_run_task


def test_failed_task_still_writes_the_other_results(tmp_path, monkeypatch):
    def flaky(task, **kwargs):
        if "3perfwhm" in task.slug.lower():
            raise ValueError("stage metrics exploded")
        return real_run_task(task, **kwargs)

    monkeypatch.setattr("nvision.cli.run.run_task", flaky)
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--out",
            str(tmp_path),
            "--repeats",
            "1",
            "--runners",
            "1",
            "--no-progress",
            "--filter-generator",
            "NVCenter-lorentzian",
            "--filter-noise",
            "Gauss",
            "--filter-strategy",
            "SimpleSweep-2perFWHM,SimpleSweep-3perFWHM",
            "--logs-root",
            str(tmp_path / "logs"),
        ],
    )

    assert result.exit_code == 1, result.output
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "error"
    strategies = pl.read_parquet(tmp_path / "locator_results.parquet")["strategy"].unique().to_list()
    assert strategies == ["SimpleSweep-2perFWHM"]
    assert "NVCenter-lorentzian" in result.output or "stage metrics exploded" in result.output
