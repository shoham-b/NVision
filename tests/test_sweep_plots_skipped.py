"""SimpleSweep / SimpleSobol figures are skipped by default; their entry, series and metrics are kept."""

import queue

import pytest
from rich.console import Console

from nvision.cli import defaults as cli_defaults
from nvision.cli.monitor import ProgressMonitor
from nvision.runner import plots as plots_module
from nvision.runner.executor import run_task
from nvision.runner.task_builder import TaskListBuildConfig, build_task_list
from nvision.tools.utils import NVISION_RNG_SEED

GEN = "NVCenter-voigt-w0.50MHz-c0.10-si0.60MHz-hfn14"
NOISE = "Gauss(0.002)"


def _run_one_repeat(tmp_path, strategy: str):
    for sub in ("scans", "bayes", "cache"):
        (tmp_path / sub).mkdir(exist_ok=True)
    pq: queue.Queue = queue.Queue()
    monitor = ProgressMonitor(Console(), pq, log_incoming=None, live_mode=False)
    tasks, _ = build_task_list(
        TaskListBuildConfig(
            repeats=1,
            seed=NVISION_RNG_SEED,
            out_dir=tmp_path,
            scans_dir=tmp_path / "scans",
            bayes_dir=tmp_path / "bayes",
            cache_dir=tmp_path / "cache",
            log_queue=queue.Queue(),
            progress_queue=pq,
            log_level_value=30,
            loc_max_steps=cli_defaults.DEFAULT_LOC_MAX_STEPS,
            sweep_max_steps=None,
            loc_timeout_s=cli_defaults.DEFAULT_LOC_TIMEOUT_S,
            no_cache=True,
            ignore_cache_strategy=None,
            require_cache=False,
            filter_category=None,
            filter_strategy=None,
            filter_generator=None,
            filter_noise=None,
            filter_signal=None,
            dry_run=True,
            combination_names=[(GEN, NOISE, strategy)],
        ),
        monitor=monitor,
    )
    (entries, main_row), *_ = run_task(tasks[0])
    return entries, main_row


@pytest.mark.parametrize("strategy", ["SimpleSweep", "SimpleSobol"])
def test_sweep_plots_skipped_by_default_but_data_kept(tmp_path, monkeypatch, strategy):
    monkeypatch.setattr(plots_module, "NVISION_PLOT_SWEEP_STRATEGIES", False)
    entries, main_row = _run_one_repeat(tmp_path, strategy)

    assert len(entries) == 1
    scan = entries[0]
    assert scan["type"] == "scan"
    assert scan["plot_skipped"] is True
    assert "path" not in scan
    assert "_bytes" not in scan
    # Data the UI's Highlights view and the metrics need is still there.
    assert scan.get("series"), "per-step (error, uncertainty) series must be kept"
    assert scan.get("true_params")
    assert main_row.get("strategy") == strategy
    assert main_row.get("final_steps") is not None


def test_sweep_plots_built_when_flag_on(tmp_path, monkeypatch):
    monkeypatch.setattr(plots_module, "NVISION_PLOT_SWEEP_STRATEGIES", True)
    entries, _ = _run_one_repeat(tmp_path, "SimpleSweep")

    scan = next(e for e in entries if e["type"] == "scan")
    # (run_task strips the heavy `_bytes` payload from returned entries; the path is what marks
    # a figure as built.)
    assert scan["path"]
    assert "plot_skipped" not in scan
