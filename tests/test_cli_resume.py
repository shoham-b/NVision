from __future__ import annotations

import queue
from pathlib import Path

from rich.console import Console

from nvision.cli.monitor import ProgressMonitor
from nvision.cli.run import _find_latest_session_log, _parse_started_combos_from_log
from nvision.runner.task_builder import TaskListBuildConfig, build_task_list


def test_parse_started_combos_from_log(tmp_path: Path) -> None:
    log_file = tmp_path / "nvision-run-2026-09-22_10-00-00.log"
    log_content = (
        "2026-09-22 10:00:00 INFO nvision: Starting simulations...\n"
        "2026-09-22 10:00:01 INFO nvision.runner.executor: "
        "Running task: GenA/NoiseA/StratA (50 total repeats, 0 loaded from cache)\n"
        "2026-09-22 10:00:02 INFO nvision.runner.executor: "
        "Running task: GenB/NoiseB/StratB (50 total repeats, 0 loaded from cache)\n"
        "2026-09-22 10:00:03 WARNING nvision: Run interrupted by user (Ctrl-C).\n"
    )
    log_file.write_text(log_content, encoding="utf-8")

    parsed = _parse_started_combos_from_log(log_file)
    assert parsed == {
        ("GenA", "NoiseA", "StratA"),
        ("GenB", "NoiseB", "StratB"),
    }


def test_find_latest_session_log(tmp_path: Path) -> None:
    # Empty dir
    assert _find_latest_session_log(tmp_path) is None

    # Irrelevant log
    other_log = tmp_path / "nvision-run-2026-09-22_09-00-00.log"
    other_log.write_text("no tasks here\n", encoding="utf-8")

    # Real log
    real_log = tmp_path / "nvision-run-2026-09-22_10-00-00.log"
    real_log.write_text("Running task: G/N/S (10 repeats)\n", encoding="utf-8")

    found = _find_latest_session_log(tmp_path)
    assert found == real_log


def test_task_builder_resume_cache_flag(tmp_path: Path) -> None:
    monitor = ProgressMonitor(Console(), queue.Queue(), log_incoming=None, error_incoming=None, live_mode=False)

    combos = [
        ("NVCenter-lorentzian", "Gauss(0.0)", "SimpleSweep-3perFWHM"),
        ("NVCenter-lorentzian", "Gauss(0.05)", "SimpleSweep-3perFWHM"),
    ]
    # Session where only the first combo ran, not the second
    ran_session = {combos[0]}

    cfg = TaskListBuildConfig(
        repeats=10,
        seed=123,
        out_dir=tmp_path,
        scans_dir=tmp_path / "scans",
        bayes_dir=tmp_path / "bayes",
        cache_dir=tmp_path / "cache",
        log_queue=queue.Queue(),
        progress_queue=queue.Queue(),
        log_level_value=20,
        loc_max_steps=100,
        sweep_max_steps=None,
        loc_timeout_s=100,
        no_cache=False,
        ignore_cache_strategy=None,
        require_cache=False,
        filter_category=None,
        filter_strategy=None,
        filter_generator=None,
        filter_noise=None,
        filter_signal=None,
        combination_names=combos,
        ran_in_resume_session=ran_session,
    )

    tasks, _ = build_task_list(cfg, monitor)
    assert len(tasks) == 2

    for task in tasks:
        triple = (task.generator_name, task.noise_name, task.strategy_name)
        if triple in ran_session:
            assert task.use_cache is True, f"{triple} should use cache (was in session)"
        else:
            assert task.use_cache is False, f"{triple} should bypass cache (was unrun in session)"


def test_resume_chain_roundtrips_through_cache_db(tmp_path: Path) -> None:
    from nvision.cache import CacheBridge
    from nvision.cli.run import _load_resume_chain, _save_resume_chain

    store = CacheBridge(tmp_path / "cache").nv_center
    assert _load_resume_chain(store) == set()
    chain = {("NVCenter-lorentzian", "Gauss(0.0)", "Bayesian-SBED")}
    _save_resume_chain(store, chain)
    assert _load_resume_chain(store) == chain
    _save_resume_chain(store, set())
    assert _load_resume_chain(store) == set()
    store.close()


def test_combos_written_since_filters_by_local_time() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest
    import typer

    from nvision.cli.run import _combos_written_since

    now = datetime.now().astimezone()
    fmt = "%Y-%m-%d %H:%M:%S"

    def combo(name: str, when: datetime) -> dict:
        return {
            "generator": name,
            "noise": "n",
            "strategy": "s",
            "updated_at": when.astimezone(UTC).strftime(fmt),
        }

    class FakeBridge:
        def list_combinations_with_updated_at(self) -> list[dict]:
            return [combo("old", now - timedelta(hours=10)), combo("new", now - timedelta(hours=1))]

    since = (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
    assert _combos_written_since(FakeBridge(), since) == {("new", "n", "s")}  # type: ignore[arg-type]
    with pytest.raises(typer.BadParameter):
        _combos_written_since(FakeBridge(), "last night")  # type: ignore[arg-type]
