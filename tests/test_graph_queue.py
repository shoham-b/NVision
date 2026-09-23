"""Deferred graph generation: queue protocol, the first-N rule, and an end-to-end deferral round trip."""

import queue
from pathlib import Path

import pytest
from rich.console import Console

from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_keys import combination_base_cache_config
from nvision.cli import defaults as cli_defaults
from nvision.cli.monitor import ProgressMonitor
from nvision.runner import graph_queue
from nvision.runner import plots as plots_module
from nvision.runner.executor import run_task
from nvision.runner.task_builder import TaskListBuildConfig, build_task_list
from nvision.tools.utils import NVISION_RNG_SEED


def _job(slug: str, rid: int) -> graph_queue.GraphJob:
    return graph_queue.GraphJob(
        slug=slug,
        rid=rid,
        combo_kw={},
        target_repeats=1,
        cache_dir=Path("cache"),
        shard_index=None,
        out_dir=Path("."),
        scans_dir=Path("."),
        bayes_dir=Path("."),
        entry_base={},
        main_result_row={},
        current_scan=None,
        history_df=None,
        noise_obj=None,
        strat_obj=None,
        run_result=None,
        sobol_baseline=None,
        simplesweep_baseline=None,
    )


def test_states_are_grouped_by_combination(tmp_path):
    spool = tmp_path / "graph_queue"
    graph_queue.write_state(spool, _job("a_b", 0))
    graph_queue.write_state(spool, _job("a_b", 1))
    graph_queue.write_state(spool, _job("a_b_c", 0))  # a slug that merely starts with another's

    assert [p.name for p in graph_queue.state_paths(spool, "a_b")] == [
        "a_b__r000000.state",
        "a_b__r000001.state",
    ]
    assert graph_queue.slugs_with_state(spool) == {"a_b", "a_b_c"}
    assert graph_queue.has_state(spool, "a_b_c")
    assert not graph_queue.has_state(spool, "missing")


def test_a_ready_combination_is_claimed_by_exactly_one_worker(tmp_path):
    spool = tmp_path / "graph_queue"
    graph_queue.mark_ready(spool, "combo_1")
    graph_queue.mark_ready(spool, "combo_2")

    first = graph_queue.claim_ready(spool)
    second = graph_queue.claim_ready(spool)
    third = graph_queue.claim_ready(spool)

    assert {first, second} == {"combo_1", "combo_2"}
    assert third is None
    assert graph_queue.pending_counts(spool)[0] == 2  # claimed = still in progress


def test_prepare_spool_returns_abandoned_claims_and_clears_the_done_marker(tmp_path):
    spool = tmp_path / "graph_queue"
    graph_queue.mark_ready(spool, "combo_1")
    assert graph_queue.claim_ready(spool) == "combo_1"  # a worker that then died
    graph_queue.mark_producers_done(spool)

    graph_queue.prepare_spool(spool)

    assert not graph_queue.producers_done(spool)
    assert graph_queue.claim_ready(spool) == "combo_1"


def test_failed_repeats_are_only_requeued_on_request(tmp_path):
    spool = tmp_path / "graph_queue"
    path = graph_queue.write_state(spool, _job("c", 0))
    path.rename(path.with_suffix(".failed"))

    graph_queue.prepare_spool(spool)
    assert not graph_queue.has_state(spool, "c")
    graph_queue.prepare_spool(spool, retry_failed=True)
    assert graph_queue.has_state(spool, "c")


def test_graphs_are_limited_to_the_first_n_repeats(monkeypatch):
    monkeypatch.setattr(plots_module, "NVISION_GRAPH_REPEATS", 2)
    monkeypatch.setattr(plots_module, "NVISION_PLOT_SWEEP_STRATEGIES", False)
    assert plots_module.plots_wanted("Bayesian-SBED", 0)
    assert plots_module.plots_wanted("Bayesian-SBED", 1)
    assert not plots_module.plots_wanted("Bayesian-SBED", 2)
    assert not plots_module.plots_wanted("SimpleSweep", 0)  # sweep figures are off by default

    monkeypatch.setattr(plots_module, "NVISION_GRAPH_REPEATS", 0)  # 0 = every repeat
    assert plots_module.plots_wanted("Bayesian-SBED", 50)


def _saved_repeat(cache_dir: Path, task, idx: int) -> dict:
    """Read one repeat back from the cache the way the results server does."""
    from nvision.runner.executor import _TaskRunner

    runner = _TaskRunner(task)
    try:
        kw = runner._combo_kw()
        key = stable_config_hash(combination_base_cache_config(**kw, repeat_offset=0))
        repo = runner.cache
        entries, _row = repo._repeats.load_repeat(key, idx)
        return {e["type"]: e for e in entries}
    finally:
        runner._saver_pool.shutdown(wait=True)
        runner.bridge.close()


@pytest.mark.slow
def test_deferred_graphs_round_trip(tmp_path, monkeypatch):
    """Results are saved without graphs, a worker adds them, and only then is the combination archived."""
    monkeypatch.setattr(plots_module, "NVISION_GRAPH_REPEATS", 2)
    for sub in ("scans", "bayes", "cache"):
        (tmp_path / sub).mkdir()
    gen = "NVCenter-voigt-w0.50MHz-c0.10-si0.60MHz-hfn14"

    pq: queue.Queue = queue.Queue()
    monitor = ProgressMonitor(Console(), pq, log_incoming=None, live_mode=False)
    tasks, _ = build_task_list(
        TaskListBuildConfig(
            repeats=3,
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
            no_cache=False,
            ignore_cache_strategy=None,
            require_cache=False,
            filter_category=None,
            filter_strategy=None,
            filter_generator=None,
            filter_noise=None,
            filter_signal=None,
            dry_run=False,
            combination_names=[(gen, "Gauss(0.002)", "Bayesian-SBED")],
            defer_graphs=True,
        ),
        monitor=monitor,
    )
    (task,) = tasks
    assert task.defer_graphs

    run_task(task)

    spool = graph_queue.spool_dir(tmp_path / "cache")
    # Repeats 0 and 1 are queued; repeat 2 is beyond the first-N limit, so it never gets graphs.
    assert [p.name.split("__r")[1][:6] for p in graph_queue.state_paths(spool, task.slug)] == ["000000", "000001"]

    before = _saved_repeat(tmp_path / "cache", task, 0)
    assert before["scan"]["plots_pending"] is True
    assert "content_bin" not in before["scan"]
    assert "_blob" not in before["scan"]
    assert len(before) == 1  # no Bayesian extras yet
    assert _saved_repeat(tmp_path / "cache", task, 2)["scan"]["plot_skipped"] is True

    # The combination is complete: hand it over and let a worker (here in-process) drain the queue.
    graph_queue.mark_ready(spool, task.slug)
    graph_queue.mark_producers_done(spool)
    assert graph_queue.run_worker_loop(spool, tmp_path / "cache") == 1

    assert not graph_queue.has_state(spool, task.slug)
    assert graph_queue.pending_counts(spool) == (0, 0)
    for idx in (0, 1):
        after = _saved_repeat(tmp_path / "cache", task, idx)
        assert "plots_pending" not in after["scan"]
        assert after["scan"].get("_blob") or after["scan"].get("content_bin")
        assert "bayesian_posterior_data" in after
    # Repeat 2 was never queued, so it is untouched.
    assert _saved_repeat(tmp_path / "cache", task, 2)["scan"]["plot_skipped"] is True
