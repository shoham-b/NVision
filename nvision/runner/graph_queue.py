"""Deferred graph generation: a spool of per-repeat plot inputs and the worker that drains it.

Building a repeat's graphs (scan figure, posterior / Fisher / convergence payloads) is not
needed to *have* the result, so runners save the result first and leave the graph work to
separate graph-worker processes:

1. A runner finishes a repeat, saves its slim entries (metrics, per-step series; flagged
   ``plots_pending``) and writes a **state file** with everything the plots need
   (:class:`GraphJob`).
2. When *all* repeats of a combination are saved -- the same moment it becomes archivable --
   the parent marks the combination **ready** (:func:`mark_ready`).
3. A graph worker claims a ready combination (:func:`claim_ready`, an atomic rename, so any
   number of workers can run side by side and never share a combination), generates the graphs
   of all its repeats, re-saves each repeat with them, and only then archives the combination.

Spool layout (``<out_dir>/graph_queue/``)::

    <slug>__r000012.state     plot inputs for repeat 12 of the combination (pickle + zlib)
    <slug>.ready              the combination is complete and waiting for a worker
    <slug>.claimed            a worker is processing it
    <slug>__r000012.failed    a repeat whose graphs failed (kept for inspection / retry)
    producers_done            no more combinations will be marked ready (workers may exit)

Everything is plain files so the queue survives a crash or Ctrl-C: ``nv graphs`` drains
whatever is left. ``slug`` is the combination's task slug (unique per combination).
"""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SPOOL_DIRNAME = "graph_queue"
_STATE = ".state"
_READY = ".ready"
_CLAIMED = ".claimed"
_FAILED = ".failed"
_PRODUCERS_DONE = "producers_done"
_POLL_S = 0.5
_SAVE_ATTEMPTS = 5


@dataclass
class GraphJob:
    """Everything needed to build one repeat's graphs in another process.

    ``combo_kw`` are the keyword arguments of ``LocatorResultsRepository.save_repeat`` that
    identify the combination (generator, noise, strategy, seed, max_steps, timeout_s);
    ``rid`` is the global repeat index, which is also the attempt index in the combination.
    """

    slug: str
    rid: int
    combo_kw: dict[str, Any]
    target_repeats: int
    cache_dir: Path
    shard_index: str | None
    out_dir: Path
    scans_dir: Path
    bayes_dir: Path
    entry_base: dict[str, Any]
    main_result_row: dict[str, Any]
    current_scan: Any  # CoreExperiment
    history_df: Any  # polars.DataFrame
    noise_obj: Any
    strat_obj: Any
    run_result: Any  # RunResult
    sobol_baseline: dict[str, Any] | None  # the SBED task's own baselines, so the worker need not re-simulate
    simplesweep_baseline: dict[str, Any] | None


def spool_dir(cache_dir: Path) -> Path:
    """Spool directory for a run whose cache lives in ``cache_dir`` (a sibling of it)."""
    return Path(cache_dir).parent / SPOOL_DIRNAME


def _names(spool: Path, suffix: str, prefix: str = "") -> list[Path]:
    if not spool.is_dir():
        return []
    return sorted(p for p in spool.iterdir() if p.name.endswith(suffix) and p.name.startswith(prefix))


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def write_state(spool: Path, job: GraphJob) -> Path:
    """Persist one repeat's plot inputs (atomic; re-running a repeat overwrites its file)."""
    spool.mkdir(parents=True, exist_ok=True)
    path = spool / f"{job.slug}__r{job.rid:06d}{_STATE}"
    _atomic_write(path, zlib.compress(pickle.dumps(job, protocol=5), 1))
    return path


def state_paths(spool: Path, slug: str) -> list[Path]:
    return _names(spool, _STATE, prefix=f"{slug}__r")


def has_state(spool: Path, slug: str) -> bool:
    return bool(state_paths(spool, slug))


def slugs_with_state(spool: Path) -> set[str]:
    return {p.name.rsplit("__r", 1)[0] for p in _names(spool, _STATE)}


def mark_ready(spool: Path, slug: str) -> None:
    """Declare a combination complete: a graph worker may now build all its graphs."""
    spool.mkdir(parents=True, exist_ok=True)
    _atomic_write(spool / f"{slug}{_READY}", b"")


def claim_ready(spool: Path) -> str | None:
    """Atomically take one ready combination; returns its slug, or ``None`` if none is ready."""
    for path in _names(spool, _READY):
        slug = path.name[: -len(_READY)]
        try:
            os.replace(path, spool / f"{slug}{_CLAIMED}")
        except (FileNotFoundError, PermissionError):
            continue  # another worker claimed it first
        return slug
    return None


def pending_counts(spool: Path) -> tuple[int, int]:
    """(combinations ready or in progress, repeats still waiting for graphs)."""
    return len(_names(spool, _READY)) + len(_names(spool, _CLAIMED)), len(_names(spool, _STATE))


def mark_producers_done(spool: Path) -> None:
    spool.mkdir(parents=True, exist_ok=True)
    _atomic_write(spool / _PRODUCERS_DONE, b"")


def producers_done(spool: Path) -> bool:
    return (spool / _PRODUCERS_DONE).exists()


def prepare_spool(spool: Path, *, retry_failed: bool = False) -> None:
    """Reset control files before workers start; call only from the single launcher.

    Drops a stale ``producers_done``, and returns combinations a killed worker had claimed to
    the ready queue. ``retry_failed`` also puts previously failed repeats back in the queue.
    """
    spool.mkdir(parents=True, exist_ok=True)
    (spool / _PRODUCERS_DONE).unlink(missing_ok=True)
    for path in _names(spool, _CLAIMED):
        os.replace(path, spool / f"{path.name[: -len(_CLAIMED)]}{_READY}")
    if retry_failed:
        for path in _names(spool, _FAILED):
            os.replace(path, path.with_suffix(_STATE))


def _load_job(path: Path) -> GraphJob:
    return pickle.loads(zlib.decompress(path.read_bytes()))


def _generate_and_save(job: GraphJob, bridge: Any) -> None:
    """Build one repeat's full graph entries and re-save the repeat with them."""
    from nvision.runner.cache import embed_graph_content
    from nvision.runner.plots import generate_attempt_plots
    from nvision.runner.sweep_cache import put_cached_simplesweep_baseline, put_cached_sobol_baseline
    from nvision.sim.combinations import CombinationGrid
    from nvision.viz import Viz

    kw = job.combo_kw
    # This process has no SBED task's in-memory baselines; hand it the ones that task computed,
    # otherwise the plots would re-simulate them.
    if job.sobol_baseline is not None:
        put_cached_sobol_baseline(
            job.current_scan, kw["seed"], kw["generator"], kw["noise"], job.rid, job.sobol_baseline
        )
    if job.simplesweep_baseline is not None:
        put_cached_simplesweep_baseline(
            job.current_scan, kw["seed"], kw["generator"], kw["noise"], job.rid, job.simplesweep_baseline
        )

    entries = generate_attempt_plots(
        viz=Viz(job.out_dir / "graphs"),
        entry_base=job.entry_base,
        attempt_idx_in_combo=job.rid,
        current_scan=job.current_scan,
        current_history_df=job.history_df,
        noise_obj=job.noise_obj,
        strat_obj=job.strat_obj,
        slug_base=job.slug,
        out_dir=job.out_dir,
        scans_dir=job.scans_dir,
        bayes_dir=job.bayes_dir,
        run_result=job.run_result,
        defer=False,
    )
    embedded = embed_graph_content(entries, job.out_dir)
    repo = bridge.get_cache_for_category(CombinationGrid.generator_category(kw["generator"]))
    for attempt in range(_SAVE_ATTEMPTS):
        try:
            repo.save_repeat(
                **kw,
                repeat_offset=0,
                repeat_idx=job.rid,
                entries=embedded,
                main_result_row=job.main_result_row,
            )
            return
        except Exception:
            if attempt == _SAVE_ATTEMPTS - 1:
                raise
            time.sleep(0.1 * (2**attempt))


def _archive(job: GraphJob, bridge: Any) -> None:
    """Move the (now fully graphed) combination to its Parquet archive; a no-op until complete."""
    from nvision.sim.combinations import CombinationGrid

    try:
        repo = bridge.get_cache_for_category(CombinationGrid.generator_category(job.combo_kw["generator"]))
        repo.archive_if_complete(**job.combo_kw, target_repeats=job.target_repeats, log=log)
    except Exception:
        log.debug("Post-graph archive check failed for %s (non-fatal)", job.slug, exc_info=True)


def process_combo(spool: Path, slug: str, bridge: Any) -> tuple[int, int]:
    """Generate and save the graphs of every queued repeat of ``slug``; archive if all succeeded.

    Returns ``(n_ok, n_failed)``. A failed repeat keeps its state file, renamed ``.failed``.
    """
    n_ok = n_failed = 0
    last: GraphJob | None = None
    for path in state_paths(spool, slug):
        try:
            job = _load_job(path)
            _generate_and_save(job, bridge)
        except Exception:
            log.error("Graph generation failed for %s (kept as %s)", path.name, path.with_suffix(_FAILED).name)
            log.debug("Graph generation failure detail for %s", path.name, exc_info=True)
            with contextlib.suppress(OSError):
                os.replace(path, path.with_suffix(_FAILED))
            n_failed += 1
            continue
        path.unlink(missing_ok=True)
        n_ok += 1
        last = job
    if n_failed == 0 and last is not None:
        _archive(last, bridge)
    return n_ok, n_failed


def run_worker_loop(spool: Path, cache_dir: Path, shard_index: str | None = None) -> int:
    """Claim and process ready combinations until the producers are done and none are left.

    Returns the number of combinations this worker processed.
    """
    from nvision.cache import CacheBridge

    bridge = CacheBridge(cache_dir, shard_suffix=shard_index)
    processed = 0
    try:
        while True:
            # Read the sentinel BEFORE claiming: it is written after the last ready marker, so if
            # it was already set and nothing is claimable, no combination can still appear.
            finished = producers_done(spool)
            slug = claim_ready(spool)
            if slug is None:
                if finished:
                    return processed
                time.sleep(_POLL_S)
                continue
            try:
                n_ok, n_failed = process_combo(spool, slug, bridge)
                log.info("Graphs for %s: %s repeat(s) done%s", slug, n_ok, f", {n_failed} failed" if n_failed else "")
            finally:
                (spool / f"{slug}{_CLAIMED}").unlink(missing_ok=True)
            processed += 1
    finally:
        bridge.close()
