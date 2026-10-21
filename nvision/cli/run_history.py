"""Run history kept in the cache DB, so ``--resume`` can pick any earlier run (like ``git reflog``).

One document under ``RUN_HISTORY_KEY`` in ``nv_center.db`` holds the most recent runs. Each run records the
combinations it started and the *chain* it can resume: everything it inherited from the run it resumed plus
what it started itself. Resuming run X reuses the cached results of ``X["chain"]`` only. Deleting the DB
clears the history together with the results it refers to.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nvision.cache.data_store import CategoryDataStore

RUN_HISTORY_KEY = "run_history"
MAX_RUNS = 50

Combo = tuple[str, str, str]
RunEntry = dict[str, Any]

_RUNNING_TASK_LOG_RE = re.compile(r"Running task:\s*(\S+)")


def parse_started_combos_from_log(log_path: Path) -> set[Combo]:
    """Extract (generator, noise, strategy) triples that executed in the session log."""
    found: set[Combo] = set()
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _RUNNING_TASK_LOG_RE.search(line)
            if m:
                parts = m.group(1).split("/", 2)
                if len(parts) == 3:
                    found.add((parts[0], parts[1], parts[2]))
    return found


def load_runs(store: CategoryDataStore) -> list[RunEntry]:
    """Runs, oldest first."""
    payload = store.backend.get(RUN_HISTORY_KEY)
    if not isinstance(payload, dict):
        return []
    return list(payload["runs"])


def save_runs(store: CategoryDataStore, runs: list[RunEntry]) -> None:
    store.backend.set(RUN_HISTORY_KEY, {"kind": "run_history", "runs": runs[-MAX_RUNS:]})


def finalize_runs(runs: list[RunEntry], logs_root: Path, *, min_idle_s: float = 0.0) -> bool:
    """Fill in ``started``/``chain`` of runs that ended (or were killed) since they were recorded.

    A run's log is the only record of which combinations it actually started, so this reads it once and
    stores the result. Returns True when any entry changed. A run whose log is already gone keeps an empty
    ``started`` (its inherited chain is still resumable). Runs whose log changed less than ``min_idle_s``
    seconds ago are left alone, since they may still be running.
    """
    changed = False
    for run in runs:
        if run["started"] is not None:
            continue
        log_path = logs_root / run["log"]
        if min_idle_s and log_path.exists() and time.time() - log_path.stat().st_mtime < min_idle_s:
            continue
        started = parse_started_combos_from_log(log_path) if log_path.exists() else set()
        run["started"] = sorted(started)
        run["chain"] = sorted({tuple(c) for c in run["inherited"]} | started)
        changed = True
    return changed


def record_run(
    runs: list[RunEntry],
    *,
    run_id: str,
    group: str | None,
    repeats: int,
    log_name: str,
    resumed_from: str | None,
    inherited: set[Combo],
) -> None:
    runs.append(
        {
            "id": run_id,
            "started_at": datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "group": group,
            "repeats": repeats,
            "log": log_name,
            "resumed_from": resumed_from,
            "inherited": sorted(inherited),
            "started": None,
            "chain": None,
        }
    )


def find_run(runs: list[RunEntry], ref: str) -> RunEntry:
    """Look up a run by id or unambiguous id prefix; ``ref`` may also be ``latest`` or ``~N`` (N-th last)."""
    if not runs:
        raise ValueError("no runs recorded in the cache DB yet")
    if ref == "latest":
        return runs[-1]
    if ref.startswith("~") and ref[1:].isdigit():
        n = int(ref[1:])
        if n >= len(runs):
            raise ValueError(f"only {len(runs)} run(s) recorded, cannot pick {ref}")
        return runs[-1 - n]
    matches = [r for r in runs if r["id"].startswith(ref)]
    if len(matches) != 1:
        raise ValueError(f"run reference {ref!r} matches {len(matches)} runs (see `nv runs`)")
    return matches[0]
