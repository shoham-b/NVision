"""Remembered lab data sources (Drive links / local folders) and their incremental sync.

``nv matlab <source>`` adds a source to ``<dir>/labs/sources.json`` and syncs it; later syncs are
``nv matlab`` with no source, by hand or in a background subprocess (:class:`BackgroundSync`) from the lab
page's Sync button. ``nv serve`` does not sync on start.
The sync is incremental: a per-source :class:`FileIndex` remembers each ``.mat`` file's kind by size
and mtime, and a raw sweep's result record is redone only when its file or :data:`PROCESSING_VERSION` changes.

The sync runs in its own process, not a server thread: processing a sweep rewrites process-global convergence
thresholds for the duration of the run (``_real_data_contrast_total_threshold``), which must never leak into the
requests the server answers meanwhile.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import scipy.io as sio

from nvision.metrics.lab_study import labs_dir
from nvision.tools.drive_fetch import fetch_drive_folder, looks_like_drive_folder, parse_drive_folder_id

log = logging.getLogger(__name__)

# Bump when the processing of a raw sweep changes (locator settings, reference fit, record fields): every sweep's
# record is then redone on the next sync.
PROCESSING_VERSION = 3  # 3: per-step sigma history + fit of the lab's averaged sweep; 2: no z on a bound reference
# A held lock is touched every _LOCK_HEARTBEAT_S; one not touched for _STALE_LOCK_S belongs to a sync that died
# (killed with its console, say) and is taken over.
_LOCK_HEARTBEAT_S = 20.0
_STALE_LOCK_S = 120.0


def _labs(directory: Path) -> Path:
    return labs_dir(directory)


def load_sources(directory: Path) -> list[dict[str, Any]]:
    """The remembered sources, oldest first: ``{"kind": "drive"|"path", "source", "label", "added"}``."""
    path = _labs(directory) / "sources.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def add_source(directory: Path, source: str) -> dict[str, Any]:
    """Remember *source* (a Drive folder link/id or an existing local path) and return its entry.

    Adding the same source again returns its existing entry. The label names the source's folder under
    ``<dir>/labs`` and in run names: the Drive folder id, or the local folder's (file's) name, suffixed when two
    different sources would share one.
    """
    if looks_like_drive_folder(source):
        kind, value, base_label = "drive", parse_drive_folder_id(source), parse_drive_folder_id(source)
    else:
        p = Path(source)
        if not p.exists():
            raise ValueError(f"{source} is neither a Drive folder link nor an existing path")
        kind, value, base_label = "path", str(p.resolve()), (p.stem if p.is_file() else p.resolve().name)
    sources = load_sources(directory)
    for entry in sources:
        if entry["kind"] == kind and entry["source"] == value:
            return entry
    taken = {e["label"] for e in sources}
    label, n = base_label, 2
    while label in taken:
        label, n = f"{base_label}-{n}", n + 1
    entry = {"kind": kind, "source": value, "label": label, "added": datetime.now(UTC).isoformat()}
    _write_json(_labs(directory) / "sources.json", [*sources, entry])
    return entry


def resolve_source(entry: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    """``(root, base)`` for a remembered source, fetching a Drive folder first; *base* is what paths are relative to."""
    if entry["kind"] == "drive":
        root = fetch_drive_folder(entry["source"], directory / "downloads" / entry["label"])
    else:
        root = Path(entry["source"])
        if not root.exists():
            raise ValueError(f"Source folder {root} no longer exists")
    return root, (root.parent if root.is_file() else root)


def classify_mat(path: Path) -> str:
    """``"bayesian"`` (a lab Bayesian ESR run), ``"sweep"`` (a raw ESR recording), ``"other"`` or ``"unreadable"``."""
    try:
        top = sio.loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["myStruct"]).get("myStruct")
    except Exception:  # any file scipy cannot read is reported as such, never processed
        return "unreadable"
    if top is None:
        return "other"
    esr = getattr(top, "BayesianESR", None)
    if esr is not None and hasattr(esr, "state"):
        return "bayesian"
    if hasattr(top, "ESR"):
        return "sweep"
    return "other"


class FileIndex:
    """Each source file's kind, remembered by size and mtime so unchanged files are not re-read."""

    def __init__(self, label_dir: Path) -> None:
        self._path = label_dir / "file_index.json"
        self._previous: dict[str, dict[str, Any]] = json.loads(self._path.read_text()) if self._path.exists() else {}
        self._current: dict[str, dict[str, Any]] = {}
        self._changed: set[str] = set()

    def classify(self, files: list[Path], base: Path) -> dict[Path, str]:
        kinds: dict[Path, str] = {}
        for f in files:
            rel = f.relative_to(base).as_posix()
            stat = f.stat()
            prev = self._previous.get(rel)
            if prev is not None and prev["size"] == stat.st_size and prev["mtime"] == stat.st_mtime:
                kind = prev["kind"]
            else:
                kind = classify_mat(f)
                self._changed.add(rel)
            self._current[rel] = {"size": stat.st_size, "mtime": stat.st_mtime, "kind": kind}
            kinds[f] = kind
        return kinds

    def folder_changed(self, folder_files: list[Path], base: Path) -> bool:
        """True when one of *folder_files* is new or changed, or a file of that folder disappeared."""
        rels = {f.relative_to(base).as_posix() for f in folder_files}
        if rels & self._changed:
            return True
        folder = Path(next(iter(rels))).parent.as_posix()
        return any(Path(r).parent.as_posix() == folder and r not in self._current for r in self._previous)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in self._current.values():
            out[item["kind"]] = out.get(item["kind"], 0) + 1
        return out

    def save(self) -> None:
        _write_json(self._path, self._current)


def sweep_record_path(label_dir: Path, matlab_file: Path, base: Path) -> Path:
    """Where a raw sweep's result record lives: ``<labs>/<label>/sweeps/<path relative to the source>.json``."""
    return label_dir / "sweeps" / matlab_file.relative_to(base).with_suffix(".json")


def sweep_record_is_current(record_path: Path, matlab_file: Path) -> bool:
    """True when the record matches the file and the processing version; a failed file is always retried."""
    if not record_path.exists():
        return False
    record = json.loads(record_path.read_text())
    stat = matlab_file.stat()
    return (
        "error" not in record
        and record.get("version") == PROCESSING_VERSION
        and record.get("source_size") == stat.st_size
        and record.get("source_mtime") == stat.st_mtime
    )


def load_sweep_records(directory: Path) -> list[dict[str, Any]]:
    """Every raw sweep's result record under ``<dir>/labs``, each with its source ``label``."""
    records = []
    for entry in load_sources(directory):
        sweeps_dir = _labs(directory) / entry["label"] / "sweeps"
        for path in sorted(sweeps_dir.rglob("*.json")) if sweeps_dir.exists() else []:
            records.append({"label": entry["label"], **json.loads(path.read_text())})
    return records


def lock_is_held(directory: Path) -> bool:
    """True while some process holds the sync lock (it is fresh: its heartbeat is still touching it)."""
    path = _labs(directory) / "sync.lock"
    try:
        return time.time() - path.stat().st_mtime <= _STALE_LOCK_S
    except FileNotFoundError:
        return False


class SyncLock:
    """``<dir>/labs/sync.lock``: one writer at a time across ``nv matlab`` runs and the lab page's sync.

    While held, a daemon thread touches the file every :data:`_LOCK_HEARTBEAT_S`, so a lock left by a process that
    was killed goes stale within :data:`_STALE_LOCK_S` and the next sync takes it over.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._path = _labs(directory) / "sync.lock"
        self.acquired = False
        self._stop = threading.Event()

    def __enter__(self) -> SyncLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists() and not lock_is_held(self._directory):
            self._path.unlink(missing_ok=True)
        try:
            fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return self
        with os.fdopen(fd, "w") as fh:
            fh.write(str(os.getpid()))
        self.acquired = True
        threading.Thread(target=self._heartbeat, daemon=True).start()
        return self

    def _heartbeat(self) -> None:
        while not self._stop.wait(_LOCK_HEARTBEAT_S):
            self._path.touch()

    def __exit__(self, *exc: object) -> None:
        if self.acquired:
            self._stop.set()
            self._path.unlink(missing_ok=True)


def write_sync_status(directory: Path, status: dict[str, Any]) -> None:
    _write_json(_labs(directory) / "sync_status.json", status)


def read_sync_status(directory: Path) -> dict[str, Any] | None:
    path = _labs(directory) / "sync_status.json"
    return json.loads(path.read_text()) if path.exists() else None


class BackgroundSync:
    """Runs ``nv matlab --dir <dir>`` (no SOURCE: re-sync all) as a subprocess and calls *on_done* when it exits."""

    def __init__(self, directory: Path, on_done: Callable[[], None]) -> None:
        self._directory = directory
        self._on_done = on_done
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.log_path = _labs(directory) / "sync.log"

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        """Start a sync unless one is running or no source was added; True when one was started."""
        with self._lock:
            if self.running or not load_sources(self._directory):
                return False
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = self.log_path.open("w", encoding="utf-8")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            cmd = [sys.executable, "-m", "nvision", "matlab", "--dir", str(self._directory)]
            self._proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
            proc = self._proc

        def _wait() -> None:
            proc.wait()
            log_fh.close()
            self._on_done()

        threading.Thread(target=_wait, daemon=True).start()
        return True

    def state(self) -> dict[str, Any]:
        tail: list[str] = []
        if self.log_path.exists():
            tail = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        status = read_sync_status(self._directory)
        # Another process (a foreground `nv matlab`) may hold the lock; a "running" status that no process holds is
        # a sync that was killed before it could write its final state.
        running = self.running or lock_is_held(self._directory)
        if status is not None and status.get("state") == "running" and not running:
            status = {**status, "state": "interrupted"}
        return {
            "sources": [e["label"] for e in load_sources(self._directory)],
            "running": running,
            "status": status,
            "log_tail": tail,
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
