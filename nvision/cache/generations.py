"""Cache "generations": named snapshots of the whole ``<out>/cache`` directory.

Lets a user archive the current cache before starting a differently-configured
run (e.g. after a code change that alters numerics, not just speed), keep a
rolling window of the last N attempts, and roll back to an earlier one --
without having to reason about the sharded SQLite files underneath directly.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

GENERATIONS_DIRNAME = "cache_generations"
_METADATA_FILENAME = "generation.json"
_TIMESTAMP_FMT = "%Y%m%d-%H%M%S-%f"


@dataclass
class GenerationInfo:
    name: str  # directory name on disk: "<timestamp>_<label>"
    label: str
    created_at: str  # ISO 8601 UTC
    commit: str | None
    path: Path
    size_bytes: int


def _git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _checkpoint_wal(cache_dir: Path) -> None:
    """Best-effort: merge each shard's WAL back into its main file before copying.

    Safe to call even while another process still holds the DB open (SQLite
    permits concurrent checkpoints); a locked or invalid file is skipped rather
    than failing the whole snapshot. This does not guarantee a perfectly
    consistent snapshot of a cache a run is actively writing to -- for that,
    stop the run first.
    """
    for db_path in cache_dir.glob("*.db"):
        with suppress(Exception):
            conn = sqlite3.connect(db_path, timeout=5.0)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                conn.commit()
            finally:
                conn.close()


def _sanitize_label(label: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in label)
    return safe.strip("-") or "gen"


def save_generation(out: Path, label: str | None = None) -> Path:
    """Copy ``<out>/cache`` into a new timestamped folder under ``<out>/cache_generations/``.

    Never touches the live cache -- this only ever copies.
    """
    cache_dir = out / "cache"
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"No cache directory at {cache_dir}")

    commit = _git_commit(Path(__file__).resolve().parent)
    resolved_label = _sanitize_label(label) if label else (commit or "snapshot")
    name = f"{datetime.now(UTC).strftime(_TIMESTAMP_FMT)}_{resolved_label}"

    generations_root = out / GENERATIONS_DIRNAME
    generations_root.mkdir(parents=True, exist_ok=True)
    dest = generations_root / name
    if dest.exists():
        raise FileExistsError(f"Generation {name} already exists at {dest}")

    _checkpoint_wal(cache_dir)
    shutil.copytree(cache_dir, dest / "cache")

    metadata = {
        "label": resolved_label,
        "created_at": datetime.now(UTC).isoformat(),
        "commit": commit,
    }
    (dest / _METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dest


def _read_metadata(gen_dir: Path) -> dict:
    meta_path = gen_dir / _METADATA_FILENAME
    if meta_path.exists():
        with suppress(Exception):
            return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            with suppress(OSError):
                total += f.stat().st_size
    return total


def list_generations(out: Path) -> list[GenerationInfo]:
    """All saved generations, most recently created first."""
    root = out / GENERATIONS_DIRNAME
    if not root.is_dir():
        return []
    infos = []
    for gen_dir in sorted(root.iterdir()):
        if not gen_dir.is_dir():
            continue
        meta = _read_metadata(gen_dir)
        infos.append(
            GenerationInfo(
                name=gen_dir.name,
                label=meta.get("label", gen_dir.name),
                created_at=meta.get("created_at", "-"),
                commit=meta.get("commit"),
                path=gen_dir,
                size_bytes=_dir_size(gen_dir),
            )
        )
    infos.sort(key=lambda g: g.created_at, reverse=True)
    return infos


def _resolve_generation(out: Path, name_or_label: str) -> GenerationInfo:
    infos = list_generations(out)
    for info in infos:
        if info.name == name_or_label:
            return info
    matches = [info for info in infos if info.label == name_or_label]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Label {name_or_label!r} matches multiple generations ({[m.name for m in matches]}); "
            "use the full generation name instead."
        )
    raise FileNotFoundError(f"No generation named or labeled {name_or_label!r} found in {out / GENERATIONS_DIRNAME}")


def restore_generation(out: Path, name_or_label: str, *, archive_current: bool = True) -> Path:
    """Replace ``<out>/cache`` with a copy of a saved generation.

    The current cache is archived first by default (labeled "pre-restore") so
    restoring never silently discards whatever was live -- pass
    ``archive_current=False`` to skip that.
    """
    info = _resolve_generation(out, name_or_label)
    cache_dir = out / "cache"

    if archive_current and cache_dir.is_dir() and any(cache_dir.iterdir()):
        save_generation(out, label="pre-restore")

    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)
    shutil.copytree(info.path / "cache", cache_dir)
    return cache_dir


def prune_generations(out: Path, keep: int, *, dry_run: bool = False) -> list[str]:
    """Delete all but the ``keep`` most recent generations. Returns the removed names."""
    infos = list_generations(out)
    to_remove = infos[keep:] if keep >= 0 else []
    removed = []
    for info in to_remove:
        removed.append(info.name)
        if not dry_run:
            shutil.rmtree(info.path)
    return removed


def delete_generation(out: Path, name_or_label: str) -> str:
    """Delete one specific generation by name or label. Returns the deleted name."""
    info = _resolve_generation(out, name_or_label)
    shutil.rmtree(info.path)
    return info.name
