"""One-off migration: re-encode existing `graphs` BLOB rows from the old
float32-only _f32_json format to the new adaptive float16/float32 format.

Safe to interrupt and re-run: each row is migrated independently inside its
own transaction, and a per-shard ``graphs_migrated_f16`` checkpoint table
tracks which keys are already done, so a partial run just picks up where it
left off without re-decoding rows it already handled.

Usage:
    uv run python scripts/migrate_f16_cache.py <path-to-shard.db> [--dry-run]
    uv run python scripts/migrate_f16_cache.py <path-to-cache-dir> [--dry-run]   # migrates every *_shard*.db in the dir
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nvision.viz._f32_json import _decode_arrays, payload_to_gz_bytes  # noqa: E402


def _migrate_blob(raw_gz: bytes) -> tuple[bytes, int, int]:
    """Decode an old (or already-new) blob fully, re-encode with the current
    adaptive scheme. Returns (new_bytes, old_len, new_len)."""
    decoded = _decode_arrays(json.loads(gzip.decompress(raw_gz)))
    new_bytes = payload_to_gz_bytes(decoded)
    return new_bytes, len(raw_gz), len(new_bytes)


def migrate_shard(db_path: Path, dry_run: bool = False, batch_size: int = 200) -> None:
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")

    has_graphs = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='graphs'"
    ).fetchone()
    if not has_graphs:
        print(f"[{db_path.name}] no 'graphs' table (pre-dates blob extraction) -- nothing to do")
        conn.close()
        return

    conn.execute(
        "CREATE TABLE IF NOT EXISTS graphs_migrated_f16 (key TEXT PRIMARY KEY)"
    )
    conn.commit()

    cur = conn.execute("SELECT COUNT(*) FROM graphs")
    total = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM graphs_migrated_f16")
    already_done = cur.fetchone()[0]
    print(f"[{db_path.name}] {total} rows total, {already_done} already migrated")

    done_this_run = 0
    orig_bytes_total = 0
    new_bytes_total = 0
    t0 = time.time()

    while True:
        rows = conn.execute(
            """
            SELECT g.key, g.data FROM graphs g
            LEFT JOIN graphs_migrated_f16 m ON g.key = m.key
            WHERE m.key IS NULL
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
        if not rows:
            break

        updates = []
        marks = []
        for key, data in rows:
            try:
                new_bytes, old_len, new_len = _migrate_blob(bytes(data))
            except Exception as exc:  # corrupt/unexpected row -- skip, don't crash the whole run
                print(f"  SKIP {key}: {exc!r}")
                marks.append((key,))
                continue
            orig_bytes_total += old_len
            new_bytes_total += new_len
            if not dry_run:
                updates.append((new_bytes, key))
            marks.append((key,))

        if not dry_run:
            with conn:
                conn.executemany("UPDATE graphs SET data = ? WHERE key = ?", updates)
                conn.executemany("INSERT OR IGNORE INTO graphs_migrated_f16 (key) VALUES (?)", marks)
        done_this_run += len(rows)

        elapsed = time.time() - t0
        rate = done_this_run / elapsed if elapsed > 0 else 0
        pct = 100 * (already_done + done_this_run) / total if total else 100
        saved_pct = 100 * (1 - new_bytes_total / orig_bytes_total) if orig_bytes_total else 0
        print(
            f"  {already_done + done_this_run}/{total} ({pct:.1f}%)  "
            f"{rate:.0f} rows/s  running_saved={saved_pct:.1f}%"
        )

    print(
        f"[{db_path.name}] done. {done_this_run} rows this run, "
        f"{orig_bytes_total/1e6:.1f}MB -> {new_bytes_total/1e6:.1f}MB "
        f"({100*(1-new_bytes_total/orig_bytes_total) if orig_bytes_total else 0:.1f}% saved this run)"
    )
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="shard .db file, or a directory containing *_shard*.db files")
    ap.add_argument("--dry-run", action="store_true", help="measure savings without writing anything")
    args = ap.parse_args()

    if args.path.is_dir():
        shards = sorted(args.path.glob("*_shard*.db"))
        if not shards:
            print(f"No *_shard*.db files found in {args.path}")
            return
        for shard in shards:
            migrate_shard(shard, dry_run=args.dry_run)
    else:
        migrate_shard(args.path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
