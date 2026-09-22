"""Cold storage for finished combinations: one Parquet file per combo_key.

A combination's cache footprint is a pointer row + N repeat rows + N ``:meta``
sidecars + up to N*M blob rows (`nvision/cache/repeats_repository.py`). Once a
combination has reached its requested repeat count it stops being written to,
so its SQLite rows are pure dead weight sitting in a live, randomly-accessed
store that's optimized for point writes -- while remaining fully needed for
reading (the results UI still renders it). :class:`ComboArchive` moves that
data into a small Parquet file per combo (immutable, columnar, ~4-5x smaller
than the equivalent SQLite storage for the JSON rows; the already-gzipped
blob rows don't shrink further but at least stop occupying a mutable store).

Reads are made transparent via :class:`_ArchiveFallbackBackend`
(`nvision/cache/data_store.py`): every key format the live backend uses
(`nvision/cache/sqlite.py`) is byte-for-byte reproduced here, so nothing above
``CategoryDataStore`` -- ``RepeatsRepository``, ``LocatorResultsRepository``,
``CacheBridge``, ``api_server.py`` -- needs to know a combo has been archived.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import polars as pl

# Bound on how many combos' DataFrames _load_combo_df keeps resident at once.
# Repeat/blob rehydration (nvision/cache/repeats_repository.py) does one
# store.load_blob() call per plot entry -- without this, viewing one archived
# combo with N repeats x M plot types would re-read and re-decompress the same
# small Parquet file N*M times. Bounded (not unbounded) since a long-lived
# `nv serve` process can touch many distinct combos over its lifetime.
_DF_CACHE_MAX_COMBOS = 64


def combo_key_for(key: str) -> str:
    """Extract the owning combo_key from any live-backend key string.

    Pointer keys *are* the bare combo_key (no prefix). Repeat/meta/blob keys
    are ``repeat:{combo_key}:{idx}[:meta]`` / ``blob:{combo_key}:{idx}:{type}``
    -- combo_key is always the MD5 hex digest in field 1, never containing ':'.
    """
    if key.startswith("repeat:") or key.startswith("blob:"):
        return key.split(":")[1]
    return key


class ComboArchive:
    """Parquet-backed cold store for one category (nv_center / complementary).

    One file per combo: ``<archive_dir>/<combo_key>.parquet``, columns
    ``key`` (Utf8), ``kind`` ('json' | 'blob'), ``payload`` (Binary -- UTF-8
    JSON bytes for 'json' rows, raw bytes for 'blob' rows, matching exactly
    what :class:`~nvision.cache.sqlite.ShardedSqliteCache`'s ``cache``/
    ``graphs`` tables would hand back from ``get``/``blob_get``).
    """

    def __init__(self, archive_dir: Path) -> None:
        self.archive_dir = archive_dir
        self._df_cache: OrderedDict[str, pl.DataFrame] = OrderedDict()
        # key -> (kind, payload), built once per combo from its DataFrame. Single-key
        # lookups (get/blob_get/contains) go through this instead of re-filtering the
        # DataFrame linearly on every call -- without it, an admin loop that visits
        # every key of an archived combo (nv cache list/progress/clean) does O(rows)
        # work per key, i.e. O(rows^2) per combo.
        self._index_cache: OrderedDict[str, dict[str, tuple[str, bytes]]] = OrderedDict()

    def _path(self, combo_key: str) -> Path:
        return self.archive_dir / f"{combo_key}.parquet"

    def has_combo(self, combo_key: str) -> bool:
        return self._path(combo_key).exists()

    def _invalidate(self, combo_key: str) -> None:
        self._df_cache.pop(combo_key, None)
        self._index_cache.pop(combo_key, None)

    def write_combo(self, combo_key: str, rows: list[tuple[str, str, bytes]]) -> Path:
        """Write (overwrite) one combo's full row set. Atomic via temp file + rename."""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(
            {
                "key": [r[0] for r in rows],
                "kind": [r[1] for r in rows],
                "payload": [r[2] for r in rows],
            },
            schema={"key": pl.String, "kind": pl.String, "payload": pl.Binary},
        )
        path = self._path(combo_key)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.archive_dir), prefix=f".{combo_key}.", suffix=".parquet.tmp")
        os.close(fd)
        try:
            df.write_parquet(tmp_name, compression="zstd")
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        self._invalidate(combo_key)
        return path

    def _load_combo_df(self, combo_key: str) -> pl.DataFrame | None:
        """Read one combo's Parquet file, cached in-process (LRU, bounded).

        A single combo view/render calls this dozens of times (one per repeat,
        one per blob) -- caching means the file is actually read from disk once,
        not once per key.
        """
        cached = self._df_cache.get(combo_key)
        if cached is not None:
            self._df_cache.move_to_end(combo_key)
            return cached

        path = self._path(combo_key)
        if not path.exists():
            return None
        try:
            df = pl.read_parquet(path)
        except Exception:
            return None

        self._df_cache[combo_key] = df
        self._df_cache.move_to_end(combo_key)
        if len(self._df_cache) > _DF_CACHE_MAX_COMBOS:
            self._df_cache.popitem(last=False)
        return df

    def _load_combo_index(self, combo_key: str) -> dict[str, tuple[str, bytes]] | None:
        cached = self._index_cache.get(combo_key)
        if cached is not None:
            self._index_cache.move_to_end(combo_key)
            return cached

        df = self._load_combo_df(combo_key)
        if df is None:
            return None
        index = {row["key"]: (row["kind"], row["payload"]) for row in df.iter_rows(named=True)}
        self._index_cache[combo_key] = index
        self._index_cache.move_to_end(combo_key)
        if len(self._index_cache) > _DF_CACHE_MAX_COMBOS:
            self._index_cache.popitem(last=False)
        return index

    def _row_payload(self, combo_key: str, key: str) -> tuple[str, bytes] | None:
        index = self._load_combo_index(combo_key)
        if index is None:
            return None
        return index.get(key)

    def get(self, key: str) -> dict | None:
        found = self._row_payload(combo_key_for(key), key)
        if found is None or found[0] != "json":
            return None
        try:
            return json.loads(found[1].decode("utf-8"))
        except Exception:
            return None

    def blob_get(self, key: str) -> bytes | None:
        found = self._row_payload(combo_key_for(key), key)
        if found is None or found[0] != "blob":
            return None
        return bytes(found[1])

    def _group_by_combo(self, keys: list[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for k in keys:
            grouped.setdefault(combo_key_for(k), []).append(k)
        return grouped

    def batch_get(self, keys: list[str]) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for combo_key, combo_keys in self._group_by_combo(keys).items():
            df = self._load_combo_df(combo_key)
            if df is None:
                continue
            match = df.filter(pl.col("key").is_in(combo_keys) & (pl.col("kind") == "json"))
            for row in match.iter_rows(named=True):
                with contextlib.suppress(Exception):
                    result[row["key"]] = json.loads(row["payload"].decode("utf-8"))
        return result

    def blob_batch_get(self, keys: list[str]) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        for combo_key, combo_keys in self._group_by_combo(keys).items():
            df = self._load_combo_df(combo_key)
            if df is None:
                continue
            match = df.filter(pl.col("key").is_in(combo_keys) & (pl.col("kind") == "blob"))
            for row in match.iter_rows(named=True):
                result[row["key"]] = bytes(row["payload"])
        return result

    def keys_exist_batch(self, keys: list[str]) -> set[str]:
        found: set[str] = set()
        for combo_key, combo_keys in self._group_by_combo(keys).items():
            df = self._load_combo_df(combo_key)
            if df is None:
                continue
            match = df.filter(pl.col("key").is_in(combo_keys))
            found.update(match.get_column("key").to_list())
        return found

    def contains(self, key: str) -> bool:
        return self._row_payload(combo_key_for(key), key) is not None

    def _all_files(self) -> list[Path]:
        if not self.archive_dir.exists():
            return []
        return sorted(self.archive_dir.glob("*.parquet"))

    def iter_keys(self):
        for path in self._all_files():
            try:
                df = pl.read_parquet(path, columns=["key"])
            except Exception:
                continue
            yield from df.get_column("key").to_list()

    def list_keys_excluding_prefixes(self, prefixes: list[str]) -> list[str]:
        files = self._all_files()
        if not files:
            return []
        try:
            lf = pl.scan_parquet([str(p) for p in files]).select("key")
            for prefix in prefixes:
                lf = lf.filter(~pl.col("key").str.starts_with(prefix))
            return lf.collect().get_column("key").to_list()
        except Exception:
            return [k for k in self.iter_keys() if not any(k.startswith(p) for p in prefixes)]

    def delete_combo(self, combo_key: str) -> None:
        """Remove one combo's archive file entirely (e.g. as part of a cache purge)."""
        path = self._path(combo_key)
        if path.exists():
            path.unlink()
        self._invalidate(combo_key)


class _ArchiveFallbackBackend:
    """Wraps a live backend + :class:`ComboArchive`; reads fall through on a live miss.

    Writes/deletes always target the live backend only -- the archive is
    written exclusively via :func:`archive_combination`. Matches the public
    interface of :class:`~nvision.cache.sqlite.ShardedSqliteCache` /
    :class:`~nvision.cache.mysql.MySqlCache` so every existing caller (the
    ``.backend`` property, ``CategoryDataStore``'s own methods) keeps working
    unmodified.
    """

    def __init__(self, live: Any, archive: ComboArchive) -> None:
        self._live = live
        self.archive = archive

    def close(self) -> None:
        self._live.close()

    def get(self, key: str) -> dict | None:
        v = self._live.get(key)
        if v is not None:
            return v
        return self.archive.get(key)

    def set(self, key: str, value: dict) -> None:
        self._live.set(key, value)

    def delete(self, key: str) -> None:
        self._live.delete(key)

    def delete_many(self, keys: list[str]) -> None:
        """Delete live rows, and drop the whole archive file for any combo touched.

        Every existing caller of ``delete_many`` (``nv cache clean``, the purge
        helpers in ``nvision/tools/artifacts.py``) already deletes a combo's
        pointer plus every speculative repeat/meta/blob key in one call -- i.e.
        it's always a full-combo wipe, never a partial one. So dropping the
        combo's entire archive file alongside its live rows is correct here,
        not just convenient.
        """
        self._live.delete_many(keys)
        for combo_key in {combo_key_for(k) for k in keys}:
            self.archive.delete_combo(combo_key)

    def batch_get(self, keys: list[str]) -> dict[str, dict]:
        result = self._live.batch_get(keys)
        missing = [k for k in keys if k not in result]
        if missing:
            result.update(self.archive.batch_get(missing))
        return result

    def batch_set(self, items: dict[str, dict]) -> None:
        self._live.batch_set(items)

    def blob_get(self, key: str) -> bytes | None:
        v = self._live.blob_get(key)
        if v is not None:
            return v
        return self.archive.blob_get(key)

    def blob_set(self, key: str, data: bytes) -> None:
        self._live.blob_set(key, data)

    def blob_batch_get(self, keys: list[str]) -> dict[str, bytes]:
        result = self._live.blob_batch_get(keys)
        missing = [k for k in keys if k not in result]
        if missing:
            result.update(self.archive.blob_batch_get(missing))
        return result

    def keys_exist_batch(self, keys: list[str]) -> set[str]:
        found = self._live.keys_exist_batch(keys)
        missing = [k for k in keys if k not in found]
        if missing:
            found = found | self.archive.keys_exist_batch(missing)
        return found

    def __contains__(self, key: str) -> bool:
        if key in self._live:
            return True
        return self.archive.contains(key)

    def __iter__(self):
        seen: set[str] = set()
        for k in self._live:
            seen.add(k)
            yield k
        for k in self.archive.iter_keys():
            if k not in seen:
                yield k

    def list_keys_excluding_prefixes(self, prefixes: list[str]) -> list[str]:
        live_keys = self._live.list_keys_excluding_prefixes(prefixes)
        archived_keys = self.archive.list_keys_excluding_prefixes(prefixes)
        seen = set(live_keys)
        return live_keys + [k for k in archived_keys if k not in seen]


def archive_combination(
    store: Any,
    combo_key: str,
    achieved_repeats: int,
    log: logging.Logger,
) -> bool:
    """Move one finished combination's pointer+repeats+blobs to its Parquet archive file.

    Reads go through ``store.backend``, which is already archive-fallback-aware
    (see :class:`_ArchiveFallbackBackend`), so calling this again later after more
    repeats were appended past a prior archive pass re-archives the merged set --
    safe and idempotent. Only deletes live rows after the Parquet write round-trips.
    Returns True if the combo was archived (or already fully archived with nothing
    left live), False if it was skipped (e.g. pointer missing or repeats incomplete).
    """
    from nvision.cache.repeats_repository import RepeatsRepository

    backend = store.backend
    archive: ComboArchive = backend.archive

    ptr_payload = backend.get(combo_key)
    if ptr_payload is None:
        log.warning("archive_combination: no pointer found for %s, skipping", combo_key)
        return False

    repeat_keys = [RepeatsRepository.make_repeat_key(combo_key, i) for i in range(achieved_repeats)]
    meta_keys = [k + ":meta" for k in repeat_keys]
    repeat_payloads = backend.batch_get(repeat_keys)
    meta_payloads = backend.batch_get(meta_keys)

    if len(repeat_payloads) < achieved_repeats:
        log.warning(
            "archive_combination: expected %d repeats for %s, found %d -- skipping",
            achieved_repeats,
            combo_key,
            len(repeat_payloads),
        )
        return False

    rows: list[tuple[str, str, bytes]] = [(combo_key, "json", json.dumps(ptr_payload).encode("utf-8"))]

    blob_keys_needed: list[str] = []
    for i in range(achieved_repeats):
        key = repeat_keys[i]
        payload = repeat_payloads.get(key)
        if payload is None:
            log.warning("archive_combination: missing repeat %d for %s, skipping", i, combo_key)
            return False
        rows.append((key, "json", json.dumps(payload).encode("utf-8")))
        try:
            raw = payload["data"][0]["results"]
            entries = json.loads(raw)["entries"]
        except Exception:
            log.warning("archive_combination: could not parse repeat %d for %s, skipping", i, combo_key)
            return False
        for entry in entries:
            if entry.get("_blob"):
                blob_keys_needed.append(f"blob:{combo_key}:{i}:{entry.get('type', 'unknown')}")

    for key, payload in meta_payloads.items():
        rows.append((key, "json", json.dumps(payload).encode("utf-8")))

    blob_payloads = backend.blob_batch_get(blob_keys_needed) if blob_keys_needed else {}
    missing_blobs = [k for k in blob_keys_needed if k not in blob_payloads]
    if missing_blobs:
        log.warning(
            "archive_combination: missing %d blob(s) for %s (e.g. %s), skipping",
            len(missing_blobs),
            combo_key,
            missing_blobs[0],
        )
        return False
    for key, data in blob_payloads.items():
        rows.append((key, "blob", data))

    archive.write_combo(combo_key, rows)

    # Verify the pointer round-trips before touching anything live.
    verify_ptr = archive.get(combo_key)
    if verify_ptr != ptr_payload:
        log.error("archive_combination: verification failed for %s, leaving live data in place", combo_key)
        return False

    # backend.delete_many() (the wrapper) also purges the combo's archive file, since
    # every *external* caller (nv cache clean, purge_cache_and_artifacts_for_combinations)
    # uses delete_many for a full-combo wipe. Here it's the opposite: we just wrote the
    # archive file and are cleaning up the live rows that made it redundant, so this must
    # go straight to the live backend and leave the freshly-written archive alone.
    live_keys = [combo_key, *repeat_keys, *meta_keys, *blob_keys_needed]
    backend._live.delete_many(live_keys)
    log.info(
        "Archived combination %s (%d repeats, %d blobs) to Parquet", combo_key, achieved_repeats, len(blob_payloads)
    )
    return True
