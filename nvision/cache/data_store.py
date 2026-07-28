"""Persistence for Polars payloads (DB layer only) — SQLite or shared MySQL backend."""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from typing import Any

import polars as pl
from polars.exceptions import ColumnNotFoundError

from nvision.cache.mysql import MySqlCache
from nvision.cache.sqlite import ShardedSqliteCache


class CategoryDataStore:
    """One category DB: opaque string keys → serialized Polars frames.

    Backed by local SQLite files by default, or a shared MySQL database when
    ``NVISION_CACHE_BACKEND=mysql`` -- see ``nvision.cache.mysql.MySqlCache``.
    ``shard_suffix`` is only meaningful for the MySQL backend: pass the
    current shard's index when writing from a sharded worker pod, or leave it
    ``None`` (the default) for local/single-writer runs and for read-only
    aggregate access (``nv serve``, `nv render`, `nv cache` admin commands),
    which transparently see all shards' data merged.
    """

    def __init__(self, db_path: Path, *, shard_suffix: str | None = None) -> None:
        self.db_path = db_path
        backend_kind = os.getenv("NVISION_CACHE_BACKEND", "sqlite").lower()
        if backend_kind == "mysql":
            self._backend: ShardedSqliteCache | MySqlCache = MySqlCache(
                table_prefix=db_path.stem, shard_suffix=shard_suffix
            )
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._backend = ShardedSqliteCache(db_path)

    @property
    def backend(self) -> ShardedSqliteCache | MySqlCache:
        """Low-level KV store (used by admin CLI for iteration)."""
        return self._backend

    def close(self) -> None:
        self._backend.close()

    def load_df(self, key: str) -> pl.DataFrame | None:
        """Load a cached Polars DataFrame by key."""
        try:
            obj = self._backend.get(key)
            if isinstance(obj, dict) and obj.get("__nvision_cache__") == "dataframe":
                rows = obj.get("data", [])
                df = pl.DataFrame(rows)
                columns = obj.get("columns")
                if columns:
                    with suppress(ColumnNotFoundError):
                        df = df.select(columns)
                return df
            return None
        except Exception:
            return None

    def load_df_batch(self, keys: list[str]) -> dict[str, pl.DataFrame]:
        """Load multiple DataFrames in a single batch. Returns {key: DataFrame} for found keys."""
        raw = self._backend.batch_get(keys)
        result: dict[str, pl.DataFrame] = {}
        for key, obj in raw.items():
            if not (isinstance(obj, dict) and obj.get("__nvision_cache__") == "dataframe"):
                continue
            try:
                df = pl.DataFrame(obj.get("data", []))
                columns = obj.get("columns")
                if columns:
                    with suppress(Exception):
                        df = df.select(columns)
                result[key] = df
            except Exception:
                pass
        return result

    def keys_exist_batch(self, keys: list[str]) -> set[str]:
        """Return the subset of ``keys`` that exist in the store (no DataFrame parsing)."""
        return self._backend.keys_exist_batch(keys)

    def save_df_batch(self, items: dict[str, pl.DataFrame | dict]) -> None:
        """Persist multiple keys atomically — delegates to batch_set for one transaction per shard."""
        payloads: dict[str, dict] = {}
        for key, item in items.items():
            if isinstance(item, pl.DataFrame):
                payloads[key] = {
                    "__nvision_cache__": "dataframe",
                    "columns": list(item.columns),
                    "data": item.to_dicts(),
                }
            else:
                # Pre-built payload dict (e.g. from save_repeat fast path)
                payloads[key] = item
        self._backend.batch_set(payloads)

    def save_df(self, df: pl.DataFrame, key: str, metadata: dict[str, Any] | None = None) -> Path:
        """Persist a Polars DataFrame under ``key`` with optional metadata merged into the blob."""
        payload: dict[str, Any] = {
            "__nvision_cache__": "dataframe",
            "columns": list(df.columns),
            "data": df.to_dicts(),
        }
        if metadata:
            payload.update(metadata)

        self._backend.set(key, payload)
        return self.db_path

    def delete(self, key: str) -> None:
        """Delete a key from the store."""
        with suppress(Exception):
            self._backend.delete(key)

    def save_blob(self, key: str, data: bytes) -> None:
        """Persist raw bytes under ``key`` in the BLOB table — no text encoding at all."""
        self._backend.blob_set(key, data)

    def load_blob(self, key: str) -> bytes | None:
        return self._backend.blob_get(key)

    def load_blob_batch(self, keys: list[str]) -> dict[str, bytes]:
        return self._backend.blob_batch_get(keys)
