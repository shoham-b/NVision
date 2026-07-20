"""SQLite-backed persistence for Polars payloads (DB layer only)."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import polars as pl
from polars.exceptions import ColumnNotFoundError

from nvision.cache.sqlite import ShardedSqliteCache


class CategoryDataStore:
    """One category DB file: opaque string keys → serialized Polars frames."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._backend = ShardedSqliteCache(db_path)

    @property
    def backend(self) -> ShardedSqliteCache:
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
