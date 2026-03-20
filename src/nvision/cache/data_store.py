"""SQLite-backed persistence for Polars payloads (DB layer only)."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import polars as pl

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
                    with suppress(pl.ColumnNotFoundError):
                        df = df.select(columns)
                return df
            return None
        except Exception:
            return None

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
