from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import polars as pl


class SqliteCache:
    """A thread-safe key-value cache backed by SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA synchronous=NORMAL;")
        return self._local.conn

    def _ensure_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            conn.commit()

    def get(self, key: str) -> dict | None:
        try:
            conn = self._get_conn()
            cur = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
        except Exception:
            pass
        return None

    def set(self, key: str, value: dict):
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            conn.commit()
        except Exception:
            pass

    def delete(self, key: str):
        try:
            conn = self._get_conn()
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
        except Exception:
            pass

    def __iter__(self):
        try:
            conn = self._get_conn()
            cur = conn.execute("SELECT key FROM cache")
            for row in cur:
                yield row[0]
        except Exception:
            pass


class CategoryCache:
    """A cache for a specific category of simulations."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = SqliteCache(db_path)

    @staticmethod
    def _json_dumps_canonical(obj: Any) -> str:
        """Dump JSON with sorted keys for stable hashing."""
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @staticmethod
    def make_key(config: dict) -> str:
        """Make a stable cache key from a configuration dictionary."""
        canon = CategoryCache._json_dumps_canonical(config)
        return hashlib.md5(canon.encode("utf-8")).hexdigest()

    def load_df(self, key: str) -> pl.DataFrame | None:
        """Load a cached Polars DataFrame by key."""
        try:
            obj = self.backend.get(key)
            # Strict checking: must be our dict wrapper format
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

    def save_df(self, df: pl.DataFrame, key: str, metadata: dict | None = None) -> Path:
        """Save a Polars DataFrame with optional metadata."""
        # Simple in-memory serialization for now (JSON)
        # Note: Large data should ideally not be stored in SQLite text column,
        # but for this use case it seems fine as we replace the file-based system.
        payload = {
            "__nvision_cache__": "dataframe",
            "columns": list(df.columns),
            "data": df.to_dicts(),
        }
        if metadata:
            payload.update(metadata)

        self.backend.set(key, payload)
        return self.db_path


class CacheBridge:
    """Bridges the program to the cache interface."""

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self.nv_center = CategoryCache(cache_root / "nv_center.db")
        self.complementary = CategoryCache(cache_root / "complementary.db")

    def get_cache_for_category(self, category: str) -> CategoryCache:
        if category == "NVCenter":
            return self.nv_center
        return self.complementary

    def make_key(self, config: dict) -> str:
        return CategoryCache.make_key(config)


__all__ = ["CacheBridge", "CategoryCache", "SqliteCache"]
