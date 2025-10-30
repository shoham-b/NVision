from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import polars as pl
from diskcache import Cache


class DataFrameCache:
    """A simple cache wrapper for Polars DataFrames backed by diskcache.

    This class encapsulates the cache directory and exposes small helpers for
    making deterministic keys and storing/loading DataFrames.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._ensure_dir()

    # -------------------
    # Key helpers
    # -------------------
    @staticmethod
    def _json_dumps_canonical(obj: Any) -> str:
        """Dump JSON with sorted keys and stable formatting for hashing.

        Non-serializable objects should be converted by the caller to basic types.
        """
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @staticmethod
    def make_key(config: dict) -> str:
        """Make a stable cache key from a configuration dictionary.

        The dictionary should contain only JSON-serializable primitives
        (str, int, float, bool, None, list, dict).
        """
        canon = DataFrameCache._json_dumps_canonical(config)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    # -------------------
    # IO helpers
    # ---------------````----
    def _ensure_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_df(self, key: str) -> pl.DataFrame | None:
        """Load a cached Polars DataFrame by key; return None on miss or error."""
        try:
            with Cache(self.cache_dir.as_posix()) as cache:
                obj = cache.get(key, default=None)
                if isinstance(obj, pl.DataFrame):
                    return obj
                if isinstance(obj, dict) and obj.get("__nvision_cache__") == "dataframe":
                    rows = obj.get("data", [])
                    df = pl.DataFrame(rows)
                    columns = obj.get("columns")
                    if columns:
                        with suppress(pl.ColumnNotFoundError):
                            df = df.select(columns)
                    return df
                if isinstance(obj, list):
                    return pl.DataFrame(obj)
                return None
        except Exception:
            return None

    def save_df(self, df: pl.DataFrame, key: str) -> Path:
        """Save a Polars DataFrame and return a logical cache path (cache_dir/key).

        Note: diskcache manages the underlying storage; the returned Path is for
        logging only.
        """
        if any(dtype == pl.Object for dtype in df.dtypes):
            # Object columns (e.g., callables, custom classes) are not safely serializable.
            # Skip caching silently; callers will recompute results on subsequent runs.
            return self.cache_dir / key
        payload = {
            "__nvision_cache__": "dataframe",
            "columns": list(df.columns),
            "data": df.to_dicts(),
        }
        try:
            with Cache(self.cache_dir.as_posix()) as cache:
                cache.set(key, payload)
        except Exception:
            # Best effort only; ignore caching errors so CLI continues.
            pass
        return self.cache_dir / key


# ---------------------------------------------------------------------------
# Backward-compatible functional wrappers (deprecated)
# ---------------------------------------------------------------------------


def make_key(config: dict) -> str:
    """Deprecated functional wrapper around DataFrameCache.make_key."""
    return DataFrameCache.make_key(config)


def load_df(cache_dir: Path, key: str) -> pl.DataFrame | None:
    """Deprecated functional wrapper; prefer DataFrameCache(cache_dir).load_df(key)."""
    return DataFrameCache(cache_dir).load_df(key)


def save_df(df: pl.DataFrame, cache_dir: Path, key: str) -> Path:
    """Deprecated functional wrapper; prefer DataFrameCache(cache_dir).save_df(df, key)."""
    return DataFrameCache(cache_dir).save_df(df, key)
