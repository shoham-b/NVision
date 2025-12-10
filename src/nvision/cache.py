from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import polars as pl
from diskcache import FanoutCache

from nvision.pathutils import slugify


class CategoryCache:
    """A cache for a specific category of simulations."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _json_dumps_canonical(obj: Any) -> str:
        """Dump JSON with sorted keys for stable hashing."""
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @staticmethod
    def make_key(config: dict) -> str:
        """Make a stable cache key from a configuration dictionary."""
        canon = CategoryCache._json_dumps_canonical(config)
        return hashlib.md5(canon.encode("utf-8")).hexdigest()

    def load(self, config: dict) -> pl.DataFrame | None:
        """Load a DataFrame based on the configuration."""
        key = self.make_key(config)
        return self.load_df(key)

    def get_config(self, key: str) -> dict | None:
        """Get the configuration associated with a cache key."""
        try:
            with FanoutCache(self.cache_dir.as_posix()) as cache:
                payload = cache.get(key, default=None)
                if isinstance(payload, dict):
                    return payload.get("config")
                return None
        except Exception:
            return None

    def save(self, config: dict, df: pl.DataFrame) -> Path:
        """Save a DataFrame associated with a configuration."""
        key = self.make_key(config)
        return self.save_df(df, key, metadata={"config": config})

    def load_df(self, key: str) -> pl.DataFrame | None:
        """Load a cached Polars DataFrame by key."""
        try:
            with FanoutCache(self.cache_dir.as_posix()) as cache:
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

    def save_df(self, df: pl.DataFrame, key: str, metadata: dict | None = None) -> Path:
        """Save a Polars DataFrame with optional metadata."""
        if any(dtype == pl.Object for dtype in df.dtypes):
            return self.cache_dir / key

        payload = {
            "__nvision_cache__": "dataframe",
            "columns": list(df.columns),
            "data": df.to_dicts(),
        }
        if metadata:
            payload.update(metadata)

        try:
            with FanoutCache(self.cache_dir.as_posix()) as cache:
                cache.set(key, payload)
        except Exception as e:
            print(f"CACHE SAVE ERROR: {e}")
            pass
        return self.cache_dir / key

    def list_content(self) -> list[dict]:
        """List the configurations of all cached items."""
        items = []
        try:
            with FanoutCache(self.cache_dir.as_posix()) as cache:
                for key in cache:
                    try:
                        payload = cache.get(key)
                        if isinstance(payload, dict) and "config" in payload:
                            items.append(payload["config"])
                    except Exception:
                        pass
        except Exception:
            pass
        return items


class SimulationCache:
    """Manager for simulation caches organized by category."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def for_category(self, category: str) -> CategoryCache:
        """Get the cache for a specific category."""
        slug = slugify(category or "unknown")
        return CategoryCache(self.base_dir / slug)
