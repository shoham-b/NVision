from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import polars as pl

from nvision.cache.sqlite import SqliteCache


class CategoryCache:
    """A cache for a specific category of simulations."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = SqliteCache(db_path)

    def close(self):
        self.backend.close()

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

    def get_cached_results(self, config: dict) -> list[tuple[list[dict[str, Any]], dict[str, Any]]] | None:
        """High-level method to retrieve cached simulation results."""
        key = self.make_key(config)
        cached_df = self.load_df(key)
        if cached_df is not None and "results" in cached_df.columns and not cached_df.is_empty():
            cached_payload_raw = cached_df.get_column("results")[0]
            if isinstance(cached_payload_raw, str):
                try:
                    cached_payload = json.loads(cached_payload_raw)
                    cached_results: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
                    for record in cached_payload:
                        if not isinstance(record, dict):
                            break
                        entries = record.get("entries")
                        result_row = record.get("main_result_row")
                        if not isinstance(entries, list) or not isinstance(result_row, dict):
                            break
                        cached_results.append((entries, result_row))
                    else:
                        if cached_results:
                            return cached_results
                except Exception:
                    pass
        return None

    def save_cached_results(self, config: dict, results: list[tuple[list[dict[str, Any]], dict[str, Any]]]) -> Path:
        """High-level method to save simulation results."""
        key = self.make_key(config)
        combo_payload = [
            {"entries": entries, "main_result_row": main_result_row} for entries, main_result_row in results
        ]
        combo_df = pl.DataFrame({"results": [json.dumps(combo_payload)]})
        return self.save_df(combo_df, key, metadata={"config": config})

    def save_cached_repeat(self, config: dict, entries: list[dict[str, Any]], result_row: dict[str, Any]) -> Path:
        """High-level method to save a single simulation repeat."""
        key = self.make_key(config)
        cache_df = pl.DataFrame(
            {
                "plot_manifest": [json.dumps(entries)],
                "result_row": [json.dumps(result_row)],
            }
        )
        return self.save_df(cache_df, key, metadata={"config": config})
