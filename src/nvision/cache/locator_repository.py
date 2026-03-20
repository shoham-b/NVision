"""Business layer: locator result payloads on top of :class:`CategoryDataStore`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from nvision.cache.data_store import CategoryDataStore
from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_keys import locator_combination_cache_config, locator_repeat_cache_config

CachedComboResults = list[tuple[list[dict[str, Any]], dict[str, Any]]]


class LocatorResultsRepository:
    """Load/save cached locator runs using semantic configs and stable row keys."""

    def __init__(self, store: CategoryDataStore) -> None:
        self._store = store

    @property
    def backend(self) -> Any:
        """Expose SQLite backend for admin tooling (list/clean)."""
        return self._store.backend

    @staticmethod
    def make_key(config: dict[str, Any]) -> str:
        """Stable hash for ``config`` (same as historical :meth:`CategoryCache.make_key`)."""
        return stable_config_hash(config)

    @staticmethod
    def combination_cache_hash(
        *,
        generator: str,
        noise: str,
        strategy: str,
        repeats: int,
        seed: int,
        max_steps: int,
        timeout_s: int,
    ) -> str:
        """Storage key hash for a full (generator, noise, strategy) combination (executor / render)."""
        cfg = locator_combination_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            repeats=repeats,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
        return stable_config_hash(cfg)

    def close(self) -> None:
        self._store.close()

    def get_cached_combination(
        self,
        *,
        generator: str,
        noise: str,
        strategy: str,
        repeats: int,
        seed: int,
        max_steps: int,
        timeout_s: int,
    ) -> CachedComboResults | None:
        """Retrieve cached simulation results for one combination (same parameters as a locator run)."""
        config = locator_combination_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            repeats=repeats,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
        return self._get_cached_results_for_config(config)

    def _get_cached_results_for_config(self, config: dict[str, Any]) -> CachedComboResults | None:
        """Retrieve cached simulation results for a combination config dict (internal wire format)."""
        key = stable_config_hash(config)
        cached_df = self._store.load_df(key)
        if cached_df is not None and "results" in cached_df.columns and not cached_df.is_empty():
            cached_payload_raw = cached_df.get_column("results")[0]
            if isinstance(cached_payload_raw, str):
                try:
                    cached_payload = json.loads(cached_payload_raw)
                    cached_results: CachedComboResults = []
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

    def save_cached_combination(
        self,
        *,
        generator: str,
        noise: str,
        strategy: str,
        repeats: int,
        seed: int,
        max_steps: int,
        timeout_s: int,
        results: CachedComboResults,
    ) -> Path:
        """Persist full combination results."""
        config = locator_combination_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            repeats=repeats,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
        key = stable_config_hash(config)
        combo_payload = [
            {"entries": entries, "main_result_row": main_result_row} for entries, main_result_row in results
        ]
        combo_df = pl.DataFrame({"results": [json.dumps(combo_payload)]})
        return self._store.save_df(combo_df, key, metadata={"config": config})

    def save_cached_repeat_slice(
        self,
        *,
        generator: str,
        noise: str,
        strategy: str,
        repeat: int,
        seed: int,
        max_steps: int,
        timeout_s: int,
        entries: list[dict[str, Any]],
        result_row: dict[str, Any],
    ) -> Path:
        """Persist a single repeat slice."""
        config = locator_repeat_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            repeat=repeat,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
        key = stable_config_hash(config)
        cache_df = pl.DataFrame(
            {
                "plot_manifest": [json.dumps(entries)],
                "result_row": [json.dumps(result_row)],
            }
        )
        return self._store.save_df(cache_df, key, metadata={"config": config})
