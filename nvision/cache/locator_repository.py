"""Business layer: locator result payloads on top of :class:`CategoryDataStore`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import polars as pl

from nvision.cache.data_store import CategoryDataStore
from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_keys import (
    combination_base_cache_config,
    locator_combination_cache_config,
)
from nvision.cache.repeats_repository import RepeatsRepository

CachedComboResults = list[tuple[list[dict[str, Any]], dict[str, Any]]]

STREAMING_REPEAT_THRESHOLD = int(os.getenv("NVISION_STREAMING_REPEAT_THRESHOLD", "5"))


class LocatorResultsRepository:
    """Load/save cached locator runs using semantic configs and stable row keys."""

    def __init__(self, store: CategoryDataStore) -> None:
        self._store = store
        self._repeats = RepeatsRepository(store)

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
        """Retrieve cached simulation results for one combination.

        Checks for streaming pointers first, then falls back to inline entries.
        """
        # 1. Check streaming pointer
        ptr_config = combination_base_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
        ptr_key = stable_config_hash(ptr_config)
        ptr_df = self._store.load_df(ptr_key)
        if ptr_df is not None and not ptr_df.is_empty():
            achieved = int(ptr_df.get_column("achieved_repeats")[0])
            if achieved >= repeats:
                # Full streaming hit
                return self._repeats.load_repeats(ptr_key, repeats)

        # 2. Fall back to inline entry
        inline_config = locator_combination_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            repeats=repeats,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
        return self._get_cached_results_for_config(inline_config)

    def get_cached_combination_partial(
        self,
        *,
        generator: str,
        noise: str,
        strategy: str,
        repeats: int,
        seed: int,
        max_steps: int,
        timeout_s: int,
    ) -> tuple[CachedComboResults, int]:
        """Retrieve partial cached results and count for resumable runs.

        Only checks the streaming pointer path.
        """
        ptr_config = combination_base_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
        ptr_key = stable_config_hash(ptr_config)
        ptr_df = self._store.load_df(ptr_key)
        if ptr_df is not None and not ptr_df.is_empty():
            achieved = int(ptr_df.get_column("achieved_repeats")[0])
            # Return up to requested repeats
            count = min(achieved, repeats)
            results = self._repeats.load_repeats(ptr_key, count)
            return results, len(results)

        return [], 0

    def get_cached_combination_by_config(self, config: dict[str, Any]) -> CachedComboResults | None:
        """Retrieve cached simulation results from a full combination config payload."""
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
        """Persist full combination results.

        Uses streaming format if repeats > threshold, otherwise inline.
        """
        if repeats > STREAMING_REPEAT_THRESHOLD:
            # Streaming path: write pointer and all repeats
            ptr_config = combination_base_cache_config(
                generator=generator,
                noise=noise,
                strategy=strategy,
                seed=seed,
                max_steps=max_steps,
                timeout_s=timeout_s,
            )
            stable_config_hash(ptr_config)
            self.append_cached_repeats(
                generator=generator,
                noise=noise,
                strategy=strategy,
                seed=seed,
                max_steps=max_steps,
                timeout_s=timeout_s,
                new_results=results,
                start_idx=0,
            )
            return self._store.db_path

        # Inline path
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

    def append_cached_repeats(
        self,
        *,
        generator: str,
        noise: str,
        strategy: str,
        seed: int,
        max_steps: int,
        timeout_s: int,
        new_results: CachedComboResults,
        start_idx: int,
    ) -> None:
        """Append new repeats to a streaming cache entry and update pointer."""
        ptr_config = combination_base_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
        ptr_key = stable_config_hash(ptr_config)

        # Save each repeat row
        for i, (entries, main_result_row) in enumerate(new_results):
            self._repeats.save_repeat(ptr_key, start_idx + i, entries, main_result_row)

        # Update pointer row
        new_total = start_idx + len(new_results)
        ptr_df = pl.DataFrame({"achieved_repeats": [new_total], "streaming": [True]})
        self._store.save_df(ptr_df, ptr_key, metadata={"config": ptr_config})

    def save_repeat(
        self,
        *,
        generator: str,
        noise: str,
        strategy: str,
        seed: int,
        max_steps: int,
        timeout_s: int,
        repeat_idx: int,
        entries: list[dict[str, Any]],
        main_result_row: dict[str, Any],
    ) -> None:
        """Save a single repeat (streaming). Caller is responsible for updating the pointer later if needed."""
        ptr_config = combination_base_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
        ptr_key = stable_config_hash(ptr_config)
        self._repeats.save_repeat(ptr_key, repeat_idx, entries, main_result_row)
