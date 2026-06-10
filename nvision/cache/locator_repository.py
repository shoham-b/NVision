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

STREAMING_REPEAT_THRESHOLD = int(os.getenv("NVISION_STREAMING_REPEAT_THRESHOLD", "0"))


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
        repeat_offset: int = 0,
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
            repeat_offset=repeat_offset,
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
        repeat_offset: int = 0,
        allow_gaps: bool = False,
    ) -> CachedComboResults | None:
        """Retrieve cached simulation results for one combination.

        Checks for streaming pointers first, then falls back to inline entries for backward compatibility.
        """
        # 1. Check streaming pointer
        ptr_config = combination_base_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
            repeat_offset=repeat_offset,
        )
        ptr_key = stable_config_hash(ptr_config)
        ptr_df = self._store.load_df(ptr_key)

        # Temporary fallback for schema version 8
        if ptr_df is None or ptr_df.is_empty():
            ptr_config_v8 = dict(ptr_config)
            ptr_config_v8["schema_version"] = 8
            ptr_key_v8 = stable_config_hash(ptr_config_v8)
            ptr_df_v8 = self._store.load_df(ptr_key_v8)
            if ptr_df_v8 is not None and not ptr_df_v8.is_empty():
                ptr_key = ptr_key_v8
                ptr_df = ptr_df_v8

        if ptr_df is not None and not ptr_df.is_empty():
            achieved = int(ptr_df.get_column("achieved_repeats")[0])
            if achieved >= repeat_offset + repeats:
                # Full streaming hit
                return self._repeats.load_repeats(ptr_key, repeats, start_idx=repeat_offset, allow_gaps=allow_gaps)

        # 2. Fall back to inline entry
        inline_config = locator_combination_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            repeats=repeats,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
            repeat_offset=repeat_offset,
        )
        inline_res = self._get_cached_results_for_config(inline_config)
        if inline_res is not None:
            return inline_res

        inline_config_v8 = dict(inline_config)
        inline_config_v8["schema_version"] = 8
        return self._get_cached_results_for_config(inline_config_v8)

    def get_cached_combination_fast(
        self,
        *,
        generator: str,
        noise: str,
        strategy: str,
        repeats: int,
        seed: int,
        max_steps: int,
        timeout_s: int,
        repeat_offset: int = 0,
        out_dir: Path | None = None,
    ) -> CachedComboResults | None:
        """Fast cache restore: use lightweight ``:meta`` sidecars to avoid loading MB of plot bytes.

        Falls back to ``None`` (caller must use :meth:`get_cached_combination`) when:
        - Sidecars are missing (old cache entries written before the sidecar feature)
        - ``out_dir`` is not provided
        - Any referenced gz file does not exist on disk (needs ``restore_graphs``)
        """
        if out_dir is None:
            return None

        ptr_config = combination_base_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
            repeat_offset=repeat_offset,
        )
        ptr_key = stable_config_hash(ptr_config)
        ptr_df = self._store.load_df(ptr_key)

        if ptr_df is None or ptr_df.is_empty():
            ptr_config_v8 = dict(ptr_config)
            ptr_config_v8["schema_version"] = 8
            ptr_key_v8 = stable_config_hash(ptr_config_v8)
            ptr_df_v8 = self._store.load_df(ptr_key_v8)
            if ptr_df_v8 is not None and not ptr_df_v8.is_empty():
                ptr_key = ptr_key_v8
                ptr_df = ptr_df_v8

        if ptr_df is None or ptr_df.is_empty():
            return None

        achieved = int(ptr_df.get_column("achieved_repeats")[0])
        if achieved < repeat_offset + repeats:
            return None

        meta = self._repeats.load_repeats_meta(ptr_key, repeats, start_idx=repeat_offset)
        if meta is None:
            return None

        # Only use fast path if every referenced gz file already exists on disk
        for entries, _ in meta:
            for entry in entries:
                p = entry.get("path")
                if p and p.endswith(".gz") and not (out_dir / p).exists():
                    return None

        return meta

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
        repeat_offset: int = 0,
        chunk_size: int | None = None,
        allow_gaps: bool = False,
    ) -> tuple[CachedComboResults, int]:
        """Retrieve partial cached results and count for resumable runs.

        Only checks the streaming pointer path.

        Parameters
        ----------
        chunk_size:
            When this task is a sub-task split from a larger one, ``chunk_size``
            is the number of repeats this sub-task owns.  The returned count is
            capped at ``chunk_size`` so that a sub-task never claims more cached
            repeats than it is responsible for.  If ``None``, defaults to
            ``repeats`` (no sub-task splitting).
        """
        ptr_config = combination_base_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
            repeat_offset=repeat_offset,
        )
        ptr_key = stable_config_hash(ptr_config)
        ptr_df = self._store.load_df(ptr_key)

        # Temporary fallback for schema version 8
        if ptr_df is None or ptr_df.is_empty():
            ptr_config_v8 = dict(ptr_config)
            ptr_config_v8["schema_version"] = 8
            ptr_key_v8 = stable_config_hash(ptr_config_v8)
            ptr_df_v8 = self._store.load_df(ptr_key_v8)
            if ptr_df_v8 is not None and not ptr_df_v8.is_empty():
                ptr_key = ptr_key_v8
                ptr_df = ptr_df_v8

        if ptr_df is not None and not ptr_df.is_empty():
            achieved = int(ptr_df.get_column("achieved_repeats")[0])
            # Cap to this sub-task's own chunk, not the total repeats
            max_count = chunk_size if chunk_size is not None else repeats
            if achieved > repeat_offset:
                count = min(achieved - repeat_offset, max_count)
                results = self._repeats.load_repeats(ptr_key, count, start_idx=repeat_offset, allow_gaps=allow_gaps)
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

                    if not all(type(r) is dict for r in cached_payload):
                        return None

                    cached_results: CachedComboResults = [
                        (record["entries"], record["main_result_row"]) for record in cached_payload
                    ]

                    if all(type(e) is list and type(r) is dict for e, r in cached_results):
                        return cached_results
                except (KeyError, TypeError):
                    pass
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
        repeat_offset: int = 0,
        results: CachedComboResults,
        start_idx: int = 0,
    ) -> Path:
        """Persist full combination results.

        Uses streaming format always to prevent duplicate entries and allow resumes.

        Parameters
        ----------
        start_idx:
            Global repeat index of the first result in ``results``.  For
            sub-tasks this must be ``repeat_offset + n_already_cached`` so that
            results land at the correct positions in the shared pointer.
        """
        self.append_cached_repeats(
            generator=generator,
            noise=noise,
            strategy=strategy,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
            repeat_offset=repeat_offset,
            new_results=results,
            start_idx=start_idx,
        )
        return self._store.db_path

    def append_cached_repeats(
        self,
        *,
        generator: str,
        noise: str,
        strategy: str,
        seed: int,
        max_steps: int,
        timeout_s: int,
        repeat_offset: int = 0,
        new_results: CachedComboResults,
        start_idx: int,
    ) -> None:
        """Append new repeats to a streaming cache entry and update pointer."""
        import datetime

        updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ptr_config = combination_base_cache_config(
            generator=generator,
            noise=noise,
            strategy=strategy,
            seed=seed,
            max_steps=max_steps,
            timeout_s=timeout_s,
            repeat_offset=repeat_offset,
        )
        ptr_key = stable_config_hash(ptr_config)

        # Save each repeat row
        for i, (entries, main_result_row) in enumerate(new_results):
            self._repeats.save_repeat(ptr_key, start_idx + i, entries, main_result_row)

        # Update pointer row safely
        new_total = start_idx + len(new_results)

        # Calculate the true sequential achieved_repeats by counting from 0
        # This prevents holes in the cache if sub-tasks finish out of order
        actual_achieved = self._repeats.count_saved(ptr_key, max_expected=new_total + 100)

        existing_total = 0
        existing_df = self._store.load_df(ptr_key)
        if existing_df is not None and not existing_df.is_empty():
            existing_total = int(existing_df.get_column("achieved_repeats")[0])

        if actual_achieved > existing_total:
            ptr_df = pl.DataFrame({"achieved_repeats": [actual_achieved], "streaming": [True]})
            self._store.save_df(ptr_df, ptr_key, metadata={"config": ptr_config, "updated_at": updated_at})

    def save_repeat(
        self,
        *,
        generator: str,
        noise: str,
        strategy: str,
        seed: int,
        max_steps: int,
        timeout_s: int,
        repeat_offset: int = 0,
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
            repeat_offset=repeat_offset,
        )
        ptr_key = stable_config_hash(ptr_config)
        self._repeats.save_repeat(ptr_key, repeat_idx, entries, main_result_row)

    def purge_cached_combination(
        self,
        *,
        generator: str,
        noise: str,
        strategy: str,
        repeats: int,
        seed: int,
        max_steps: int,
        timeout_s: int,
        repeat_offset: int = 0,
    ) -> None:
        """Purge all cached results (both inline and streaming repeats) for this combination across all partitions."""
        backend = self._store.backend
        keys_to_delete = []
        for k in backend:
            payload = backend.get(k)
            if isinstance(payload, dict) and "config" in payload:
                config = payload["config"]
                # Match the combination base params, independent of repeat_offset and repeats
                if (
                    config.get("generator") == generator
                    and config.get("noise") == noise
                    and config.get("strategy") == strategy
                    and config.get("seed") == seed
                    and config.get("max_steps") == max_steps
                    and config.get("timeout_s") == timeout_s
                ):
                    keys_to_delete.append(k)

        # Delete all matching pointer/inline keys and all associated repeat rows
        for k in keys_to_delete:
            self._store.delete(k)
            for i in range(1000):
                rep_key = self._repeats.make_repeat_key(k, i)
                self._store.delete(rep_key)
                self._store.delete(rep_key + ":meta")
