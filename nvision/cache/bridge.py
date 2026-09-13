from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nvision.cache.data_store import CategoryDataStore
from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_repository import CachedComboResults, LocatorResultsRepository


class CacheBridge:
    """Root cache access: category-scoped data stores and locator repositories.

    ``shard_suffix`` is forwarded to each ``CategoryDataStore`` -- only relevant
    under the MySQL backend, where it selects a sharded worker's own table
    (write mode) vs. the default read-aggregate-across-all-shards mode.
    """

    def __init__(self, cache_root: Path, *, shard_suffix: str | None = None) -> None:
        self.cache_root = cache_root
        self.nv_center = CategoryDataStore(cache_root / "nv_center.db", shard_suffix=shard_suffix)
        self.complementary = CategoryDataStore(cache_root / "complementary.db", shard_suffix=shard_suffix)

    def get_cache_for_category(self, category: str) -> LocatorResultsRepository:
        """Business API for locator results (combination + repeat rows) in this category."""
        if category == "NVCenter":
            return LocatorResultsRepository(self.nv_center)
        return LocatorResultsRepository(self.complementary)

    def make_key(self, config: dict) -> str:
        return stable_config_hash(config)

    def _iter_combination_payloads(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        """Yield (combo_kwargs, raw_payload) for every stored combination config.

        Shared implementation behind list_combinations (kwargs for
        get_cached_combination) and list_combinations_with_updated_at (adds
        the payload's updated_at for freshness-based filtering). See
        list_combinations's docstring for why the repeat:/blob: prefixes are
        filtered client-side before fetching.
        """
        stores = [self.nv_center, self.complementary]
        for store in stores:
            backend = store.backend
            candidate_keys = [k for k in backend if not k.startswith("repeat:") and not k.startswith("blob:")]
            payloads = backend.batch_get(candidate_keys)
            for payload in payloads.values():
                try:
                    if not isinstance(payload, dict):
                        continue
                    config = payload.get("config")
                    if not config or config.get("kind") != "locator_combination_pointer":
                        continue
                    data = payload.get("data", [])
                    if not data or not isinstance(data[0], dict):
                        continue
                    achieved = int(data[0].get("achieved_repeats", 0))
                    if achieved <= 0:
                        continue
                    combo = {
                        "generator": config["generator"],
                        "noise": config["noise"],
                        "strategy": config["strategy"],
                        "repeats": achieved,
                        "seed": config["seed"],
                        "max_steps": config["max_steps"],
                        "timeout_s": config["timeout_s"],
                        "repeat_offset": int(config.get("repeat_offset", 0)),
                    }
                    yield combo, payload
                except Exception:
                    continue

    def list_combinations(self) -> list[dict[str, Any]]:
        """Return all stored combination configs as kwargs for get_cached_combination.

        Iterates both DB backends and returns one dict per streaming pointer entry,
        with ``repeats`` set to the achieved count. Used by _restore_missing_graphs
        to restore graph files from cache without knowing the original run params.

        Pointer rows are bare hashes; every repeat payload and its ``:meta``
        sidecar are always keyed ``repeat:...`` (see
        ``RepeatsRepository.make_repeat_key``), and every per-repeat graph blob
        is keyed ``blob:...`` (see ``make_blob_key``). Filtering those out
        client-side before fetching means one key-listing query plus one
        batched IN(...) fetch per store, instead of one network round-trip per
        row in the entire cache table (under MySQL, that's every repeat across
        every shard -- the dominant cost of building /api/manifest). Leaving
        either prefix in also risks a single IN(...) query with one SQL
        variable per candidate key blowing past SQLite's variable limit once
        the cache has enough blob entries, which fails silently (each shard
        query is wrapped in a bare except) and makes the manifest come back
        empty.
        """
        return [combo for combo, _payload in self._iter_combination_payloads()]

    def list_combinations_with_updated_at(self) -> list[dict[str, Any]]:
        """Same rows as list_combinations, each with an added 'updated_at' string.

        NOT safe to splat into get_cached_combination(**combo) -- that method's
        signature has no **kwargs catch-all, so the extra key raises. This is
        for freshness-based filtering only (e.g. api_server.py's manifest
        keeping only the most-recently-updated generation per
        generator/noise/strategy, so old superseded max_steps/timeout_s
        generations don't bloat the response).
        """
        return [
            {**combo, "updated_at": payload.get("updated_at", "")}
            for combo, payload in self._iter_combination_payloads()
        ]

    def get_cached_combination(self, **kwargs: Any) -> CachedComboResults | None:
        """Route get_cached_combination to the correct category store.

        Determines the category from the ``generator`` kwarg via
        ``CombinationGrid.generator_category`` and delegates to the matching
        :class:`LocatorResultsRepository`.
        """
        from nvision.sim.combinations import CombinationGrid

        generator = kwargs.get("generator", "")
        category = CombinationGrid.generator_category(str(generator))
        repo = self.get_cache_for_category(category)
        return repo.get_cached_combination(**kwargs)

    def close(self) -> None:
        self.nv_center.close()
        self.complementary.close()
