from __future__ import annotations

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

    def list_combinations(self) -> list[dict[str, Any]]:
        """Return all stored combination configs as kwargs for get_cached_combination.

        Iterates both DB backends and returns one dict per streaming pointer entry,
        with ``repeats`` set to the achieved count. Used by _restore_missing_graphs
        to restore graph files from cache without knowing the original run params.
        """
        results: list[dict[str, Any]] = []
        stores = [self.nv_center, self.complementary]
        for store in stores:
            for key in store.backend:
                try:
                    payload = store.backend.get(key)
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
                    results.append(
                        {
                            "generator": config["generator"],
                            "noise": config["noise"],
                            "strategy": config["strategy"],
                            "repeats": achieved,
                            "seed": config["seed"],
                            "max_steps": config["max_steps"],
                            "timeout_s": config["timeout_s"],
                            "repeat_offset": int(config.get("repeat_offset", 0)),
                        }
                    )
                except Exception:
                    continue
        return results

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
