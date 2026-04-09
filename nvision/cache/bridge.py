from __future__ import annotations

from pathlib import Path

from nvision.cache.data_store import CategoryDataStore
from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_repository import LocatorResultsRepository


class CacheBridge:
    """Root cache access: category-scoped data stores and locator repositories."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.nv_center = CategoryDataStore(cache_root / "nv_center.db")
        self.complementary = CategoryDataStore(cache_root / "complementary.db")

    def get_cache_for_category(self, category: str) -> LocatorResultsRepository:
        """Business API for locator results (combination + repeat rows) in this category."""
        if category == "NVCenter":
            return LocatorResultsRepository(self.nv_center)
        return LocatorResultsRepository(self.complementary)

    def make_key(self, config: dict) -> str:
        return stable_config_hash(config)

    def close(self) -> None:
        self.nv_center.close()
        self.complementary.close()
