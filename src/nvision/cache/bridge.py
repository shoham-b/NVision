from __future__ import annotations

from pathlib import Path

from nvision.cache.category import CategoryCache


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

    def close(self):
        self.nv_center.close()
        self.complementary.close()
