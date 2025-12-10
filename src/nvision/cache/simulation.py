from __future__ import annotations

from pathlib import Path

from nvision.cache.category import CategoryCache
from nvision.core.paths import slugify


class SimulationCache:
    """Manager for simulation caches organized by category."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def for_category(self, category: str) -> CategoryCache:
        """Get the cache for a specific category."""
        slug = slugify(category or "unknown")
        return CategoryCache(self.base_dir / slug)
