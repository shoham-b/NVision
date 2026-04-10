"""Test script to verify cache entries are loaded with content stripped."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nvision import CacheBridge

ARTIFACTS_DIR = Path("artifacts")
CACHE_DIR = ARTIFACTS_DIR / "cache"


def test_cache_loading():
    if not CACHE_DIR.exists():
        print("No cache directory found")
        return

    bridge = CacheBridge(CACHE_DIR)

    # Count entries and check for content field
    total_entries = 0
    entries_with_content = 0
    entries_without_content = 0
    type_counts = {}

    # Try to iterate through cache
    for category in ["NVCenter", "Complementary"]:
        cache = bridge.get_cache_for_category(category)
        store = cache._store

        try:
            for key in store.backend:
                payload = store.backend.get(key)
                if not isinstance(payload, dict):
                    continue

                cfg = payload.get("config")
                if not isinstance(cfg, dict) or cfg.get("kind") != "locator_combination":
                    continue

                print(f"Found cache entry: {cfg.get('generator')}/{cfg.get('noise')}/{cfg.get('strategy')}")

                # Try to load results
                results = cache.get_cached_combination_by_config(cfg)
                if results:
                    for entries, _ in results:
                        for entry in entries:
                            total_entries += 1
                            etype = entry.get("type", "unknown")
                            type_counts[etype] = type_counts.get(etype, 0) + 1

                            if "content" in entry:
                                entries_with_content += 1
                                # Show size of content
                                content_size = len(entry["content"]) if isinstance(entry["content"], str) else 0
                                if content_size > 1000:
                                    print(f"  Entry {etype} has content field with {content_size} chars")
                            else:
                                entries_without_content += 1
        except Exception as e:
            print(f"Error iterating {category} cache: {e}")

    bridge.close()

    print("\nSummary:")
    print(f"  Total entries: {total_entries}")
    print(f"  With content field: {entries_with_content}")
    print(f"  Without content field: {entries_without_content}")
    print(f"  Type counts: {type_counts}")


if __name__ == "__main__":
    test_cache_loading()
