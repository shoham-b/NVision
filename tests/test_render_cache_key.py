"""``nvision render`` must use the same SQLite cache key as the executor."""

from __future__ import annotations

from nvision import LocatorResultsRepository
from nvision import locator_combination_cache_config
from nvision import stable_config_hash


def test_render_combo_cache_hash_matches_executor() -> None:
    """Regression: render and executor must address the same combination storage key."""
    kwargs = dict(
        generator="NVCenter-one_peak",
        noise="NoNoise",
        strategy="SimpleSweep",
        repeats=5,
        seed=42,
        max_steps=150,
        timeout_s=1500,
    )
    h = LocatorResultsRepository.combination_cache_hash(**kwargs)
    assert h == LocatorResultsRepository.combination_cache_hash(**kwargs)

    full = locator_combination_cache_config(**kwargs)
    stale = {k: v for k, v in full.items() if k != "schema_version"}
    assert stable_config_hash(full) != stable_config_hash(stale)
