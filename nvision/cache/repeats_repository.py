"""Storage layer for individual repeat results (streaming cache)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from nvision.cache.data_store import CategoryDataStore

# Matches LocatorResultsRepository.CachedComboResults
RepeatResult = tuple[list[dict[str, Any]], dict[str, Any]]


class RepeatsRepository:
    """Manages individual repeat rows using a 'repeat:{combo_key}:{idx}' key format."""

    def __init__(self, store: CategoryDataStore) -> None:
        self._store = store

    @staticmethod
    def make_repeat_key(combo_key: str, repeat_idx: int) -> str:
        """Storage key for a single repeat."""
        return f"repeat:{combo_key}:{repeat_idx}"

    def save_repeat(
        self, combo_key: str, repeat_idx: int, entries: list[dict[str, Any]], main_result_row: dict[str, Any]
    ) -> None:
        """Persist one repeat immediately."""
        key = self.make_repeat_key(combo_key, repeat_idx)
        payload = {"entries": entries, "main_result_row": main_result_row}
        df = pl.DataFrame({"results": [json.dumps(payload)]})
        self._store.save_df(df, key)

    def load_repeat(self, combo_key: str, repeat_idx: int) -> RepeatResult | None:
        """Load a single repeat by index."""
        key = self.make_repeat_key(combo_key, repeat_idx)
        df = self._store.load_df(key)
        if df is not None and not df.is_empty():
            raw = df.get_column("results")[0]
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                    return payload["entries"], payload["main_result_row"]
                except (Exception, KeyError):
                    pass
        return None

    def load_repeats(self, combo_key: str, count: int, start_idx: int = 0, allow_gaps: bool = False) -> list[RepeatResult]:
        """Load N repeats in order. If allow_gaps is True, missing repeats are skipped but we continue loading.
        Otherwise, we stop at the first missing repeat (gap)."""
        results: list[RepeatResult] = []
        for i in range(count):
            res = self.load_repeat(combo_key, start_idx + i)
            if res:
                results.append(res)
            elif not allow_gaps:
                # If we hit a gap, stop (assume sequential)
                break
        return results

    def count_saved(self, combo_key: str, max_expected: int = 1000) -> int:
        """Count how many sequential repeats are already saved."""
        count = 0
        for i in range(max_expected):
            self.make_repeat_key(combo_key, i)
            # Efficient check: load_df is relatively fast, but we only need to know if it exists.
            # ShardedSqliteCache doesn't have an 'exists' method, so we load.
            if self.load_repeat(combo_key, i) is not None:
                count += 1
            else:
                break
        return count
