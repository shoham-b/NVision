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

    _HEAVY_FIELDS = frozenset({"content", "content_bin", "plot_data", "_bytes"})

    def save_repeat(
        self, combo_key: str, repeat_idx: int, entries: list[dict[str, Any]], main_result_row: dict[str, Any]
    ) -> None:
        """Persist one repeat immediately.

        Also writes a lightweight ``:meta`` sidecar that omits binary content_bin
        so cache-hit runs can rebuild df_rows without deserialising MB of plot bytes.
        """
        key = self.make_repeat_key(combo_key, repeat_idx)
        payload = {"entries": entries, "main_result_row": main_result_row}
        df = pl.DataFrame({"results": [json.dumps(payload)]})
        self._store.save_df(df, key)

        # Lightweight sidecar: stripped entries + result row only (no binary blobs)
        stripped_entries = [
            {k: v for k, v in e.items() if k not in self._HEAVY_FIELDS}
            for e in entries
        ]
        meta_payload = {"entries": stripped_entries, "main_result_row": main_result_row}
        meta_df = pl.DataFrame({"results": [json.dumps(meta_payload)]})
        self._store.save_df(meta_df, key + ":meta")

    def load_repeat(self, combo_key: str, repeat_idx: int) -> RepeatResult | None:
        """Load a single repeat by index.

        Lazily backfills the ``:meta`` sidecar when it is absent so subsequent
        runs can use the fast path without a migration step.
        """
        key = self.make_repeat_key(combo_key, repeat_idx)
        df = self._store.load_df(key)
        if df is not None and not df.is_empty():
            raw = df.get_column("results")[0]
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                    entries = payload["entries"]
                    row = payload["main_result_row"]
                    # Backfill sidecar if missing so next load can use fast path
                    meta_key = key + ":meta"
                    if self._store.load_df(meta_key) is None:
                        stripped = [
                            {k: v for k, v in e.items() if k not in self._HEAVY_FIELDS}
                            for e in entries
                        ]
                        meta_df = pl.DataFrame(
                            {"results": [json.dumps({"entries": stripped, "main_result_row": row})]}
                        )
                        self._store.save_df(meta_df, meta_key)
                    return entries, row
                except (Exception, KeyError):
                    pass
        return None

    def load_repeats_meta(self, combo_key: str, count: int, start_idx: int = 0) -> list[RepeatResult] | None:
        """Load stripped repeat metadata (no content_bin) using ``:meta`` sidecars.

        Returns ``None`` if any sidecar is missing (old cache — caller must fall back
        to :meth:`load_repeats` to get the full data including restore_graphs bytes).
        """
        results: list[RepeatResult] = []
        for i in range(count):
            key = self.make_repeat_key(combo_key, start_idx + i) + ":meta"
            df = self._store.load_df(key)
            if df is None or df.is_empty():
                return None  # sidecar missing — fall back to full load
            raw = df.get_column("results")[0]
            if not isinstance(raw, str):
                return None
            try:
                payload = json.loads(raw)
                results.append((payload["entries"], payload["main_result_row"]))
            except (Exception, KeyError):
                return None
        return results

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
