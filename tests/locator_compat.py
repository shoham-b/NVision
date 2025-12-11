from __future__ import annotations

import copy
from typing import Any

import polars as pl


class LegacyLocatorShim:
    """Adapts batched locators to the legacy single-repeat interface used in tests."""

    def __init__(self, locator: Any, repeat_id: int = 0):
        self._locator = locator
        self._repeat_id = int(repeat_id)
        self._repeats_df = pl.DataFrame({"repeat_id": [self._repeat_id], "active": [True]})

    def __deepcopy__(self, memo: dict[int, Any]) -> LegacyLocatorShim:
        return LegacyLocatorShim(copy.deepcopy(self._locator, memo), repeat_id=self._repeat_id)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple delegation
        return getattr(self._locator, name)

    @staticmethod
    def _empty_history() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "repeat_id": pl.Series("repeat_id", [], dtype=pl.Int64),
                "step": pl.Series("step", [], dtype=pl.Int64),
                "x": pl.Series("x", [], dtype=pl.Float64),
                "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
            }
        )

    def _prepare_history(self, history: pl.DataFrame | list[dict[str, float]] | None) -> pl.DataFrame:
        if history is None:
            history_df = self._empty_history()
        elif isinstance(history, list):
            history_df = pl.DataFrame(history)
        else:
            history_df = history.clone()

        if history_df.is_empty():
            return self._empty_history()

        if "repeat_id" not in history_df.columns:
            history_df = history_df.with_columns(pl.lit(self._repeat_id).cast(pl.Int64).alias("repeat_id"))
        else:
            history_df = history_df.with_columns(pl.col("repeat_id").cast(pl.Int64))

        if "step" not in history_df.columns:
            history_df = history_df.with_row_index("step")

        return history_df.with_columns(pl.col("step").cast(pl.Int64))

    def _extract_row(self, frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        row = frame.filter(pl.col("repeat_id") == self._repeat_id)
        return row if not row.is_empty() else frame.head(1)

    def propose_next(self, history: pl.DataFrame, repeats_df: pl.DataFrame, scan, **kwargs) -> pl.DataFrame:
        prepared = self._prepare_history(history)
        # Update our internal repeats_df with the one passed in
        if not repeats_df.is_empty():
            self._repeats_df = repeats_df.filter(pl.col("repeat_id") == self._repeat_id)
        # Get proposals from the underlying locator
        proposals = self._locator.propose_next(prepared, self._repeats_df, scan, **kwargs)
        # Extract the row for this repeat
        row = self._extract_row(proposals)
        if row.is_empty() or "x" not in row.columns:
            # Return a default proposal if no valid one was found
            return pl.DataFrame({"repeat_id": [self._repeat_id], "x": [float("nan")]})
        # Return the proposal with the correct structure
        return pl.DataFrame({"repeat_id": [self._repeat_id], "x": [float(row.get_column("x")[0])]})

    def should_stop(self, history: pl.DataFrame, repeats_df: pl.DataFrame, scan, **kwargs) -> pl.DataFrame:
        prepared = self._prepare_history(history)
        # Update our internal repeats_df with the one passed in
        if not repeats_df.is_empty():
            self._repeats_df = repeats_df.filter(pl.col("repeat_id") == self._repeat_id)
        # Pass through any additional keyword arguments
        decisions = self._locator.should_stop(prepared, self._repeats_df, scan, **kwargs)
        row = self._extract_row(decisions)
        stop = bool(row.get_column("stop")[0]) if not row.is_empty() and "stop" in row.columns else False
        # Update the active status in our internal repeats_df
        self._repeats_df = self._repeats_df.with_columns(pl.lit(not stop).alias("active"))
        # Return a DataFrame with the stop decision for this repeat
        return pl.DataFrame({"repeat_id": [self._repeat_id], "stop": [stop]})

    def finalize(self, history: pl.DataFrame, scan) -> dict[str, float]:
        prepared = self._prepare_history(history)
        results = self._locator.finalize(prepared, self._repeats_df, scan)
        row = self._extract_row(results)
        if row.is_empty():
            return {}
        payload = row.drop("repeat_id", strict=False).to_dicts()[0]
        for key, value in list(payload.items()):
            if isinstance(value, int | float) and value is not None:
                payload[key] = float(value)
        return payload
