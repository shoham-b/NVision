from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class ScanBatch:
    """Represents a 1-D scan, defining the domain and the ideal signal function."""

    x_min: float
    x_max: float
    signal: Callable[[float], float]
    meta: dict[str, float]
    truth_positions: list[float]


class Locator(ABC):
    """Abstract base class for 1-D peak-finding strategies.

    Batched operation relies on two Polars frames:

    ``history``
        Contains *all* recorded measurements across every repeat with at least the
        columns ``repeat_id`` (int), ``step`` (int), ``x`` (float), and
        ``signal_values`` (float). Additional columns are allowed and preserved.

    ``repeats``
        Stores per-repeat metadata and cached state. The minimum expected columns are
        ``repeat_id`` (int) and ``active`` (bool). Concrete locators may rely on extra
        columns (e.g., domain bounds, prior parameters) populated by the runner.
    """

    @abstractmethod
    def propose_next(
        self,
        history: pl.DataFrame,
        repeats: pl.DataFrame,
        scan: ScanBatch,
    ) -> pl.DataFrame:
        """Return the next proposal for every active repeat.

        Parameters
        ----------
        history:
            Polars DataFrame containing *all* past measurements for every repeat, as
            described above. The frame is empty for newly initialised locators.
        repeats:
            A per-repeat state DataFrame which at minimum exposes ``repeat_id`` and an
            ``active`` boolean column indicating whether the repeat still requires
            another proposal. Additional columns (e.g., RNG seeds, cached parameters)
            may be provided by the runner and can be leveraged by concrete locators.

        Returns
        -------
        pl.DataFrame
            Must contain at least ``repeat_id`` and ``x`` columns. Each row represents
            the next position that should be sampled for the corresponding repeat.
            Implementations may add auxiliary columns (e.g., utility scores) that will
            be persisted alongside the history.
        """
        raise NotImplementedError

    @abstractmethod
    def should_stop(
        self,
        history: pl.DataFrame,
        repeats: pl.DataFrame,
        scan: ScanBatch,
    ) -> pl.DataFrame:
        """Decide which repeats have satisfied their stopping criteria.

        Returns a Polars DataFrame with two columns:

        - ``repeat_id`` (int)
        - ``stop`` (bool)

        Repeats absent from the returned DataFrame are interpreted as ``stop = False``.
        Implementations may rely on the ``repeats`` DataFrame to source cached per-repeat
        quantities (e.g., uncertainty thresholds) when making decisions.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        history: pl.DataFrame,
        repeats: pl.DataFrame,
        scan: ScanBatch,
    ) -> pl.DataFrame:
        """Summarise all repeats once acquisition is complete.

        The result must include one row per ``repeat_id`` and is expected to expose the
        metrics required by downstream reporting (e.g., ``x_hat``, ``abs_err_x``, etc.).
        Additional metadata columns are allowed. Implementations may merge values from
        ``repeats`` (e.g., pre-computed stop reasons) into the returned frame.
        """
        raise NotImplementedError

    def _coerce_inputs(
        self,
        history: pl.DataFrame | Sequence[Mapping[str, float]] | None,
        repeats: pl.DataFrame | Sequence[Mapping[str, float]] | None,
        scan: ScanBatch | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, ScanBatch, bool]:
        """Normalise caller inputs to batched DataFrames.

        Returns a tuple ``(history_df, repeats_df, scan, compat_mode)`` where
        ``compat_mode`` is ``True`` when the caller used the legacy per-repeat API.
        """

        if isinstance(repeats, ScanBatch) and scan is None:
            scan = repeats
            repeats = None

        if scan is None:
            raise ValueError("scan must be provided")

        compat_mode = repeats is None

        if isinstance(history, pl.DataFrame):
            history_df = history.clone()
        else:
            records: Sequence[Mapping[str, float]] = history or []  # type: ignore[assignment]
            history_df = pl.DataFrame(records)

        if "repeat_id" not in history_df.columns:
            history_df = history_df.with_columns(pl.lit(0).cast(pl.Int64).alias("repeat_id"))
        else:
            history_df = history_df.with_columns(pl.col("repeat_id").cast(pl.Int64))

        if "step" not in history_df.columns:
            history_df = history_df.with_row_count("step").with_columns(pl.col("step").cast(pl.Int64))
        else:
            history_df = history_df.with_columns(pl.col("step").cast(pl.Int64))

        if repeats is None:
            repeats_df = pl.DataFrame({"repeat_id": [0], "active": [True]})
        elif isinstance(repeats, pl.DataFrame):
            repeats_df = repeats.clone()
        else:
            repeat_records: Sequence[Mapping[str, float]] = repeats  # type: ignore[assignment]
            repeats_df = pl.DataFrame(repeat_records)

        if "repeat_id" not in repeats_df.columns:
            repeats_df = repeats_df.with_columns(pl.lit(0).cast(pl.Int64).alias("repeat_id"))
        else:
            repeats_df = repeats_df.with_columns(pl.col("repeat_id").cast(pl.Int64))

        if "active" not in repeats_df.columns:
            repeats_df = repeats_df.with_columns(pl.lit(True).alias("active"))
        else:
            repeats_df = repeats_df.with_columns(pl.col("active").cast(pl.Boolean))

        return history_df, repeats_df, scan, compat_mode
