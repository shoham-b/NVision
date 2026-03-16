"""Adapter to make v2 stateless locators work with the existing polars-based runner."""

from dataclasses import dataclass

import pandas as pd
import polars as pl

from nvision.sim.locs.base import Locator as OldLocator, ScanBatch
from nvision.sim.locs.v2.base import Locator as V2Locator


@dataclass
class V2LocatorAdapter(OldLocator):
    """Adapter that wraps a v2 stateless locator to work with the existing polars infrastructure.

    This allows incremental migration - v2 locators can be used in the existing codebase
    without requiring a full rewrite of the runner.

    The adapter:
    - Converts polars DataFrames to pandas for the v2 locator
    - Handles the batched repeat_id interface by running each repeat independently
    - Returns results in the expected polars format
    """

    v2_locator: V2Locator

    def propose_next(
        self,
        history: pl.DataFrame,
        repeats: pl.DataFrame,
        scan: ScanBatch,
    ) -> pl.DataFrame:
        """Propose next measurement for each active repeat.

        Parameters
        ----------
        history : pl.DataFrame
            Polars DataFrame with columns ['repeat_id', 'x', 'signal_values']
        repeats : pl.DataFrame
            Polars DataFrame with columns ['repeat_id', 'active']
        scan : ScanBatch
            Scan configuration (provides x_min, x_max for scaling)

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ['repeat_id', 'x'] containing proposals
        """
        active = repeats.filter(pl.col("active"))
        if active.is_empty():
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})

        proposals = []

        for row in active.iter_rows(named=True):
            repeat_id = row["repeat_id"]

            # Get history for this repeat
            repeat_history = history.filter(pl.col("repeat_id") == repeat_id)

            # Convert to pandas DataFrame with v2 schema
            if repeat_history.is_empty():
                history_df = pd.DataFrame(columns=["x", "signal_value"])
            else:
                history_df = pd.DataFrame(
                    {
                        "x": repeat_history.get_column("x").to_list(),
                        "signal_value": repeat_history.get_column("signal_values").to_list(),
                    }
                )

            # Get proposal from v2 locator
            # Note: v2 locators work in normalized [0, 1] space
            x_normalized = self.v2_locator.next(history_df)

            # Scale to scan domain
            x_scaled = scan.x_min + x_normalized * (scan.x_max - scan.x_min)

            proposals.append({"repeat_id": repeat_id, "x": x_scaled})

        if not proposals:
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})

        return pl.DataFrame(proposals)

    def should_stop(
        self,
        history: pl.DataFrame,
        repeats: pl.DataFrame,
        scan: ScanBatch,
    ) -> pl.DataFrame:
        """Determine which repeats should stop.

        Parameters
        ----------
        history : pl.DataFrame
            Polars DataFrame with columns ['repeat_id', 'x', 'signal_values']
        repeats : pl.DataFrame
            Polars DataFrame with columns ['repeat_id']
        scan : ScanBatch
            Scan configuration

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ['repeat_id', 'stop']
        """
        stop_results = []

        for row in repeats.iter_rows(named=True):
            repeat_id = row["repeat_id"]

            # Get history for this repeat
            repeat_history = history.filter(pl.col("repeat_id") == repeat_id)

            # Convert to pandas
            if repeat_history.is_empty():
                history_df = pd.DataFrame(columns=["x", "signal_value"])
            else:
                history_df = pd.DataFrame(
                    {
                        "x": repeat_history.get_column("x").to_list(),
                        "signal_value": repeat_history.get_column("signal_values").to_list(),
                    }
                )

            # Check if done
            should_stop = self.v2_locator.done(history_df)

            stop_results.append({"repeat_id": repeat_id, "stop": should_stop})

        return pl.DataFrame(stop_results)

    def finalize(
        self,
        history: pl.DataFrame,
        repeats: pl.DataFrame,
        scan: ScanBatch,
    ) -> pl.DataFrame:
        """Extract final results for each repeat.

        Parameters
        ----------
        history : pl.DataFrame
            Polars DataFrame with columns ['repeat_id', 'x', 'signal_values']
        repeats : pl.DataFrame
            Polars DataFrame with columns ['repeat_id']
        scan : ScanBatch
            Scan configuration

        Returns
        -------
        pl.DataFrame
            DataFrame with repeat_id and result columns
        """
        finalize_results = []

        for row in repeats.iter_rows(named=True):
            repeat_id = row["repeat_id"]

            # Get history for this repeat
            repeat_history = history.filter(pl.col("repeat_id") == repeat_id)

            # Convert to pandas
            if repeat_history.is_empty():
                history_df = pd.DataFrame(columns=["x", "signal_value"])
            else:
                history_df = pd.DataFrame(
                    {
                        "x": repeat_history.get_column("x").to_list(),
                        "signal_value": repeat_history.get_column("signal_values").to_list(),
                    }
                )

            # Get results from v2 locator
            result = self.v2_locator.result(history_df)

            # Scale x values back to scan domain
            result_scaled = {"repeat_id": repeat_id}
            for key, value in result.items():
                if "x" in key.lower() and not pd.isna(value):
                    # Scale normalized x back to scan domain
                    result_scaled[key] = scan.x_min + value * (scan.x_max - scan.x_min)
                else:
                    result_scaled[key] = value

            # Add standard columns expected by the system
            result_scaled.setdefault("measurements", len(history_df))
            result_scaled.setdefault("n_peaks", 1.0)
            result_scaled.setdefault("uncert", 0.0)

            # Map peak_x to x1_hat if needed
            if "peak_x" in result_scaled and "x1_hat" not in result_scaled:
                result_scaled["x1_hat"] = result_scaled["peak_x"]

            finalize_results.append(result_scaled)

        return pl.DataFrame(finalize_results)
