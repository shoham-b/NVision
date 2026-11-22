"""Per-locator summary of a finished run's results table."""

from __future__ import annotations

import polars as pl

# Runs that produced data. A run that merely hit its step budget still has a trajectory and counts.
_USABLE_FAILURES = (None, "max_steps")


def usable_runs(df: pl.DataFrame) -> pl.DataFrame:
    """Rows of the locator-results table that carry data (drops timeouts and runs with no measurements)."""
    if "failure_reason" in df.columns:
        df = df.filter(pl.col("failure_reason").is_null() | pl.col("failure_reason").is_in(["max_steps"]))
    return df.filter(pl.col("measurements") > 0) if "measurements" in df.columns else df


def compare_locators(df: pl.DataFrame, *, group_by: tuple[str, ...] = ()) -> pl.DataFrame:
    """One row per strategy (per ``group_by`` value too) comparing every locator on the same footing.

    Columns:

    * ``runs``: usable repeats; ``measurements``: median measurements taken.
    * ``err_median`` / ``err_p90``: |final primary-parameter error| (Hz); ``unc_median``: reported uncertainty;
      ``cover_2sigma``: share of runs with error within 2x their reported uncertainty.
    * ``detected``: share of runs whose dip test fired (the end of the search); ``detect_step``: its median step over
      those runs. A step few runs reach is a finding about the locator, not missing data.

    The final error mostly reflects the stop target, so read ``cover_2sigma`` (is the claimed sigma honest?) before
    the error columns (``docs/evaluation_metrics.md``).
    """
    data = usable_runs(df)
    keys = [*group_by, "strategy"]

    def reached(col: str) -> pl.Expr:
        return pl.col(col).is_not_null().mean() if col in data.columns else pl.lit(None)

    def med(col: str) -> pl.Expr:
        return pl.col(col).median() if col in data.columns else pl.lit(None)

    exprs = [
        pl.len().alias("runs"),
        med("measurements").alias("measurements"),
        pl.col("final_err_primary").median().alias("err_median"),
        pl.col("final_err_primary").quantile(0.9).alias("err_p90"),
        med("uncert").alias("unc_median"),
        (pl.col("final_err_primary") <= 2.0 * pl.col("uncert")).mean().alias("cover_2sigma"),
        reached("dip_detected_step").alias("detected"),
        med("dip_detected_step").alias("detect_step"),
    ]
    return data.group_by(keys).agg(exprs).sort(keys)
