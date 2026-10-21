"""Stage-by-stage comparison of locators from a finished run's results table."""

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
    * ``focus_found`` / ``max_info``: share of runs that reached the stage boundary; ``*_step``: median step.
    * ``stage{1,2,3}_steps``: median measurements in the search / focus / refine stage (reached runs only);
      ``stage{1,2,3}_unc_ratio``: median of (uncertainty at stage end) / (at its start);
      ``stage{1,2,3}_err_end``: median true error (Hz) of the primary parameter after the stage's last measurement;
      ``stage{1,2,3}_sens_eff``: median of the stage's mean sensitivity efficiency (1 = measuring at the most
      informative drive frequency; see ``metrics/stages.py``).

    Every stage column is computed over the runs that reached that stage, so read it with ``focus_found`` and
    ``max_info``: a stage few runs reach is a finding about the locator, not missing data.
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
        reached("focus_found_step").alias("focus_found"),
        med("focus_found_step").alias("focus_step"),
        reached("max_info_step").alias("max_info"),
        med("max_info_step").alias("max_info_step"),
    ]
    for k in (1, 2, 3):
        exprs += [
            med(f"stage{k}_steps").alias(f"stage{k}_steps"),
            med(f"stage{k}_unc_ratio").alias(f"stage{k}_unc_ratio"),
            med(f"stage{k}_err_end").alias(f"stage{k}_err_end"),
            med(f"stage{k}_sens_eff").alias(f"stage{k}_sens_eff"),
        ]
    return data.group_by(keys).agg(exprs).sort(keys)
