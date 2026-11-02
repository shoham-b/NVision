"""Calibration/convergence study of the lab's own MATLAB Bayesian ESR runs.

Reads the ``convergence_statistics`` folder written by the lab's ``Gather_and_Process_data.m``
(``window<W>MHz_statistics_perRun.csv`` and the two per-step curves per window). There is no
ground truth for real data, so the reference resonance is the median of all converged runs'
final means and "error" is relative to it; across-run spread therefore also contains any real
drift of the resonance between runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl

_WINDOW_RE = re.compile(r"window(\d+)MHz_statistics_perRun\.csv")


def _windows(root: Path) -> list[int]:
    found = sorted(
        int(m.group(1)) for p in root.glob("window*MHz_statistics_perRun.csv") if (m := _WINDOW_RE.fullmatch(p.name))
    )
    if not found:
        raise ValueError(f"No window*MHz_statistics_perRun.csv files in {root}")
    return found


def lab_study_payload(root: Path) -> dict:
    """Per-window summary rows and per-step curves for the UI's Lab study page.

    *root*: the ``convergence_statistics`` directory. Returned ``windows`` rows are in MHz;
    ``curves[window]`` holds arrays over ``step`` (1-based measurement count):
    ``claimed_median/q25/q75`` (the runs' self-reported posterior std) and ``across_run_std``
    (the std of the runs' posterior means at that step, the error they actually made).
    """
    windows = _windows(root)
    runs = pl.concat(
        [
            pl.read_csv(root / f"window{w}MHz_statistics_perRun.csv").with_columns(pl.lit(w).alias("window"))
            for w in windows
        ],
        how="vertical_relaxed",
    )
    n_runs = {w: n for w, n in runs.group_by("window").len().iter_rows()}
    conv = runs.filter(pl.col("converged") == 1)
    ref = float(conv["finalMeanMHz"].median())
    conv = conv.with_columns(
        ((pl.col("finalMeanMHz") - ref).abs() / pl.col("finalStdMHz")).alias("z"),
        ((pl.col("credibleIntervalMHz_1") <= ref) & (ref <= pl.col("credibleIntervalMHz_2")))
        .cast(pl.Float64)
        .alias("ci_hit"),
    )
    rows = []
    for (w,), g in conv.group_by("window", maintain_order=True):
        claimed = float(g["finalStdMHz"].median())
        spread = float(g["finalMeanMHz"].std())
        rows.append(
            {
                "window_mhz": w,
                "n_runs": n_runs[w],
                "n_converged": g.height,
                "steps_median": float(g["stepsToConvergence"].median()),
                "steps_q25": float(g["stepsToConvergence"].quantile(0.25)),
                "steps_q75": float(g["stepsToConvergence"].quantile(0.75)),
                "claimed_sigma_mhz": claimed,
                "across_run_std_mhz": spread,
                "overconfidence": spread / claimed,
                "within_1sigma": float((g["z"] <= 1).mean()),
                "within_2sigma": float((g["z"] <= 2).mean()),
                "ci95_hit": float(g["ci_hit"].mean()),
            }
        )
    rows.sort(key=lambda r: r["window_mhz"])

    curves = {}
    for w in windows:
        claimed = pl.read_csv(root / f"window{w}MHz_self_reported_std_vs_step.csv")
        actual = pl.read_csv(root / f"window{w}MHz_resonance_std_vs_step_retained.csv")
        curves[str(w)] = {
            "step": claimed["measurementStep"].to_list(),
            "claimed_median": claimed["medianPosteriorStdMHz"].to_list(),
            "claimed_q25": claimed["q25PosteriorStdMHz"].to_list(),
            "claimed_q75": claimed["q75PosteriorStdMHz"].to_list(),
            "n_contributing": claimed["numContributingRuns"].to_list(),
            "actual_step": actual["measurementStep"].to_list(),
            "across_run_std": actual["acrossRunStdMHz"].to_list(),
            "actual_n_contributing": actual["numContributingRuns"].to_list(),
            "prior_sigma_mhz": w / float(np.sqrt(12.0)),
        }
    return {"reference_mhz": ref, "windows": rows, "curves": curves}
