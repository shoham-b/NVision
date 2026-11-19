"""Calibration/convergence study of the lab's own MATLAB Bayesian ESR runs.

Reads the ``convergence_statistics`` folder written by the lab's ``Gather_and_Process_data.m``
(``window<W>MHz_statistics_perRun.csv`` and the two per-step curves per window). There is no
ground truth for real data, so the reference resonance is the median of all converged runs'
final means and "error" is relative to it; across-run spread therefore also contains any real
drift of the resonance between runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl

from nvision.metrics.lab_process import PER_RUN_HISTORY_SUFFIX
from nvision.metrics.lab_search_fit import fit_search_curve, is_current

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
    (the std of the runs' posterior means at that step: scatter between runs, not an error against a truth).
    ``curves[window]["runs"]`` is every run's own estimate, claimed sigma and error against the reference (see
    :func:`_per_run_histories`), and ``curves[window]["rms_err_mhz"]`` the rms over runs of that error at each step
    (shape (n_steps,) over the steps every run reached); both are None for a folder imported before those histories
    were written.
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
    if conv.is_empty():
        raise ValueError(
            f"No converged run in {root}: the reference resonance (median converged final mean) is undefined"
        )
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
        spread = g["finalMeanMHz"].std()  # None for a single run
        rows.append(
            {
                "window_mhz": w,
                "n_runs": n_runs[w],
                "n_converged": g.height,
                "steps_median": float(g["stepsToConvergence"].median()),
                "steps_q25": float(g["stepsToConvergence"].quantile(0.25)),
                "steps_q75": float(g["stepsToConvergence"].quantile(0.75)),
                "claimed_sigma_mhz": claimed,
                "across_run_std_mhz": None if spread is None else float(spread),
                "overconfidence": None if spread is None else spread / claimed,
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
        # Written at import (nv matlab); folders processed before that have none. A fit from the older model
        # (without the current version) is refitted here from the median claimed sigma.
        fit_path = root / f"window{w}MHz_search_fit.json"
        search_fit = json.loads(fit_path.read_text()) if fit_path.exists() else None
        if search_fit is not None and not is_current(search_fit):
            search_fit = fit_search_curve(np.asarray(claimed["medianPosteriorStdMHz"], dtype=float), float(w))
            search_fit["model"] = search_fit["model"].tolist()
        per_run = _per_run_histories(root, w, runs.filter(pl.col("window") == w), ref)
        curves[str(w)] = {
            "search_fit": search_fit,
            "runs": per_run,
            "rms_err_mhz": None if per_run is None else _rms_error(per_run),
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
    return {"reference_mhz": ref, "windows": rows, "curves": curves, "focus": _focus_rows(root, n_runs)}


def _per_run_histories(root: Path, window: int, per_run_stats: pl.DataFrame, reference_mhz: float) -> list[dict] | None:
    """One window's runs as ``{"run_number", "file", "steps_to_convergence", "converged", "mean_mhz", "std_mhz",
    "err_mhz"}``, or None.

    ``mean_mhz``, ``std_mhz`` and ``err_mhz`` have shape (n_steps,) in MHz, after each measurement: the run's posterior
    mean, its self-reported posterior std, and its error ``mean_mhz - reference_mhz`` against the lab reference
    resonance (runs may differ in length). Rows follow ``statistics_perRun`` (``run_number``); ``file`` is the lab's
    run name. None when the folder has no per-run file (imported before it existed); a file that disagrees with
    ``statistics_perRun`` raises.
    """
    path = root / f"window{window}MHz{PER_RUN_HISTORY_SUFFIX}"
    if not path.exists():
        return None
    runs = json.loads(path.read_text())["runs"]
    numbers = [r["run_number"] for r in runs]
    if numbers != per_run_stats["runNumber"].to_list():
        raise ValueError(
            f"{path.name} has runs {numbers}, but window{window}MHz_statistics_perRun.csv has "
            f"{per_run_stats['runNumber'].to_list()}"
        )
    return [
        {
            "run_number": r["run_number"],
            "file": r["file"],
            "steps_to_convergence": stats["stepsToConvergence"],
            "converged": bool(stats["converged"]),
            "mean_mhz": r["mean_mhz"],
            "std_mhz": r["std_mhz"],
            "err_mhz": (np.asarray(r["mean_mhz"], dtype=float) - reference_mhz).tolist(),
        }
        for r, stats in zip(runs, per_run_stats.iter_rows(named=True), strict=True)
    ]


def _rms_error(per_run: list[dict]) -> list[float]:
    """Rms over runs of ``err_mhz`` at each step, shape (n_steps,) in MHz, over the steps every run reached."""
    n_steps = min(len(r["err_mhz"]) for r in per_run)
    err = np.stack([np.asarray(r["err_mhz"][:n_steps], dtype=float) for r in per_run])  # (n_runs, n_steps)
    return np.sqrt(np.mean(err**2, axis=0)).tolist()


def _median_or_none(col: pl.Series) -> float | None:
    vals = col.drop_nulls()
    return None if vals.is_empty() else float(vals.median())


def _focus_rows(root: Path, n_runs: dict[int, int]) -> list[dict]:
    """Per-window focus-window summary from ``window_focus.csv`` (empty when absent or without candidates).

    Rates are over all of a window's runs. Steps are 1-based medians over the runs that reached the event;
    ``n_*`` count those runs. ``stable_step`` is when a candidate covers the reference at every later step.
    """
    path = root / "window_focus.csv"
    if not path.exists():
        return []
    focus = pl.read_csv(path)
    out = []
    for (w,), g in focus.group_by("window", maintain_order=True):
        n = n_runs[w]

        out.append(
            {
                "window_mhz": w,
                "n_runs": n,
                "found_rate": g["foundStep"].drop_nulls().len() / n,
                "found_step": _median_or_none(g["foundStep"]),
                "cover_rate": g["coverStep"].drop_nulls().len() / n,
                "cover_step": _median_or_none(g["coverStep"]),
                "stable_rate": g["stableStep"].drop_nulls().len() / n,
                "stable_step": _median_or_none(g["stableStep"]),
                "locked_rate": float(g["finalLocked"].sum()) / n,
                "final_covers_rate": float(g["finalCovers"].sum()) / n,
                "final_width_mhz": float(g["finalWidthMHz"].median()),
            }
        )
    out.sort(key=lambda r: r["window_mhz"])
    return out


def discover_labs(roots: list[Path]) -> dict[str, Path]:
    """Map lab name -> processed statistics folder, found under each of *roots*.

    A folder is a lab when it holds ``window*MHz_statistics_perRun.csv`` files; each *root* is
    searched recursively (and may itself be one). The name is the folder's own name, or its
    parent's when it is called ``convergence_statistics``; repeats get a ``-2``, ``-3`` suffix.
    """
    labs: dict[str, Path] = {}
    for root in roots:
        for stats in sorted({p.parent for p in root.rglob("window*MHz_statistics_perRun.csv")}):
            base = stats.parent.name if stats.name == "convergence_statistics" else stats.name
            name, n = base, 2
            while name in labs and labs[name] != stats:
                name, n = f"{base}-{n}", n + 1
            labs[name] = stats
    return labs


def labs_dir(artifacts_dir: Path) -> Path:
    """Where ``nv matlab`` keeps Drive labs under an artifacts directory; ``nv serve`` lists them automatically."""
    return artifacts_dir / "labs"
