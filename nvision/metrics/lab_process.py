"""Turn a folder of the lab's full-budget MATLAB Bayesian ESR runs into ``convergence_statistics`` CSVs.

The lab's own ``Gather_and_Process_data.m`` only exists for the run-until-converged experiment. Full-budget
runs (``noConvergenceStop``) measure every step, so "convergence" is replayed from the saved history:
the first step ``>= minimumStopSteps`` whose posterior std is ``<= targetStdMHz`` is where the precision
stop would have fired. Output matches what :func:`nvision.metrics.lab_study.lab_study_payload` reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import scipy.io as sio
from scipy.io.matlab import MatReadError

_Z95 = 1.959963984540054


def _load_run(path: Path) -> dict | None:
    """One run's history, or None when the file holds no Bayesian ESR run.

    ``mean``/``std`` have shape (n_steps,) in MHz: the posterior after each measurement.
    """
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["myStruct"])
    esr = getattr(mat.get("myStruct"), "BayesianESR", None)
    if esr is None or not hasattr(esr, "state"):
        return None
    cfg, state = esr.config, esr.state
    history = np.asarray(state.history, dtype=float)
    lo, hi = np.asarray(cfg.scanRangeMHz, dtype=float)
    stopped_on_precision = not bool(esr.noConvergenceStop)
    # Older MATLAB runs did not record the focus candidates; those get no window_focus rows.
    candidates = None
    if hasattr(state, "focusCandidatesHistoryMHz"):
        candidates = [
            np.asarray(c, dtype=float).reshape(-1, 2) if np.size(c) else np.empty((0, 2))
            for c in state.focusCandidatesHistoryMHz
        ]
    return {
        "window": round(float(hi - lo)),
        "focus_candidates": candidates,  # per step: (n_candidates, 2) rows of [lo, hi] in MHz
        "focus_window": np.asarray(state.focusWindowMHz, dtype=float) if hasattr(state, "focusWindowMHz") else None,
        "focus_locked": bool(state.focusLocked) if hasattr(state, "focusLocked") else None,
        "mean": history[:, 0],
        "std": history[:, 1],
        # Runs that really stopped on precision keep their own stop; full-budget runs are replayed.
        "real_stop": None
        if not stopped_on_precision
        else {
            "steps": int(state.stepCount),
            "converged": str(state.stopReason) == "precision",
            "mean": float(state.meanMHz),
            "std": float(state.stdMHz),
            "ci": [float(x) for x in np.asarray(state.credibleIntervalMHz, dtype=float)],
        },
        "target_std": float(cfg.targetStdMHz),
        "min_stop_steps": int(cfg.minimumStopSteps),
    }


def write_statistics(runs: list[dict], out_dir: Path) -> Path:
    """Write ``window<W>MHz_*`` CSVs for every Bayesian ESR run in *folder*; return the output dir.

    Per window: ``statistics_perRun`` (replayed precision stop), ``self_reported_std_vs_step`` (quartiles
    over runs of the posterior std) and ``resonance_std_vs_step_retained`` (std over runs of the posterior
    mean). The credible interval of a replayed stop is mean +/- 1.96 std (the runs save only the final one).
    """
    if (out_dir / "window_summary.csv").exists():
        raise ValueError(f"{out_dir} was written by the lab's own script; not overwriting it")
    out_dir.mkdir(parents=True, exist_ok=True)
    converged_means: list[float] = []
    for window in sorted({r["window"] for r in runs}):
        group = [r for r in runs if r["window"] == window]
        rows = []
        for i, r in enumerate(group, start=1):
            if r["real_stop"] is not None:
                s = r["real_stop"]
                rows.append(
                    {
                        "runNumber": i,
                        "converged": int(s["converged"]),
                        "stepsToConvergence": s["steps"],
                        "finalMeanMHz": s["mean"],
                        "finalStdMHz": s["std"],
                        "credibleIntervalMHz_1": s["ci"][0],
                        "credibleIntervalMHz_2": s["ci"][1],
                    }
                )
                continue
            ok = np.flatnonzero((r["std"] <= r["target_std"]) & (np.arange(len(r["std"])) + 1 >= r["min_stop_steps"]))
            k = int(ok[0]) if ok.size else len(r["std"]) - 1
            rows.append(
                {
                    "runNumber": i,
                    "converged": int(ok.size > 0),
                    "stepsToConvergence": k + 1,
                    "finalMeanMHz": r["mean"][k],
                    "finalStdMHz": r["std"][k],
                    "credibleIntervalMHz_1": r["mean"][k] - _Z95 * r["std"][k],
                    "credibleIntervalMHz_2": r["mean"][k] + _Z95 * r["std"][k],
                }
            )
        pl.DataFrame(rows).write_csv(out_dir / f"window{window}MHz_statistics_perRun.csv")
        converged_means += [r["finalMeanMHz"] for r in rows if r["converged"]]

        n_steps = min(len(r["std"]) for r in group)
        means = np.stack([r["mean"][:n_steps] for r in group])  # (n_runs, n_steps)
        stds = np.stack([r["std"][:n_steps] for r in group])
        steps = np.arange(1, n_steps + 1)
        pl.DataFrame(
            {
                "measurementStep": steps,
                "numContributingRuns": len(group),
                "medianPosteriorStdMHz": np.median(stds, axis=0),
                "q25PosteriorStdMHz": np.quantile(stds, 0.25, axis=0),
                "q75PosteriorStdMHz": np.quantile(stds, 0.75, axis=0),
            }
        ).write_csv(out_dir / f"window{window}MHz_self_reported_std_vs_step.csv")
        pl.DataFrame(
            {
                "measurementStep": steps,
                "numContributingRuns": len(group),
                "acrossRunStdMHz": np.std(means, axis=0),  # divisor N, as the lab's script
            }
        ).write_csv(out_dir / f"window{window}MHz_resonance_std_vs_step_retained.csv")
    _write_window_focus(runs, converged_means, out_dir)
    return out_dir


def _covers(candidates: np.ndarray, reference_mhz: float) -> bool:
    """True when one of *candidates* (shape: (n_candidates, 2), rows [lo, hi] in MHz) contains the reference."""
    return bool(np.any((candidates[:, 0] <= reference_mhz) & (reference_mhz <= candidates[:, 1])))


def _write_window_focus(runs: list[dict], converged_means: list[float], out_dir: Path) -> None:
    """``window_focus.csv``: per run, when the lab's focus candidates appeared and when they covered the resonance.

    The reference resonance is the median of the converged runs' final means, as in the lab study. Steps are
    1-based; empty means never. ``coverStep``: first step some candidate contains the reference;
    ``stableStep``: first step from which one does at every later step. ``finalLocked``/``finalCovers``/
    ``finalWidthMHz`` describe the saved final focus window. Always written (zero rows when the runs carry no
    candidates) so a processed folder can be told apart from one processed before this file existed.
    """
    ref = float(np.median(converged_means)) if converged_means else float("nan")
    rows = []
    for window in sorted({r["window"] for r in runs}):
        for i, r in enumerate((r for r in runs if r["window"] == window), start=1):
            cands = r["focus_candidates"]
            if cands is None or not np.isfinite(ref):
                continue
            present = np.array([len(c) > 0 for c in cands])
            covered = np.array([_covers(c, ref) if len(c) else False for c in cands])
            lost = np.flatnonzero(~covered)
            fw = r["focus_window"]
            rows.append(
                {
                    "window": window,
                    "runNumber": i,
                    "foundStep": int(np.flatnonzero(present)[0]) + 1 if present.any() else None,
                    "coverStep": int(np.flatnonzero(covered)[0]) + 1 if covered.any() else None,
                    "stableStep": (int(lost[-1]) + 2 if lost.size else 1) if covered[-1] else None,
                    "finalLocked": int(r["focus_locked"]),
                    "finalCovers": int(fw[0] <= ref <= fw[1]),
                    "finalWidthMHz": float(fw[1] - fw[0]),
                }
            )
    schema = {
        "window": pl.Int64, "runNumber": pl.Int64, "foundStep": pl.Int64, "coverStep": pl.Int64,
        "stableStep": pl.Int64, "finalLocked": pl.Int64, "finalCovers": pl.Int64, "finalWidthMHz": pl.Float64,
    }  # fmt: skip
    pl.DataFrame(rows, schema=schema).write_csv(out_dir / "window_focus.csv")


def load_bayesian_runs(files: list[Path]) -> tuple[dict[Path, list[dict]], list[Path]]:
    """Split *files* into Bayesian ESR runs grouped by their folder, and the files that are not such runs.

    A file that cannot be read as a MATLAB file is "not a Bayesian run" too, so the caller decides what to do with it.
    """
    runs: dict[Path, list[dict]] = {}
    others: list[Path] = []
    for path in sorted(files):
        try:
            run = _load_run(path)
        except (OSError, ValueError, NotImplementedError, MatReadError):
            run = None
        if run is None:
            others.append(path)
        else:
            runs.setdefault(path.parent, []).append(run)
    return runs, others


def process_bayesian_runs(folder: Path, out_dir: Path | None = None) -> Path:
    """Write the lab statistics of the Bayesian ESR runs directly in *folder* (see :func:`write_statistics`)."""
    runs, _others = load_bayesian_runs(sorted(folder.glob("*.mat")))
    if not runs:
        raise ValueError(f"No Bayesian ESR runs in {folder}")
    return write_statistics(runs[folder], out_dir or folder / "convergence_statistics")
