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
    return {
        "window": round(float(hi - lo)),
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


def process_bayesian_runs(folder: Path, out_dir: Path | None = None) -> Path:
    """Write ``window<W>MHz_*`` CSVs for every Bayesian ESR run in *folder*; return the output dir.

    Per window: ``statistics_perRun`` (replayed precision stop), ``self_reported_std_vs_step`` (quartiles
    over runs of the posterior std) and ``resonance_std_vs_step_retained`` (std over runs of the posterior
    mean). The credible interval of a replayed stop is mean +/- 1.96 std (the runs save only the final one).
    """
    out_dir = out_dir or folder / "convergence_statistics"
    if (out_dir / "window_summary.csv").exists():
        raise ValueError(f"{out_dir} was written by the lab's own script; not overwriting it")
    runs = [r for p in sorted(folder.glob("*.mat")) if (r := _load_run(p)) is not None]
    if not runs:
        raise ValueError(f"No Bayesian ESR runs in {folder}")
    out_dir.mkdir(exist_ok=True)
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
    return out_dir
