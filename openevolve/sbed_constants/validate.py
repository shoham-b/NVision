"""Head-to-head re-measurement of a tuned candidate against the true baseline.

**This step is not optional and its result overrides the optimizer's.** The
number an optimization run reports is, at best, a shrunk estimate computed from
the same data that chose the point -- it is still selected-on, and selection on
noisy data is what produced the two false winners this package exists to avoid
(0.5855 baseline vs 0.5096 and 0.4974 for candidates that had "won" their
searches).

The protocol:

*   **Both arms are re-measured, from scratch, in this run.** A baseline number
    remembered from an earlier session is not a control: machine load, repeat
    counts and the acquisition code itself may all have moved.
*   **Arms alternate** (A, B, A, B, ...) rather than running as two blocks, so
    anything that drifts over the wall-clock of the comparison -- thermal
    throttling, another job starting -- hits both arms equally instead of
    landing entirely on whichever ran second.
*   **The baseline arm is the unmodified ``initial_program.py``**, used
    verbatim, not the renderer's output at default constants. If the renderer
    has a bug, running it on both arms cancels the bug out and hides it.
*   **Evaluations are unseeded.** Common random numbers are a search-time
    variance reduction; validating under them measures the candidate on one
    particular random stream, which is precisely the overfit being tested for.

Read the output as a confidence interval, not as a winner. With an evaluation
sd of ~0.07, ``n`` repeats per arm resolve a difference of roughly
``2.8 * 0.07 / sqrt(n)``: 8 repeats per arm resolve ~0.07, 20 resolve ~0.044,
and detecting a 0.02 improvement needs ~190 per arm. If the interval contains
zero, the honest report is "no measurable effect", not "slightly better".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from .objective import default_grid, evaluate_params
from .space import format_params, repair_params


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for ``mean(b) - mean(a)`` (candidate minus baseline)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(b, b.size, replace=True).mean() - rng.choice(a, a.size, replace=True).mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def head_to_head(
    params: dict[str, float],
    repeats_per_arm: int = 8,
    grid: list[tuple[str, str, int]] | None = None,
    log_path: str | Path | None = None,
) -> dict:
    """Alternate baseline / candidate evaluations and report the difference."""
    params = repair_params(params)
    grid = list(grid) if grid is not None else default_grid()
    log_file = Path(log_path) if log_path is not None else None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    baseline_scores: list[float] = []
    candidate_scores: list[float] = []
    started = time.perf_counter()

    for round_index in range(repeats_per_arm):
        for arm, arm_params in (("baseline", None), ("candidate", params)):
            metrics = evaluate_params(arm_params, grid=grid, crn_seed=None)
            score = float(metrics.get("combined_score", 0.0) or 0.0)
            score = score if np.isfinite(score) else 0.0
            (baseline_scores if arm == "baseline" else candidate_scores).append(score)
            print(f"[round {round_index + 1}/{repeats_per_arm}] {arm:9s} score={score:.4f}")
            if log_file is not None:
                with open(log_file, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps({"round": round_index, "arm": arm, "score": score}) + "\n")

    a = np.array(baseline_scores)
    b = np.array(candidate_scores)
    result: dict = {
        "repeats_per_arm": repeats_per_arm,
        "elapsed_s": time.perf_counter() - started,
        "params": params,
        "baseline_mean": float(a.mean()),
        "baseline_sd": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
        "candidate_mean": float(b.mean()),
        "candidate_sd": float(b.std(ddof=1)) if b.size > 1 else float("nan"),
        "difference": float(b.mean() - a.mean()),
        "baseline_scores": [float(v) for v in a],
        "candidate_scores": [float(v) for v in b],
    }
    if a.size > 1 and b.size > 1:
        t_stat, p_value = stats.ttest_ind(b, a, equal_var=False)
        result["welch_t"] = float(t_stat)
        result["welch_p"] = float(p_value)
        lo, hi = _bootstrap_ci(a, b)
        result["diff_ci95"] = [lo, hi]
        result["verdict"] = (
            "candidate better" if lo > 0 else "baseline better" if hi < 0 else "no measurable difference"
        )
    else:
        result["verdict"] = "too few repeats to conclude anything"
    return result


def print_report(result: dict) -> None:
    print("\n=== head-to-head ===")
    print(f"params           : {format_params(result['params'])}")
    print(f"repeats per arm  : {result['repeats_per_arm']}")
    print(f"baseline         : {result['baseline_mean']:.4f} (sd {result['baseline_sd']:.4f})")
    print(f"candidate        : {result['candidate_mean']:.4f} (sd {result['candidate_sd']:.4f})")
    print(f"difference       : {result['difference']:+.4f}")
    if "diff_ci95" in result:
        lo, hi = result["diff_ci95"]
        print(f"95% CI on diff   : [{lo:+.4f}, {hi:+.4f}]   Welch p = {result['welch_p']:.3f}")
    print(f"verdict          : {result['verdict']}")
    print(f"wall clock       : {result['elapsed_s'] / 60:.1f} min")
