"""Three-stage decomposition of a locator run, detected identically for every locator.

A search for a resonance whose other parameters are already known passes through three stages:

1. **search** -- until the measurements first show a dip (the *focus window* exists).
2. **focus** -- from then until the primary parameter's uncertainty drops below its convergence
   threshold (the *max-information* point: the dip is pinned down and further measurements are
   sensitivity-limited rather than search-limited).
3. **refine** -- from there to the end of the run, measuring at maximum sensitivity.

The boundaries are read off the run's per-step belief snapshots by one function, never from a
locator's own bookkeeping, so a sweep and an SBED run are compared by the same rule:

* ``focus_found_step``: first step whose belief has a significant dip, per
  :meth:`~nvision.belief.smc_marginal.SMCMarginalDistribution.current_dips` (the deterministic
  ``find_dips`` of the measurements and the conjugate noise estimate).
* ``max_info_step``: first step at or after ``focus_found_step`` whose primary-parameter uncertainty
  (``belief.uncertainty()``) is below the parameter's convergence threshold.

Steps are 1-indexed measurement counts. A stage that was never reached has ``None`` metrics -- for
a plain sweep, which does not stop when it has converged, a short or absent *refine* stage is a real
result, not missing data.
"""

from __future__ import annotations

import math
from typing import Any

from nvision.metrics.milestones import convergence_threshold, resolve_primary_param
from nvision.models.observer import RunResult

STAGE_NAMES: tuple[str, str, str] = ("search", "focus", "refine")

# Every key compute_stage_metrics returns (the results-row columns).
STAGE_METRIC_KEYS: tuple[str, ...] = (
    "stage_primary_param",
    "focus_found_step",
    "max_info_step",
    *(f"stage{k}_{field}" for k in (1, 2, 3) for field in ("steps", "unc_start", "unc_end", "unc_ratio")),
)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def compute_stage_metrics(run_result: RunResult) -> dict[str, Any]:
    """Stage boundaries, per-stage durations and primary-parameter uncertainty change for one run.

    Returns a flat dict (all values JSON-serialisable scalars or ``None``):

    * ``stage_primary_param``: the parameter whose uncertainty defines stage 3.
    * ``focus_found_step`` / ``max_info_step``: 1-indexed boundaries (``None`` if not reached).
    * ``stage{1,2,3}_steps``: measurements spent in the stage (``None`` if the stage was not reached).
    * ``stage{1,2,3}_unc_start`` / ``_unc_end`` / ``_unc_ratio``: primary-parameter uncertainty (Hz) at the
      stage's first and last step, and ``end / start`` (< 1 means the stage shrank the uncertainty).
    """
    snapshots = run_result.snapshots
    n_steps = len(snapshots)
    out: dict[str, Any] = {"stage_primary_param": None, "focus_found_step": None, "max_info_step": None}
    for k in (1, 2, 3):
        out[f"stage{k}_steps"] = None
        out[f"stage{k}_unc_start"] = None
        out[f"stage{k}_unc_end"] = None
        out[f"stage{k}_unc_ratio"] = None
    if n_steps == 0:
        return out

    primary = resolve_primary_param(run_result.true_signal.parameter_values().keys()) or "center_freq"
    out["stage_primary_param"] = primary
    threshold = convergence_threshold(run_result, primary)

    unc_trace = [float(s.belief.uncertainty().get(primary, math.nan)) for s in snapshots]

    focus_idx: int | None = next((i for i, s in enumerate(snapshots) if s.belief.current_dips()), None)
    max_info_idx: int | None = None
    if focus_idx is not None:
        max_info_idx = next((i for i in range(focus_idx, n_steps) if unc_trace[i] < threshold), None)

    out["focus_found_step"] = None if focus_idx is None else focus_idx + 1
    out["max_info_step"] = None if max_info_idx is None else max_info_idx + 1

    # (first index, last index) of each stage's endpoints in the uncertainty trace; None = not reached.
    last = n_steps - 1
    spans: list[tuple[int, int] | None] = [
        (0, focus_idx if focus_idx is not None else last),
        None if focus_idx is None else (focus_idx, max_info_idx if max_info_idx is not None else last),
        None if max_info_idx is None else (max_info_idx, last),
    ]
    steps = [
        focus_idx + 1 if focus_idx is not None else n_steps,
        None if focus_idx is None else (max_info_idx if max_info_idx is not None else last) - focus_idx,
        None if max_info_idx is None else last - max_info_idx,
    ]
    for k, (span, n_in_stage) in enumerate(zip(spans, steps, strict=True), start=1):
        if span is None:
            continue
        start, end = unc_trace[span[0]], unc_trace[span[1]]
        out[f"stage{k}_steps"] = n_in_stage
        out[f"stage{k}_unc_start"] = _finite_or_none(start)
        out[f"stage{k}_unc_end"] = _finite_or_none(end)
        out[f"stage{k}_unc_ratio"] = _finite_or_none(end / start) if start > 0 else None
    return out
