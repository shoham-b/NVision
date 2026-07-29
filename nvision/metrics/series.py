"""Per-step (error, uncertainty) series export for the UI Highlights view.

The Highlights view needs anytime performance curves (error vs measurement
count) and threshold-sensitivity survival curves. Both are derived from a
compact, downsampled per-step series stored on each scan manifest entry:

``{"s": [steps], "e": [abs errors], "u": [uncertainties], "tau": threshold}``

Steps are 1-indexed; values are rounded to 4 significant digits to keep the
manifest small. ``tau`` is the effective absolute convergence threshold for
the parameter so the UI can recompute convergence steps at scaled thresholds.
"""

from __future__ import annotations

import math
from typing import Any

from nvision.models.observer import RunResult
from nvision.sim.defaults import NVISION_CONVERGENCE_THRESHOLD, PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS

MAX_SERIES_POINTS = 80


def _round_sig(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    return float(f"{value:.{digits}g}")


def _downsample_indices(n: int, max_points: int) -> list[int]:
    """Pick up to ``max_points`` indices in [0, n): dense head, geometric tail.

    Early steps carry most of the structure of a convergence curve, so the
    first few indices are kept verbatim and the rest are log-spaced. The last
    index is always included.
    """
    if n <= max_points:
        return list(range(n))
    head = max(1, min(10, max_points // 4))
    indices = set(range(head))
    remaining = max_points - len(indices) - 1
    ratio = (n - 1) / head
    for k in range(remaining):
        t = (k + 1) / (remaining + 1)
        indices.add(round(head * ratio**t))
    indices.add(n - 1)
    return sorted(i for i in indices if 0 <= i < n)


def effective_convergence_threshold(
    run_result: RunResult,
    param: str = "frequency",
    relative_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
) -> float | None:
    """Absolute threshold used for ``param`` convergence (mirrors detect_milestone)."""
    absolute = PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS.get(param)
    if absolute is not None:
        return absolute
    if run_result.snapshots:
        bounds = run_result.snapshots[0].belief.physical_param_bounds.get(param)
        if bounds:
            lo, hi = bounds
            return relative_threshold * (hi - lo)
    return None


def _resolve_series_param(run_result: RunResult, param: str) -> str:
    """Fall back to the model's splitting parameter when ``param`` isn't actually free.

    ``frequency`` is fixed by default (``NVCenterVoigtModel(with_fixed_frequency=True)``
    and friends), so it never appears in a snapshot's belief estimates -- the Zeeman/
    hyperfine split is what's actually being localized in that case. Mirrors
    ``milestones.default_fc_param``, which the per-repeat metrics already use for the
    same reason.
    """
    try:
        if param in run_result.snapshots[0].belief.estimates():
            return param
    except Exception:
        pass
    from nvision.metrics.milestones import default_fc_param

    return default_fc_param(run_result)


def extract_step_series(
    run_result: RunResult | None,
    param: str = "frequency",
    max_points: int = MAX_SERIES_POINTS,
) -> dict[str, Any] | None:
    """Build the compact per-step series for one run, or None when unavailable."""
    if run_result is None or not run_result.snapshots:
        return None
    param = _resolve_series_param(run_result, param)
    try:
        true_value = float(run_result.true_signal.get_param_value(param))
    except Exception:
        return None

    steps: list[int] = []
    errs: list[float | None] = []
    uncerts: list[float | None] = []
    for i, snapshot in enumerate(run_result.snapshots):
        try:
            est = snapshot.belief.estimates().get(param)
            unc = snapshot.belief.reported_uncertainty().get(param)
        except Exception:
            continue
        if est is None and unc is None:
            continue
        steps.append(i + 1)
        errs.append(abs(float(est) - true_value) if est is not None and math.isfinite(est) else None)
        uncerts.append(float(unc) if unc is not None and math.isfinite(unc) else None)

    if not steps:
        return None

    keep = _downsample_indices(len(steps), max_points)
    series: dict[str, Any] = {
        "s": [steps[i] for i in keep],
        "e": [_round_sig(errs[i]) for i in keep],
        "u": [_round_sig(uncerts[i]) for i in keep],
    }
    tau = effective_convergence_threshold(run_result, param)
    if tau is not None:
        series["tau"] = _round_sig(tau)
    return series
