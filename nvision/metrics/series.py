"""Per-step series of a run, stored with its scan entry and read by the article views.

``{"step": [steps], "error": [|estimate - truth|], "claimed_sigma": [claimed sigma], "sigma_target": stop target}``

Steps are 1-indexed; values other than ``"focus_windows"`` are rounded to 4 significant digits to keep the cache
small. ``sigma_target`` is the effective absolute convergence threshold of the parameter (the stop target).

An optional ``"focus_windows"`` key carries the per-step focus-window-candidate history (physical Hz, unrounded: 4
significant digits of a ~1.5 GHz edge is 1 MHz, a large part of a section a few MHz wide), aligned with
``"step"``: each entry is ``None`` (no live candidate window yet) or a list of ``[lo, hi]`` pairs, more than one while
several dip candidates compete. It drives the timeline's focus-window playback (``static/plotly-utils.js``). The
posterior series (``z_score``, ``pit``, ``negentropy``, ``far_mass``, ``posterior_fisher``, ``kde_bandwidth``,
``far_placed``; :mod:`nvision.metrics.posterior`) are added by :func:`attach_posterior_series`.
"""

from __future__ import annotations

import math
from contextlib import suppress
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
    param: str = "zero_field_freq",
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

    ``zero_field_freq`` is fixed by default (``NVCenterVoigtModel(with_fixed_zero_field_freq=True)``
    and friends), so it never appears in a snapshot's belief estimates -- the Zeeman/
    hyperfine shift is what's actually being localized in that case. Mirrors
    ``milestones.default_shift_param``, which the per-repeat metrics already use for the
    same reason.
    """
    try:
        if param in run_result.snapshots[0].belief.estimates():
            return param
    except Exception:
        pass
    from nvision.metrics.milestones import default_shift_param

    return default_shift_param(run_result)


def extract_step_series(
    run_result: RunResult | None,
    param: str = "zero_field_freq",
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
    crlbs: list[float | None] = []
    windows: list[list[tuple[float, float]] | None] = []
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
        crlb_val = None
        with suppress(Exception):
            belief = snapshot.belief
            val = belief.crlb_zero_field_freq() if param == "zero_field_freq" else belief.crlb_per_param().get(param)
            if val is not None and math.isfinite(val) and val > 0:
                crlb_val = float(val)
        crlbs.append(crlb_val)
        windows.append(getattr(snapshot, "focus_window_candidates", None))

    if not steps:
        return None

    # A locator whose reported estimate is a fit (sweeps: ``finalize()``'s least-squares fit, which
    # ``final_err_primary`` scores) ends the series on that estimate, not on its belief, so the curve's final
    # point is the run's reported accuracy. The fit exists only after the last measurement; earlier steps stay
    # the belief, the only per-step estimate a sweep has.
    fit_est = (run_result.fit_mode_estimates or {}).get(param)
    if fit_est is not None:
        if steps[-1] != len(run_result.snapshots):
            raise ValueError("The run has a fit estimate but no final belief snapshot to put it on.")
        errs[-1] = abs(float(fit_est) - true_value) if math.isfinite(fit_est) else None
        fit_unc = (run_result.fit_mode_uncertainties or {}).get(param)
        if fit_unc is not None and math.isfinite(fit_unc):  # NaN (no covariance): keep the belief's, as convert.py
            uncerts[-1] = float(fit_unc)

    keep = _downsample_indices(len(steps), max_points)
    series: dict[str, Any] = {
        "step": [steps[i] for i in keep],
        "error": [_round_sig(errs[i]) for i in keep],
        "claimed_sigma": [_round_sig(uncerts[i]) for i in keep],
    }
    if any(c is not None for c in crlbs):
        series["crlb"] = [_round_sig(crlbs[i]) for i in keep]
    if any(windows):
        series["focus_windows"] = [
            [[float(lo), float(hi)] for lo, hi in windows[i]] if windows[i] else None for i in keep
        ]
    tau = effective_convergence_threshold(run_result, param)
    if tau is not None:
        series["sigma_target"] = _round_sig(tau)
    return series


def attach_posterior_series(series: dict[str, Any], posterior: dict[str, Any]) -> None:
    """Add the run-time posterior per-step series (``posterior.POSTERIOR_SERIES_KEYS``) to a step series.

    ``series``: the dict :func:`extract_step_series` built; ``posterior``: ``compute_posterior_metrics`` of the same
    run. Values are taken at the steps of ``series["step"]`` (1-indexed). Raises ``ValueError`` if the posterior was
    not built for those steps (the density series exist only at the steps the stored series keeps), so a mismatch
    cannot silently drop or misalign a series.
    """
    if not posterior["computed"]:  # no primary parameter or no particle cloud (e.g. a real-data run): nothing to add
        return
    steps0 = [step - 1 for step in series["step"]]
    if posterior["max_points"] != MAX_SERIES_POINTS or max(steps0) >= len(posterior["series"]["z_score"]):
        raise ValueError("The posterior series were not built for this run's steps / max_points.")
    if not set(steps0) <= set(posterior["density_steps"]):
        raise ValueError("The stored steps are not the steps the posterior density was computed at.")
    for key, values in posterior["series"].items():
        if any(v is not None for v in values):
            series[key] = [_round_sig(values[t]) for t in steps0]
