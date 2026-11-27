"""One locator in one use case -> the article-claim numbers, from the rows ``/api/scan-fields`` serves.

Glue between the cached per-repeat rows (stored scalars + the downsampled per-step ``series``) and the pure functions of
:mod:`nvision.metrics.claims` and :mod:`nvision.metrics.refine_rates`. Read-only. A number the stored data cannot
give is
``None`` with the reason in ``why``; nothing is substituted.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nvision.metrics import claims
from nvision.metrics.detection_aligned import (
    gaussian_onset,
    localize_ratio,
    localize_ratio_ci,
    moving_line_bound,
)
from nvision.metrics.refine_rates import InverseMse, inverse_mse_curve, late_window

COVERAGE_STEPS = (30, 100, 300, 1000)
Z_KEY = "z_score"
PIT_KEY = "pit"
# The stored series have at most 80 points per run; a denser grid lets each half of the fit window hold the 5 points
# the line fit needs (it adds no information: values between stored samples are interpolated).
GRID_POINTS = 200


def _med(values: list[float | None]) -> float | None:
    v = [x for x in values if x is not None and math.isfinite(x)]
    return float(np.median(v)) if v else None


def _gated_dict(g: claims.GatedFit) -> dict[str, Any]:
    f = g.fit
    return {
        "rate": f.rate if f else None,
        "search_offset": f.search_offset if f and math.isfinite(f.search_offset) else None,
        "r2": f.r2 if f else None,
        "window": [f.n_lo, f.n_hi] if f else None,
        "rate_ci": g.rate_ci,
        "search_offset_ci": g.search_offset_ci,
        "rate_halves": g.rate_halves,
        "rate_halves_ci": g.rate_halves_ci,
        "linear": g.linear,
        "why": g.why,
    }


def _gaussian_onset(series: dict[str, Any]) -> float:
    """A run's Gaussian onset (step) from its stored series ``s`` (steps) and ``g`` (negentropy); NaN if never.

    ``series["step"]``, ``series["negentropy"]``: length m. Raises when ``g`` is missing: such a run predates the stored
    negentropy and must be re-run, not silently counted as never Gaussian.
    """
    g = series.get("negentropy")
    if not g:
        raise ValueError("run has no negentropy series 'negentropy': re-run it (nv render cannot fill it)")
    s = np.asarray(series["step"], dtype=np.float64)
    gv = np.array([math.nan if x is None else x for x in g], dtype=np.float64)
    if gv.shape != s.shape:
        raise ValueError(f"series 'step' and 'negentropy' differ in length: {s.shape} vs {gv.shape}")
    onset = gaussian_onset(s, gv)
    return math.nan if onset is None else float(onset)


VAN_TREES_REPORT_STEPS = (1, 10, 30, 100, 300, 1000)
"""Measurement counts at which ``van_trees_from_start`` reports rho."""


def tracking_prior_std_phys(row: dict[str, Any]) -> float:
    """Std (Hz) of the honest Gaussian prior a tracking run's truth was drawn around: the generator's
    ``bounds['_priors']['zeeman_shift'][1]`` (``tracking_prior_std_fwhm`` x FWHM), stored in ``true_params``.

    Raises if the row stores no Gaussian ``zeeman_shift`` prior, so a tracking use case never falls back to another
    spread.
    """
    priors = ((row.get("true_params") or {}).get("bounds") or {}).get("_priors") or {}
    entry = priors.get("zeeman_shift")
    if entry is None or isinstance(entry[0], str) or not float(entry[1]) > 0.0:
        raise ValueError(f"a tracking run stores no Gaussian zeeman_shift prior in true_params, got {entry!r}")
    return float(entry[1])


def _walk_step_phys(cmp: list[dict[str, Any]]) -> float:
    """The use case's line-walk step (Hz) from the runs' stored ``line_walk_step_phys``; raises if any run lacks it
    or they disagree (a bound for a still line must never be applied to moving runs)."""
    values = [c.get("line_walk_step_phys") for c in cmp]
    if any(v is None for v in values):
        raise ValueError("a run has no stored line_walk_step_phys (recorded before the moving line: re-run)")
    if not math.isclose(min(values), max(values), rel_tol=1e-9, abs_tol=0.0):
        raise ValueError(f"runs of one use case disagree on line_walk_step_phys: {sorted(set(values))}")
    return float(values[0])


def van_trees_from_start(
    grid: np.ndarray, err: np.ndarray, sigma0_phys: float, i_max: float, walk_step_phys: float
) -> dict[str, Any]:
    """Tracking: rho(n) = B(n) / MSE(n), the Bayesian bound from the first measurement with J0 = 1 / sigma_0^2.

    ``B(n)`` is van Trees ``(J0 + n I_max)^-1`` for a still line, and its recursion for a line that takes a
    random-walk step of std ``walk_step_phys`` (Hz) between measurements (:func:`moving_line_bound`).

    ``grid``: shape (n_grid,), measurement counts n. ``err``: shape (n_runs, n_grid), true error (Hz) of the primary
    parameter after n measurements. ``sigma0_phys``: std (Hz) of the truth's Gaussian prior. ``i_max``: Hz^-2 per
    measurement. rho <= 1 at every n for any estimator (the window's edges truncate the prior of a truth within a few
    sigma_0 of them, which this ignores). ``violated``: the bootstrap 2.5 % bound of rho exceeds 1 at some n.
    """
    if not sigma0_phys > 0.0 or not i_max > 0.0:
        raise ValueError(f"sigma0_phys and i_max must be positive, got {sigma0_phys}, {i_max}")
    if err.shape != (err.shape[0], grid.size):
        raise ValueError(f"err {err.shape} does not match the grid {grid.shape}")
    j0 = np.array([1.0 / sigma0_phys**2])
    bound = np.broadcast_to(moving_line_bound(j0, grid, i_max, walk_step_phys, first_step_moves=False)[0], err.shape)
    rho = localize_ratio(err, bound)
    lo, hi = localize_ratio_ci(err, bound)
    at = {str(k): float(rho[int(np.argmin(np.abs(grid - k)))]) for k in VAN_TREES_REPORT_STEPS if k <= grid[-1]}
    i = int(np.argmax(rho))
    return {
        "n": grid.tolist(),
        "rho": rho.tolist(),
        "bound_sigma": np.sqrt(bound[0]).tolist(),  # shape: (n_grid,), Hz: the curve the page draws
        "rho_at": at,
        "rho_max": float(rho[i]),
        "rho_max_n": float(grid[i]),
        "rho_max_ci": [float(lo[i]), float(hi[i])],
        "violated": bool(np.any(lo > 1.0)),
    }


def analyse_use_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The claim numbers of one locator's repeats in one use case.

    ``rows``: items of ``/api/scan-fields`` (``compare`` = stored scalars, ``series`` = ``s`` (stored step), ``e`` (true
    error, Hz), ``u`` (claimed sigma, Hz), optionally ``z``; ``true_params``). Runs without a series are left out.
    """
    runs = [r for r in rows if r.get("series") and all(r["series"].get(k) for k in ("step", "error", "claimed_sigma"))]
    if len(runs) < claims.MIN_RUNS:
        return {
            "unavailable": f"{len(runs)} runs with a step/error/claimed_sigma series (need {claims.MIN_RUNS}; rows "
            "recorded before the 2026-11-24 rename store the old names: re-run)",
            "repeats": len(runs),
        }
    cmp = [r.get("compare") or {} for r in runs]
    if any(c.get("line_walk_step_phys") is None for c in cmp):
        # Rows from before the moving line: which bound applies is unknown, so none is drawn.
        return {"unavailable": "line_walk_step_phys missing: re-run", "repeats": len(runs)}
    imax, window, fwhm, tau = (
        _med([c.get(k) for c in cmp]) for k in ("i_max", "window_phys", "fwhm_phys", "sigma_target_phys")
    )
    if not imax:
        return {"unavailable": "i_max missing: re-run", "repeats": len(runs)}
    truths = [((r.get("true_params") or {}).get("params") or {}).get("zeeman_shift") for r in runs]
    bounds = ((runs[0].get("true_params") or {}).get("bounds") or {}).get("zeeman_shift")
    if not bounds:
        return {"unavailable": "rows carry no zeeman_shift bounds", "repeats": len(runs)}
    # The error of the best guess before any measurement: the prior's spread, W / sqrt(12) for the uniform window of a
    # global search (Eq. prior), sigma_0 of the Gaussian prior when tracking. The tracking sigma_0 is the std of the
    # honest prior the truth was drawn around (the generator's), not the spread of the locator's belief, which mixes in
    # a broad share (PRIOR_BROAD_FRACTION) over the whole window: van Trees is a property of the truth's prior, so it
    # holds for every locator with J0 = 1 / sigma_0^2 whatever prior the locator starts from.
    kinds = {c.get("prior_kind") for c in cmp}
    if len(kinds) != 1 or None in kinds:
        raise ValueError(f"runs of one use case must share one stored prior_kind, got {sorted(map(str, kinds))}")
    prior_kind = kinds.pop()
    sigma0 = (
        (float(bounds[1]) - float(bounds[0])) / math.sqrt(12.0)
        if prior_kind == "uniform"
        else _med([tracking_prior_std_phys(r) for r in runs])
    )
    steps = [np.asarray(r["series"]["step"], dtype=np.float64) for r in runs]
    errs = [np.asarray(r["series"]["error"], dtype=np.float64) for r in runs]
    uncs = [np.asarray(r["series"]["claimed_sigma"], dtype=np.float64) for r in runs]
    grid = claims.step_grid(int(max(s[-1] for s in steps)), points=GRID_POINTS)
    err, unc = claims.value_matrix(steps, errs, grid), claims.value_matrix(steps, uncs, grid)
    out: dict[str, Any] = {
        "repeats": len(runs),
        "i_max": imax,
        "window_phys": window,
        "fwhm_phys": fwhm,
        "tau_phys": tau,
        "sigma0_phys": sigma0,
        # The spread of the prior the locator started from (its belief's variance); differs from sigma0_phys when
        # tracking, because the belief mixes a broad share over the window into the honest Gaussian.
        "locator_prior_sigma_phys": _med([c.get("sigma0_phys") for c in cmp]),
        "prior_kind": prior_kind,
        "max_n": int(grid[-1]),
    }
    out["truth_spread_ratio"] = (
        claims.truth_spread_ratio(np.array(truths, dtype=np.float64), *bounds) if None not in truths else None
    )
    # The run-averaged error and claimed sigma over the whole run (Fig. overview; calibration: rms true vs claimed).
    out["rms_curve"] = {
        "n": grid.tolist(),
        "error": claims.rms(err).tolist(),
        "claimed_sigma": claims.rms(unc).tolist(),
    }
    # Tracking: van Trees from the first measurement, rho(n) = (J0 + n I_max)^-1 / MSE(n) <= 1 at every n.
    walk = _walk_step_phys(cmp)
    out["line_walk_step_phys"] = walk
    out["van_trees_from_start"] = (
        van_trees_from_start(grid, err, sigma0, imax, walk) if prior_kind == "gaussian" else None
    )
    # Per-run Gaussian onset from the stored ``negentropy`` series (NaN = never reached the Gaussian peak).
    onsets = np.array([_gaussian_onset(r["series"]) for r in runs], dtype=np.float64)
    # 3. (n0, q) on master's late window, true error and claimed sigma, with the linearity gate.
    ends = np.array([s[-1] for s in steps])
    inv: InverseMse = inverse_mse_curve(steps, errs, ends, grid)
    dips = [c.get("detection_step") for c in cmp if c.get("detection_step")]
    win = late_window(inv, dips)
    out["fit_window"] = list(win) if win else None
    out["fitted_rate"] = out["claimed_rate"] = out["claimed_over_fitted_rate"] = out["sensitivity_ratio"] = None
    if win:
        ft = claims.fit_with_gate(steps, errs, grid, imax, win)
        fc = claims.fit_with_gate(steps, uncs, grid, imax, win)
        out["fitted_rate"], out["claimed_rate"] = _gated_dict(ft), _gated_dict(fc)
        out["claimed_over_fitted_rate"] = claims.rate_ratio(fc, ft)
        out["sensitivity_ratio"] = claims.sensitivity_ratio(grid, err, imax, win)
    # 2. Regimes (needs the true-error q of section 3 and the per-run detection / Gaussian steps).
    reg = claims.regimes(
        grid,
        err,
        sigma0,
        detection_step=_med([float(d) for d in dips]),
        gaussian_onset=_med([float(o) for o in onsets if np.isfinite(o)]),
        q=out["fitted_rate"]["rate"] if out["fitted_rate"] else None,
        imax=imax,
    )
    out["regimes"] = {
        "plateau_ratio": reg.plateau_ratio,
        "detection_step": reg.detection_step,
        "gaussian_onset_step": reg.gaussian_onset,
        "converge_end_step": reg.converge_end,
        "tail_slope": reg.tail_slope,
        "tail_points": reg.tail_points,
        "why": reg.why,
    }
    # 4. Coverage: gated per-n curve (from each run's Gaussian onset), at the end of the run, and from the signed z
    # series when stored.
    cov = claims.coverage_curve(err, unc, onsets, grid)
    out["coverage"] = {n: cov[int(np.searchsorted(grid, n))] for n in COVERAGE_STEPS if n <= grid[-1]}
    out["coverage_gated_final"] = cov[-1]
    out["gaussian_share"] = float(np.isfinite(onsets).mean())
    zs = [r["series"].get(Z_KEY) for r in runs]
    if all(z for z in zs):
        z_last = np.array([z[-1] for z in zs], dtype=np.float64)
        out["coverage_z_final"] = claims.coverage_from_z(z_last[np.isfinite(z_last)])
    else:
        out["coverage_z_final"] = None
    # Quantile coverage (PIT, valid before the posterior is Gaussian): None for runs recorded before the pit series.
    pits = [r["series"].get(PIT_KEY) for r in runs]
    if all(p for p in pits):
        p_last = np.array([p[-1] for p in pits], dtype=np.float64)
        out["coverage_pit_final"] = claims.coverage_from_pit(p_last[np.isfinite(p_last)])
    else:
        out["coverage_pit_final"] = None
    return out
