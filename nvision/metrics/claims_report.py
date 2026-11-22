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
from nvision.metrics.detection_aligned import gaussian_onset
from nvision.metrics.refine_rates import InverseMse, inverse_mse_curve, late_window

R_STEPS = (1, 3, 10, 30, 100, 300, 1000)
COVERAGE_STEPS = (30, 100, 300, 1000)
Z_KEY = "z"
# The stored series have at most 80 points per run; a denser grid lets each half of the fit window hold the 5 points
# the line fit needs (it adds no information: values between stored samples are interpolated).
GRID_POINTS = 200


def _med(values: list[float | None]) -> float | None:
    v = [x for x in values if x is not None and math.isfinite(x)]
    return float(np.median(v)) if v else None


def _gated_dict(g: claims.GatedFit) -> dict[str, Any]:
    f = g.fit
    return {
        "q": f.q if f else None,
        "n0": f.n0 if f and math.isfinite(f.n0) else None,
        "r2": f.r2 if f else None,
        "window": [f.n_lo, f.n_hi] if f else None,
        "q_ci": g.q_ci,
        "n0_ci": g.n0_ci,
        "q_halves": g.q_halves,
        "q_halves_ci": g.q_halves_ci,
        "linear": g.linear,
        "why": g.why,
    }


def _gaussian_onset(series: dict[str, Any]) -> float:
    """A run's Gaussian onset (step) from its stored series ``s`` (steps) and ``g`` (negentropy); NaN if never.

    ``series["s"]``, ``series["g"]``: length m. Raises when ``g`` is missing: such a run predates the stored
    negentropy and must be re-run, not silently counted as never Gaussian.
    """
    g = series.get("g")
    if not g:
        raise ValueError("run has no negentropy series 'g': re-run it (nv render cannot fill it)")
    s = np.asarray(series["s"], dtype=np.float64)
    gv = np.array([math.nan if x is None else x for x in g], dtype=np.float64)
    if gv.shape != s.shape:
        raise ValueError(f"series 's' and 'g' differ in length: {s.shape} vs {gv.shape}")
    onset = gaussian_onset(s, gv)
    return math.nan if onset is None else float(onset)


def analyse_use_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The claim numbers of one locator's repeats in one use case.

    ``rows``: items of ``/api/scan-fields`` (``compare`` = stored scalars, ``series`` = ``s`` (stored step), ``e`` (true
    error, Hz), ``u`` (claimed sigma, Hz), optionally ``z``; ``true_params``). Runs without a series are left out.
    """
    runs = [r for r in rows if r.get("series") and all(r["series"].get(k) for k in ("s", "e", "u"))]
    if len(runs) < claims.MIN_RUNS:
        return {
            "unavailable": f"{len(runs)} runs with a per-step series (need {claims.MIN_RUNS})",
            "repeats": len(runs),
        }
    cmp = [r.get("compare") or {} for r in runs]
    j0, j0_var, imax, window, fwhm, tau = (
        _med([c.get(k) for c in cmp])
        for k in (
            "stage_prior_info",
            "stage_prior_info_var",
            "stage_fisher_star",
            "stage_window_phys",
            "stage_fwhm_true_phys",
            "stage_tau_phys",
        )
    )
    if not (j0 and j0_var and imax):
        return {"unavailable": "stage_prior_info / stage_fisher_star missing: re-run", "repeats": len(runs)}
    steps = [np.asarray(r["series"]["s"], dtype=np.float64) for r in runs]
    errs = [np.asarray(r["series"]["e"], dtype=np.float64) for r in runs]
    uncs = [np.asarray(r["series"]["u"], dtype=np.float64) for r in runs]
    grid = claims.step_grid(int(max(s[-1] for s in steps)), points=GRID_POINTS)
    err, unc = claims.value_matrix(steps, errs, grid), claims.value_matrix(steps, uncs, grid)
    sigma0 = j0_var**-0.5
    out: dict[str, Any] = {
        "repeats": len(runs),
        "J0": j0,
        "J0_var": j0_var,
        "I_max": imax,
        "window_phys": window,
        "fwhm_phys": fwhm,
        "tau_phys": tau,
        "sigma0_phys": sigma0,
        "n_c": j0 / imax,
        "prior_gap": claims.prior_gap(j0, j0_var),
        "max_n": int(grid[-1]),
    }
    truths = [((r.get("true_params") or {}).get("params") or {}).get("zeeman_shift") for r in runs]
    bounds = ((runs[0].get("true_params") or {}).get("bounds") or {}).get("zeeman_shift")
    out["truth_spread_ratio"] = (
        claims.truth_spread_ratio(np.array(truths, dtype=np.float64), *bounds)
        if bounds and None not in truths
        else None
    )
    # 1. The bound.
    r_true, r_claimed = claims.bound_ratio(grid, err, j0, imax), claims.bound_ratio(grid, unc, j0, imax)
    lo, hi = claims.bound_ratio_ci(grid, err, j0, imax)
    out["R_true"] = {n: float(r_true[np.searchsorted(grid, n)]) for n in R_STEPS if n <= grid[-1]}
    out["R_true_ci"] = {
        n: [float(lo[i]), float(hi[i])] for n in R_STEPS if n <= grid[-1] for i in [np.searchsorted(grid, n)]
    }
    out["R_claimed"] = {n: float(r_claimed[np.searchsorted(grid, n)]) for n in R_STEPS if n <= grid[-1]}
    out["bound_violated"] = claims.bound_violated(lo)
    # Per-run Gaussian onset from the stored negentropy series ``g`` (NaN = never reached the Gaussian peak).
    onsets = np.array([_gaussian_onset(r["series"]) for r in runs], dtype=np.float64)
    # 3. (n0, q) on master's late window, true error and claimed sigma, with the linearity gate.
    ends = np.array([s[-1] for s in steps])
    inv: InverseMse = inverse_mse_curve(steps, errs, ends, grid)
    dips = [c.get("dip_detected_step") for c in cmp if c.get("dip_detected_step")]
    win = late_window(inv, dips)
    out["fit_window"] = list(win) if win else None
    out["fit_true"] = out["fit_claimed"] = out["rate_ratio_claimed_over_true"] = out["sensitivity_ratio"] = None
    if win:
        ft = claims.fit_with_gate(steps, errs, grid, imax, win)
        fc = claims.fit_with_gate(steps, uncs, grid, imax, win)
        out["fit_true"], out["fit_claimed"] = _gated_dict(ft), _gated_dict(fc)
        out["rate_ratio_claimed_over_true"] = claims.rate_ratio(fc, ft)
        out["sensitivity_ratio"] = claims.sensitivity_ratio(grid, err, imax, win)
    # 2. Regimes (needs the true-error q of section 3 and the per-run detection / Gaussian steps).
    reg = claims.regimes(
        grid,
        err,
        sigma0,
        detection_step=_med([float(d) for d in dips]),
        gaussian_onset=_med([float(o) for o in onsets if np.isfinite(o)]),
        q=out["fit_true"]["q"] if out["fit_true"] else None,
        imax=imax,
    )
    out["regimes"] = {
        "plateau_ratio": reg.plateau_ratio,
        "detection_step": reg.detection_step,
        "gaussian_onset": reg.gaussian_onset,
        "converge_end": reg.converge_end,
        "tail_slope": reg.tail_slope,
        "tail_points": reg.tail_points,
        "why": reg.why,
    }
    # 4. Coverage: gated per-n curve (from each run's Gaussian onset), at the end of the run, and from the signed z
    # series when stored.
    cov = claims.coverage_curve(err, unc, onsets, grid)
    out["coverage"] = {n: cov[int(np.searchsorted(grid, n))] for n in COVERAGE_STEPS if n <= grid[-1]}
    out["coverage_at_budget"] = cov[-1]
    out["gaussian_share"] = float(np.isfinite(onsets).mean())
    zs = [r["series"].get(Z_KEY) for r in runs]
    if all(z for z in zs):
        z_last = np.array([z[-1] for z in zs], dtype=np.float64)
        out["coverage_z_at_budget"] = claims.coverage_from_z(z_last[np.isfinite(z_last)])
    else:
        out["coverage_z_at_budget"] = None
    return out
