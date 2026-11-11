"""One locator in one use case -> the article-claim numbers, from the rows ``/api/scan-fields`` serves.

Glue between the cached per-repeat rows (stored scalars + the downsampled per-step ``series``) and the pure functions of
:mod:`nvision.metrics.claims`. Read-only; nothing here writes to the cache. A number that the stored data cannot give
is ``None`` and the reason is in ``why`` (never a stand-in).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nvision.metrics import claims

# Steps at which the run-averaged bound ratio R(n) is reported.
R_STEPS = (1, 3, 10, 30, 100, 300, 1000)
COVERAGE_STEPS = (30, 100, 300, 1000)


def _med(values: list[float | None]) -> float | None:
    v = [x for x in values if x is not None and math.isfinite(x)]
    return float(np.median(v)) if v else None


def _at(grid: np.ndarray, alive: np.ndarray, arr: np.ndarray, n: int) -> float | None:
    """``arr`` (shape (n_grid,)) at the first grid step >= ``n``, only where >= ``ALIVE_MIN`` of the repeats run."""
    i = int(np.searchsorted(grid, n))
    if i >= grid.size or alive[i] < claims.ALIVE_MIN or not math.isfinite(arr[i]):
        return None
    return float(arr[i])


def analyse_use_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The claim numbers of one locator's repeats in one use case.

    ``rows``: items of ``/api/scan-fields``: ``compare`` (stored scalars), ``series`` (``s``, ``e``, ``u``: stored step
    indices, true error and claimed sigma in Hz), ``true_params`` and ``failure_reason``. Runs without a series are left
    out. Returns a JSON-serialisable dict; ``unavailable`` holds the reason when the columns are missing.
    """
    runs = [
        r for r in rows if r.get("series") and r["series"].get("s") and r["series"].get("e") and r["series"].get("u")
    ]
    if len(runs) < 10:
        return {"unavailable": f"{len(runs)} runs with a per-step series (need 10)", "repeats": len(runs)}
    cmp = [r.get("compare") or {} for r in runs]
    j0, j0_var, imax = (
        _med([c.get(k) for c in cmp]) for k in ("stage_prior_info", "stage_prior_info_var", "stage_fisher_star")
    )
    if not (j0 and j0_var and imax):
        return {
            "unavailable": "stage_prior_info / stage_fisher_star missing: re-run to record them",
            "repeats": len(runs),
        }
    window, fwhm, tau = (
        _med([c.get(k) for c in cmp]) for k in ("stage_window_phys", "stage_fwhm_true_phys", "stage_tau_phys")
    )
    grid = claims.step_grid(max(r["series"]["s"][-1] for r in runs))
    err = claims.series_matrix([r["series"]["s"] for r in runs], [r["series"]["e"] for r in runs], grid)
    unc = claims.series_matrix([r["series"]["s"] for r in runs], [r["series"]["u"] for r in runs], grid)
    alive = claims.alive_fraction(err)
    sigma0 = j0_var**-0.5
    out: dict[str, Any] = {
        "repeats": len(runs), "J0": j0, "Imax": imax, "window_phys": window, "fwhm_phys": fwhm, "tau_phys": tau,
        "sigma0_phys": sigma0, "n_c": j0 / imax, "prior_gap": claims.prior_gap(j0, j0_var),
        "speedup_predicted": claims.speedup_prediction(window, fwhm) if window and fwhm else None,
        "alive90_until": int(grid[alive >= claims.ALIVE_MIN][-1]) if (alive >= claims.ALIVE_MIN).any() else None,
        "censored_share": float(np.mean([r.get("failure_reason") == "max_steps" for r in runs])),
    }  # fmt: skip
    truths = [((r.get("true_params") or {}).get("params") or {}).get("zeeman_shift") for r in runs]
    bounds = ((runs[0].get("true_params") or {}).get("bounds") or {}).get("zeeman_shift")
    out["truth_spread_ratio"] = (
        claims.truth_prior_spread_ratio(np.array(truths, dtype=float), *bounds)
        if bounds and None not in truths
        else None
    )
    ratio_true, ratio_claimed = claims.bound_ratio(grid, err, j0, imax), claims.bound_ratio(grid, unc, j0, imax)
    out["R_true"] = {n: _at(grid, alive, ratio_true, n) for n in R_STEPS}
    out["R_claimed"] = {n: _at(grid, alive, ratio_claimed, n) for n in R_STEPS}
    reg = claims.regimes(grid, err, sigma0)
    out["regimes"] = {
        "plateau_ratio": reg.plateau_ratio, "n_half": reg.n_half, "drop_factor": reg.drop_factor,
        "tail_slope": reg.tail_slope, "tail_points": reg.tail_points, "why": reg.why,
    }  # fmt: skip
    out["fit_true"] = out["fit_claimed"] = out["sensitivity_ratio"] = out["predicted_cost"] = None
    if reg.n_half is not None:
        n_min = claims.FIT_START_FACTOR * reg.n_half
        for name, arr in (("true", err), ("claimed", unc)):
            f = claims.fit_strategy_numbers(grid, arr, imax, n_min=n_min)
            out[f"fit_{name}"] = {
                "q": f.q, "q_ci": f.q_ci, "n0": f.n0, "n0_ci": f.n0_ci, "n_lo": f.n_lo, "n_hi": f.n_hi,
                "points": f.points, "why": f.why, "q_halves": f.q_halves,
            }  # fmt: skip
        out["sensitivity_ratio"] = claims.sensitivity_ratio(grid, err, imax, n_min=n_min)
        ft = out["fit_true"]
        if tau and ft["q"]:
            out["predicted_cost"] = claims.predicted_cost(ft["n0"], ft["q"], imax, tau)

    # The direct, fit-free measure of "measurements carry q Imax": the mean efficiency g^2/max g^2 of the stored
    # stage metrics (stage_k_sens_eff, weighted by stage_k_steps), i.e. the information the measurements carry
    # about the true parameter; not affected by the belief's inference, heavy tails, or the fit window.
    def _direct(c: dict[str, Any], stages: tuple[int, ...]) -> float | None:
        num = den = 0.0
        for k in stages:
            eff, steps = c.get(f"stage{k}_sens_eff"), c.get(f"stage{k}_steps")
            if eff is not None and steps:
                num, den = num + eff * steps, den + steps
        return num / den if den else None

    out["q_direct_all"] = _med([_direct(c, (1, 2, 3)) for c in cmp])
    out["q_direct_refine"] = _med([_direct(c, (3,)) for c in cmp])
    # Heavy tails: rms over median of the true error at the median stop step (1.48 for a Gaussian |error|).
    i_stop = min(int(np.searchsorted(grid, np.median([r["series"]["s"][-1] for r in runs]))), grid.size - 1)
    col = err[:, i_stop][np.isfinite(err[:, i_stop])]
    out["rms_over_median_at_stop"] = float(np.sqrt(np.mean(col**2)) / np.median(col)) if col.size >= 10 else None
    gate = np.array([c.get("refine_start_step") or math.nan for c in cmp], dtype=float)
    cov = claims.coverage_curve(err, unc, gate, grid)
    out["coverage"] = {}
    for n in COVERAGE_STEPS:
        i = int(np.searchsorted(grid, n))
        if i < grid.size:
            out["coverage"][n] = {
                k: (float(cov[k][i]) if cov[k][i] is not None and math.isfinite(cov[k][i]) else None)
                for k in ("runs", "within_1", "within_2", "ungated_within_1", "ungated_within_2")
            }
    last_e = np.array([r["series"]["e"][-1] for r in runs], dtype=float)
    last_u = np.array([r["series"]["u"][-1] for r in runs], dtype=float)
    out["coverage_at_stop"] = {
        "within_1": float(np.mean(last_e <= last_u)), "within_2": float(np.mean(last_e <= 2 * last_u)),
        "refined_share": float(np.mean(np.isfinite(gate))),
    }  # fmt: skip
    if tau:
        steps = claims.passage_steps(grid, err, tau)
        out["passage"] = {
            "median": float(np.nanmedian(steps)) if np.isfinite(steps).any() else None,
            "reached": float(np.isfinite(steps).mean()),
        }
    # Columns recorded only by runs made after nvision/metrics/claim_columns.py existed.
    pit = [c.get("stage_pit_end") for c in cmp if c.get("stage_pit_end") is not None]
    out["pit"] = claims.pit_coverage(np.array(pit)) if len(pit) >= 10 else None
    sw = _med([c.get("stage_fisher_sweep") for c in cmp])
    out["speedup_measured_rate"] = imax / sw if sw else None
    share = [c.get("stage2_far_share") for c in cmp]
    mass = [c.get("stage2_far_mass") for c in cmp]
    pairs = [(s, m) for s, m in zip(share, mass, strict=True) if s is not None and m is not None]
    out["far"] = (
        claims.far_placement(np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
        if len(pairs) >= 10
        else None
    )
    out["missing_columns"] = [
        name for name, ok in (("PIT", out["pit"]), ("sweep information", sw), ("far placement", out["far"])) if not ok
    ]
    return out
