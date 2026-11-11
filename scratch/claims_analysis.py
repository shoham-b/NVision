"""Read-only: test the article's claims on the cached pure-Lorentzian SNR x window study (hfunresolved, gs0).

Reads artifacts/cache through CacheBridge (no writes) and writes scratch/claims_results.json.
Usage: uv run python scratch/claims_analysis.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from nvision.cache import CacheBridge
from nvision.cli.api_server import _combo_key
from nvision.metrics import claims
from nvision.sim.combinations import CombinationGrid

CACHE = Path(r"C:/Users/shoha/git/NVision/artifacts/cache")
OUT = Path(__file__).with_name("claims_results.json")
SUFFIX = "gs0-hfunresolved"


def _load(bridge: CacheBridge, combo: dict) -> list[dict]:
    repo = bridge.get_cache_for_category(CombinationGrid.generator_category(combo["generator"]))
    meta = repo._repeats.load_repeats_meta(_combo_key(combo), int(combo["repeats"]))
    runs = []
    for ents, row in meta:
        scan = next((e for e in (ents or []) if e.get("type") == "scan"), None)
        if scan is None or not scan.get("series"):
            continue
        tp = scan["true_params"]
        runs.append({"row": row, "series": scan["series"], "truth": tp["params"], "bounds": tp["bounds"]})
    return runs


def _med(runs: list[dict], key: str) -> float | None:
    v = [r["row"].get(key) for r in runs if r["row"].get(key) is not None]
    return float(np.median(v)) if v else None


def _med_vals(v: list) -> float | None:
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def analyse(runs: list[dict]) -> dict:
    j0, j0v, imax = _med(runs, "stage_prior_info"), _med(runs, "stage_prior_info_var"), _med(runs, "stage_fisher_star")
    window, fwhm, tau = _med(runs, "stage_window_phys"), _med(runs, "stage_fwhm_true_phys"), _med(runs, "stage_tau_phys")
    max_n = max(r["series"]["s"][-1] for r in runs)
    grid = claims.step_grid(max_n)
    err = claims.series_matrix([r["series"]["s"] for r in runs], [r["series"]["e"] for r in runs], grid)
    unc = claims.series_matrix([r["series"]["s"] for r in runs], [r["series"]["u"] for r in runs], grid)
    out: dict = {"repeats": len(runs), "J0": j0, "Imax": imax, "window": window, "fwhm": fwhm, "tau": tau, "max_n": max_n}
    sigma0 = j0v**-0.5
    out["sigma0"] = sigma0
    out["prior_gap"] = claims.prior_gap(j0, j0v)
    z = [r["truth"]["zeeman_shift"] for r in runs]
    lo, hi = runs[0]["bounds"]["zeeman_shift"]
    out["truth_spread_ratio"] = claims.truth_prior_spread_ratio(np.array(z), lo, hi)
    out["alive90_until"] = int(grid[claims.alive_fraction(err) >= claims.ALIVE_MIN][-1]) if (claims.alive_fraction(err) >= claims.ALIVE_MIN).any() else None

    def at(arr: np.ndarray, n: int) -> float | None:
        i = int(np.searchsorted(grid, n))
        if i >= grid.size or claims.alive_fraction(err)[i] < claims.ALIVE_MIN:
            return None
        v = arr[i]
        return float(v) if np.isfinite(v) else None

    ns = [1, 3, 10, 30, 100, 300, 1000]
    out["R_true"] = {n: at(claims.bound_ratio(grid, err, j0, imax), n) for n in ns}
    out["R_claimed"] = {n: at(claims.bound_ratio(grid, unc, j0, imax), n) for n in ns}
    reg = claims.regimes(grid, err, sigma0)
    out["regimes"] = {"plateau_ratio": reg.plateau_ratio, "n_half": reg.n_half, "drop_factor": reg.drop_factor,
                      "tail_slope": reg.tail_slope, "tail_points": reg.tail_points, "why": reg.why}
    if reg.n_half is not None:
        n_min = claims.FIT_START_FACTOR * reg.n_half
        for name, a in (("true", err), ("claimed", unc)):
            f = claims.fit_strategy_numbers(grid, a, imax, n_min=n_min)
            out[f"fit_{name}"] = {"q": f.q, "q_ci": f.q_ci, "n0": f.n0, "n0_ci": f.n0_ci, "window": [f.n_lo, f.n_hi],
                                  "points": f.points, "why": f.why, "q_halves": f.q_halves}
        out["sens_ratio"] = claims.sensitivity_ratio(grid, err, imax, n_min=n_min)
    else:
        out["fit_true"] = out["fit_claimed"] = None
    gate = np.array([r["row"].get("refine_start_step") or math.nan for r in runs], dtype=float)
    cov = claims.coverage_curve(err, unc, gate, grid)
    out["coverage"] = {}
    for n in (30, 100, 300, 1000):
        i = int(np.searchsorted(grid, n))
        if i < grid.size:
            out["coverage"][n] = {k: (None if not np.isfinite(cov[k][i]) else float(cov[k][i]))
                                  for k in ("runs", "within_1", "within_2", "ungated_within_1", "ungated_within_2")}
    # coverage at each run's own last stored step (the stop)
    last_e = np.array([r["series"]["e"][-1] for r in runs], dtype=float)
    last_u = np.array([r["series"]["u"][-1] for r in runs], dtype=float)
    refined = np.isfinite(gate)
    out["coverage_at_stop"] = {
        "runs": len(runs), "refined_runs": int(refined.sum()),
        "within_1_all": float(np.mean(last_e <= last_u)), "within_2_all": float(np.mean(last_e <= 2 * last_u)),
        "within_1_refined": float(np.mean((last_e <= last_u)[refined])) if refined.any() else None,
        "within_2_refined": float(np.mean((last_e <= 2 * last_u)[refined])) if refined.any() else None,
    }
    if tau:
        steps = claims.passage_steps(grid, err, tau)
        out["passage"] = {"median": float(np.nanmedian(steps)) if np.isfinite(steps).any() else None,
                          "reached": float(np.isfinite(steps).mean())}
        fit = out.get("fit_true")
        if fit and fit["q"]:
            out["predicted_cost"] = claims.predicted_cost(fit["n0"], fit["q"], imax, tau)
    # Heavy-tail check: rms/median of the true error at the median stop step (1.48 for a Gaussian |e|).
    i_stop = min(int(np.searchsorted(grid, np.median([r["series"]["s"][-1] for r in runs]))), grid.size - 1)
    col = err[:, i_stop][np.isfinite(err[:, i_stop])]
    out["rms_over_median_at_stop"] = float(np.sqrt(np.mean(col**2)) / np.median(col)) if col.size >= 10 else None
    def _direct(row: dict, stages: tuple[int, ...]) -> float | None:
        num = den = 0.0
        for k in stages:
            eff, steps = row.get(f"stage{k}_sens_eff"), row.get(f"stage{k}_steps")
            if eff is not None and steps:
                num, den = num + eff * steps, den + steps
        return num / den if den else None

    out["q_direct_all"] = _med_vals([_direct(r["row"], (1, 2, 3)) for r in runs])
    out["q_direct_refine"] = _med_vals([_direct(r["row"], (3,)) for r in runs])
    out["measurements_median"] = _med(runs, "measurements")
    out["stopped_median"] = float(np.median([r["series"]["s"][-1] for r in runs]))
    out["censored_share"] = float(np.mean([r["row"].get("failure_reason") == "max_steps" for r in runs]))
    out["claim_columns_present"] = any(r["row"].get("stage_pit_end") is not None for r in runs)
    return out


def main() -> int:
    bridge = CacheBridge(CACHE)
    combos = [c for c in bridge.list_combinations_with_updated_at() if c["generator"].endswith(SUFFIX)]
    results: dict = {}
    for i, combo in enumerate(sorted(combos, key=lambda c: (c["generator"], c["noise"], c["strategy"]))):
        runs = _load(bridge, combo)
        if len(runs) < 10:
            continue
        key = f"{combo['generator']}|{combo['noise']}|{combo['strategy']}"
        results[key] = analyse(runs)
        print(i, key, results[key]["repeats"], file=sys.stderr)
    bridge.close()
    OUT.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
