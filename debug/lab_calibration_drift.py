"""Lab claimed sigma vs the scatter of repeated runs: is the 2x a drift between runs? (read-only)

Per window: median final claimed sigma, std of final means over runs, the same after removing a linear (and a
smooth) trend in recording time, and the successive-difference scatter std(diff)/sqrt(2) in recording order
(insensitive to slow drift). Also the residual-to-shot-noise ratio and the lag-1 autocorrelation of the residuals in
measurement order (in-run drift or model misfit shows as positive autocorrelation) and of residuals near the line.

Usage: uv run python debug/lab_calibration_drift.py <folder with Run_*MHz_*.mat> [--final-step N]
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

from nvision.metrics.lab_detection_aligned import lorentzian_signal
from nvision.metrics.lab_process import run_trace

folder = Path(sys.argv[1])
final_step = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--final-step" else None


def start_time(p: Path) -> float:
    esr = sio.loadmat(p, squeeze_me=True, struct_as_record=False, variable_names=["myStruct"])["myStruct"].BayesianESR
    ts = np.atleast_2d(np.asarray(esr.averagesTimeStamp, dtype=float))
    _y, mo, d, h, mi, s = ts[0]
    return ((d * 24 + h) * 60 + mi) * 60 + s + mo * 31 * 86400


by_w: dict[int, list] = {}
for p in sorted(folder.glob("*.mat")):
    try:
        t = run_trace(p)
    except Exception:
        continue
    t["t0"] = start_time(p)
    w = round(t["scan_range_mhz"][1] - t["scan_range_mhz"][0])
    by_w.setdefault(w, []).append(t)

print("W n  claimed_med  scatter  ratio  detrend_ratio  succdiff_ratio  resid/shot  rho1_resid  rho1_near  span_h")
for w, ts in sorted(by_w.items()):
    ts.sort(key=lambda t: t["t0"])
    k = final_step or 10**9
    m = np.array([t["mean_mhz"][min(k, t["n_steps"]) - 1] for t in ts])
    s = np.array([t["std_mhz"][min(k, t["n_steps"]) - 1] for t in ts])
    tt = np.array([t["t0"] for t in ts]) / 3600.0
    claimed = np.sqrt(np.mean(s**2))
    scatter = np.std(m, ddof=1)
    A = np.vstack([np.ones_like(tt), tt - tt.mean()]).T
    resid = m - A @ np.linalg.lstsq(A, m, rcond=None)[0]
    detr = np.sqrt(np.sum(resid**2) / (len(m) - 2))
    sd = np.std(np.diff(m), ddof=1) / np.sqrt(2)
    ratios, rho, rho_near = [], [], []
    for t in ts:
        mdl = t["model"]
        f = np.asarray(t["drive_freq_mhz"])
        y = np.asarray(t["signal"])
        mu = lorentzian_signal(f, t["mean_mhz"][-1], mdl["baseline"], mdl["contrast"], mdl["fwhm_mhz"])
        r = y - mu
        rho.append(np.corrcoef(r[:-1], r[1:])[0, 1])
        near = np.abs(f - t["mean_mhz"][-1]) < mdl["fwhm_mhz"]
        rn = r[near]
        if rn.size > 5:
            rho_near.append(np.corrcoef(rn[:-1], rn[1:])[0, 1])
    print(
        f"{w:4d} {len(ts):3d} {claimed:9.3f} {scatter:9.3f} {scatter / claimed:6.2f} {detr / claimed:9.2f}"
        f" {sd / claimed:12.2f} {'':>10} {np.median(rho):9.3f} {np.median(rho_near) if rho_near else np.nan:9.3f}"
        f" {tt.max() - tt.min():7.1f}"
    )
