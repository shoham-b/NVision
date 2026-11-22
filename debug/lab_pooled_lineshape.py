"""Pooled lineshape of each lab window: all runs' measurements against drive_freq - (run's final mean), fitted with a
free Lorentzian (baseline, contrast, FWHM). Compares the fitted shape with the fixed shape the lab model assumed, and
the residual std with the per-shot standard error the lab recorded (measurementSE). Read-only.
Usage: uv run python debug/lab_pooled_lineshape.py <folder>"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import curve_fit

from nvision.metrics.lab_process import run_trace


def lor(x, b, c, fw):
    return b * (1 - c / (1 + (x / (0.5 * fw)) ** 2))


by = {}
for p in sorted(Path(sys.argv[1]).glob("*.mat")):
    try:
        t = run_trace(p)
    except Exception:  # noqa: BLE001
        continue
    esr = sio.loadmat(p, squeeze_me=True, struct_as_record=False, variable_names=["myStruct"])["myStruct"].BayesianESR
    se = np.atleast_1d(np.asarray(esr.state.measurementSE, dtype=float))
    w = round(t["scan_range_mhz"][1] - t["scan_range_mhz"][0])
    by.setdefault(w, []).append((np.asarray(t["drive_freq_mhz"]) - t["mean_mhz"][-1], np.asarray(t["signal"]), se, t["model"]))
print("W  n_meas  model(b,C,FWHM)  fit(b,C,FWHM)  resid_std  median_SE  resid/SE")
for w, g in sorted(by.items()):
    x = np.concatenate([a[0] for a in g]); y = np.concatenate([a[1] for a in g]); se = np.concatenate([a[2] for a in g])
    m = g[0][3]
    near = np.abs(x) < 15
    popt, _ = curve_fit(lor, x[near], y[near], p0=[m["baseline"], m["contrast"], m["fwhm_mhz"]])
    r = y[near] - lor(x[near], *popt)
    print(f"{w:4d} {near.sum():6d}  ({m['baseline']:.2f},{m['contrast']:.3f},{m['fwhm_mhz']:.2f})  "
          f"({popt[0]:.3f},{popt[1]:.3f},{popt[2]:.2f})  {np.std(r):.4f}  {np.median(se):.4f}  {np.std(r) / np.median(se):.2f}")


def lor4(x, x0, b, c, fw):
    return b * (1 - c / (1 + ((x - x0) / (0.5 * fw)) ** 2))


print("\nPer-run free fit (centre, b, C, FWHM), runs with >= 40 measurements within 3 model FWHM of their mean:")
print("W  n_runs  median C [q25,q75]  median FWHM [q25,q75]  median C*FWHM (model C*FWHM)")
for w, g in sorted(by.items()):
    cs, fws = [], []
    for x, y, _se, m in g:
        near = np.abs(x) < 3 * m["fwhm_mhz"]
        if near.sum() < 40:
            continue
        try:
            popt, _ = curve_fit(lor4, x[near], y[near], p0=[0.0, m["baseline"], m["contrast"], m["fwhm_mhz"]],
                                bounds=([-5, 0.5, 0.0, 0.3], [5, 1.5, 0.5, 20]))
        except RuntimeError:
            continue
        cs.append(popt[2]); fws.append(popt[3])
    if not cs:
        print(f"{w:4d} 0"); continue
    cs, fws = np.array(cs), np.array(fws); m = g[0][3]
    q = lambda a: f"{np.median(a):.3f} [{np.quantile(a, .25):.3f},{np.quantile(a, .75):.3f}]"
    print(f"{w:4d} {len(cs):4d}  {q(cs)}  {q(fws)}  {np.median(cs * fws):.3f} ({m['contrast'] * m['fwhm_mhz']:.3f})")
