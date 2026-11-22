"""Why the lab locator measures across the window after detection (read-only over the lab .mat runs).

Per window: share of post-detection steps with focus locked, exploration share, focus window width, measurement
spread about the run's final mean (in HWHM), and the share of measurements far (>3 HWHM) from the line, split by
whether the focus was locked at that step.

Usage: uv run python debug/lab_placement_mechanism.py <folder with Run_*MHz_*.mat>
"""
import re
import sys
from pathlib import Path

import numpy as np

from nvision.metrics.lab_process import run_trace
from nvision.metrics.lab_search_fit import detection_step
from nvision.metrics.lab_detection_aligned import relative_information

folder = Path(sys.argv[1])
by_w: dict[int, list] = {}
for p in sorted(folder.glob("*.mat")):
    try:
        t = run_trace(p)
    except Exception as e:  # noqa: BLE001 - debug listing
        continue
    w = round(t["scan_range_mhz"][1] - t["scan_range_mhz"][0])
    by_w.setdefault(w, []).append(t)

print("W  n  det_med locked_after expl_after win_w_med far_locked far_unlocked far_all eff_all eff_locked frac_in_win")
for w, ts in sorted(by_w.items()):
    rows = []
    for t in ts:
        s = np.asarray(t["std_mhz"])
        nd = detection_step(s)
        if nd is None:
            continue
        fwhm = t["model"]["fwhm_mhz"]
        hw = 0.5 * fwhm
        f = np.asarray(t["drive_freq_mhz"])[nd:]
        centre = t["mean_mhz"][-1]
        lk = np.asarray(t["locked"])[nd:].astype(bool)
        ex = np.asarray(t["exploration"])[nd:].astype(bool) if t["exploration"] is not None else np.zeros_like(lk)
        lo = np.asarray(t["window_lo_mhz"])[nd:]
        hi = np.asarray(t["window_hi_mhz"])[nd:]
        far = np.abs(f - centre) > 3 * hw
        eff = relative_information(f, centre, fwhm)
        inwin = (f >= lo - 1e-9) & (f <= hi + 1e-9)
        rows.append(
            (nd, lk.mean(), ex.mean(), np.median(hi - lo), far[lk].mean() if lk.any() else np.nan,
             far[~lk].mean() if (~lk).any() else np.nan, far.mean(), eff.mean(), eff[lk].mean() if lk.any() else np.nan,
             inwin.mean())
        )
    r = np.array(rows, dtype=float)
    med = np.nanmedian(r, axis=0)
    print(f"{w:4d} {len(r):2d} " + " ".join(f"{x:8.3f}" for x in med))
