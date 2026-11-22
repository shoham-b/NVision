"""Lab calibration with the line's jumps removed (read-only).

The lab line jumps between recordings by 0.5-1 MHz (see debug/lab_means_vs_time.py). Successive runs of one window
are taken ~2 min apart, so the difference of two consecutive final means has variance s1^2 + s2^2 when the line did
not move between them. Ratio r = |m_{i+1} - m_i| / sqrt(s_i^2 + s_{i+1}^2) is a z-score free of slow drift; a jump
shows as a gross outlier. Reported per window: median |r| (0.674 if calibrated), share of |r| < 1 / < 2 (0.68/0.95),
and the same with the claimed sigma inflated by 1.5x and 2x. Also, for runs that stopped on precision, the split by
steps to stop. Usage: uv run python debug/lab_segment_calibration.py <folder>"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

from nvision.metrics.lab_process import run_trace

rows = []
for p in sorted(Path(sys.argv[1]).glob("*.mat")):
    try:
        t = run_trace(p)
    except Exception:  # noqa: BLE001
        continue
    esr = sio.loadmat(p, squeeze_me=True, struct_as_record=False, variable_names=["myStruct"])["myStruct"].BayesianESR
    ts = np.atleast_2d(np.asarray(esr.averagesTimeStamp, dtype=float))
    t0 = (((ts[0, 1] * 31 + ts[0, 2]) * 24 + ts[0, 3]) * 60 + ts[0, 4]) * 60 + ts[0, 5]
    w = round(t["scan_range_mhz"][1] - t["scan_range_mhz"][0])
    rows.append((w, t0, t["mean_mhz"][-1], t["std_mhz"][-1], t["n_steps"]))
print("W  pairs  med|r|  P(|r|<1) P(|r|<2)  med|r|@x1.5  P<2@x1.5  med|r|@x2  P<2@x2  | med|r| short(<=20 steps) long")
for w in sorted({r[0] for r in rows}):
    g = sorted([r for r in rows if r[0] == w], key=lambda r: r[1])
    m = np.array([r[2] for r in g]); s = np.array([r[3] for r in g]); n = np.array([r[4] for r in g])
    d = np.abs(np.diff(m)); se = np.sqrt(s[:-1] ** 2 + s[1:] ** 2)
    r = d / se
    short = (n[:-1] <= 20) & (n[1:] <= 20)
    longr = (n[:-1] > 20) & (n[1:] > 20)
    out = [np.median(r), np.mean(r < 1), np.mean(r < 2)]
    for k in (1.5, 2.0):
        out += [np.median(r / k), np.mean(r / k < 2)]
    print(f"{w:4d} {len(r):5d} " + " ".join(f"{x:8.2f}" for x in out)
          + f"   | {np.median(r[short]) if short.any() else np.nan:6.2f} ({short.sum()}) {np.median(r[longr]) if longr.any() else np.nan:6.2f} ({longr.sum()})")
