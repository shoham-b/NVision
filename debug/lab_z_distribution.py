"""z = (final mean - window median of final means) / claimed sigma per lab run: robust scale, tails, and the
drift-floor that would explain the scatter (read-only).

Usage: uv run python debug/lab_z_distribution.py <folder>
"""

import sys
from pathlib import Path

import numpy as np

from nvision.metrics.lab_process import run_trace

folder = Path(sys.argv[1])
by_w: dict[int, list] = {}
for p in sorted(folder.glob("*.mat")):
    try:
        t = run_trace(p)
    except Exception:
        continue
    by_w.setdefault(round(t["scan_range_mhz"][1] - t["scan_range_mhz"][0]), []).append(t)
print("W n  sd_z  robust_sd_z  frac|z|>3  cov1  cov2  floor_mhz(robust)  locked_final  stop_reasons")
for w, ts in sorted(by_w.items()):
    m = np.array([t["mean_mhz"][-1] for t in ts])
    s = np.array([t["std_mhz"][-1] for t in ts])
    z = (m - np.median(m)) / s
    rsd = 1.4826 * np.median(np.abs(z - np.median(z)))
    # extra scatter floor d such that (m-med)/sqrt(s^2+d^2) has robust sd 1
    best = 0.0
    for d in np.linspace(0, 3, 601):
        zz = (m - np.median(m)) / np.sqrt(s**2 + d**2)
        if 1.4826 * np.median(np.abs(zz - np.median(zz))) <= 1.0:
            best = d
            break
    locked = np.mean([t["locked"][-1] for t in ts])
    reasons = {}
    for t in ts:
        reasons[t["stop_reason"]] = reasons.get(t["stop_reason"], 0) + 1
    print(
        f"{w:4d} {len(ts):3d} {np.std(z, ddof=1):6.2f} {rsd:8.2f} {np.mean(np.abs(z) > 3):8.2f} "
        f"{np.mean(np.abs(z) < 1):5.2f} {np.mean(np.abs(z) < 2):5.2f} {best:10.2f} {locked:8.2f} {reasons}"
    )
