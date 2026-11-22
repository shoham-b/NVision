"""Final posterior mean of each lab run in recording order, all windows interleaved (read-only).
Shows whether the 0.8-0.9 MHz clusters of final means are steps of the line in time (drift/jumps) or random
locks onto sub-structure. Usage: uv run python debug/lab_means_vs_time.py <folder>"""
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
    t0 = ((ts[0, 2] * 24 + ts[0, 3]) * 60 + ts[0, 4]) * 60 + ts[0, 5]
    t1 = ((ts[-1, 2] * 24 + ts[-1, 3]) * 60 + ts[-1, 4]) * 60 + ts[-1, 5]
    w = round(t["scan_range_mhz"][1] - t["scan_range_mhz"][0])
    rows.append((t0, t1, w, t["mean_mhz"][-1], t["std_mhz"][-1], p.stem))
rows.sort()
T0 = rows[0][0]
for t0, t1, w, m, s, name in rows:
    print(f"{(t0 - T0) / 60:7.1f} min  dur {(t1 - t0) / 60:5.1f}  W={w:4d}  mean={m:9.3f}  sig={s:6.3f}  {name}")
