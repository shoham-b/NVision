"""Dip-test detection vs the lab-style detection (first steep, lasting drop of the claimed sigma) on the cached
article grid (read-only, in-process TestClient on /api/scan-fields).

Per run: ``detection_step`` (stored, the locator's own held dip test) and ``lab_search_fit.detection_step`` applied
to the stored claimed-sigma series ``u`` at its stored steps ``s`` (the series may be thinned, so the lab step is the
stored step at the drop: exact where the series is per step). Reports, per strategy x span x SNR, the medians and the
median and IQR of (lab - dip).

Usage: uv run python debug/detector_gap.py <artifacts dir>
"""

import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from nvision.cli.api_server import build_app
from nvision.metrics.lab_search_fit import detection_step
from nvision.sim.presets import snr_study_noises

run_dir = Path(sys.argv[1])
client = TestClient(build_app(cache_dir=run_dir / "cache", run_dir=run_dir))
spans = ["1.6", "2", "3", "5", "10", "20", "50"]
print("strategy span SNR n  dip_med  lab_med  gap_med [q25,q75]  share(lab<dip)  no_lab")
for span in spans:
    gen = f"NVCenter-voigt-span{span}-gs0-hfunresolved"
    for noise, _ in snr_study_noises():
        rows = client.get("/api/scan-fields", params={"generator": gen, "noise": noise}).json()
        by: dict[str, list] = {}
        for r in rows:
            ser, cmp = r.get("series") or {}, r.get("compare") or {}
            dip = cmp.get("detection_step")
            if not ser.get("u") or dip is None:
                continue
            s = np.asarray(ser["step"], dtype=float)
            u = np.asarray(ser["claimed_sigma"], dtype=float)
            ok = np.isfinite(u) & (u > 0)
            s, u = s[ok], u[ok]
            k = detection_step(u) if u.size >= 2 else None
            lab = None if k is None else float(s[k - 1])
            by.setdefault(r["strategy"], []).append((float(dip), lab))
        for strat, v in sorted(by.items()):
            if strat not in ("Bayesian-SBED", "SimpleSweep-optimal"):
                continue
            pairs = np.array([(d, lb) for d, lb in v if lb is not None])
            gap = pairs[:, 1] - pairs[:, 0]
            print(
                f"{strat:20s} {span:>4s} {noise:8s} {len(v):3d}"
                f" {np.median(pairs[:, 0]):7.0f} {np.median(pairs[:, 1]):7.0f}"
                f" {np.median(gap):7.0f} [{np.quantile(gap, 0.25):.0f},{np.quantile(gap, 0.75):.0f}]"
                f"  {np.mean(gap < 0):.2f}  {sum(lb is None for _, lb in v)}"
            )
