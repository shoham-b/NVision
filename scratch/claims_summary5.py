import json, math, collections
import numpy as np
R = json.load(open("scratch/claims_results.json"))
by = collections.defaultdict(list)
for k,v in R.items():
    g,n,s=k.split("|"); v["g"]=g; v["n"]=n; by[s].append(v)
print("strategy | gated fits/total | q gated median | q_direct_all median | q_direct_refine median | q_dir*R_inf (sweeps) | rms/med at stop median")
for s,vs in sorted(by.items()):
    fits=[v for v in vs if v.get("fit_true") and v["fit_true"]["q"]]
    nl=[v for v in vs if v.get("fit_true") and v["fit_true"]["q"] is None and v["fit_true"].get("why") and "not linear" in v["fit_true"]["why"]]
    short=[v for v in vs if v.get("fit_true") and v["fit_true"]["q"] is None and v["fit_true"].get("why") and "too short" in v["fit_true"]["why"]]
    q=[v["fit_true"]["q"] for v in fits]
    qd=[v["q_direct_all"] for v in vs if v.get("q_direct_all")]; qr=[v["q_direct_refine"] for v in vs if v.get("q_direct_refine")]
    rinf=[v["q_direct_all"]*27*v["window"]/(8*math.pi*v["fwhm"]) for v in vs if v.get("q_direct_all")]
    rm=[v["rms_over_median_at_stop"] for v in vs if v.get("rms_over_median_at_stop")]
    print(s, f"{len(fits)}/{len(vs)} (not linear {len(nl)}, too short {len(short)}, no detection {sum(1 for v in vs if v['fit_true'] is None)})", f"q {np.median(q):.3f}" if q else "q –", f"direct_all {np.median(qd):.2f}", f"direct_refine {np.median(qr):.2f}", f"direct*R_inf {np.median(rinf):.2f}", f"rms/med {np.median(rm):.2f}")
print("coverage at stop vs gated n=30 (SBED): see claims_summary3")
