import json, math, collections
import numpy as np
R = json.load(open("scratch/claims_results.json"))
by = collections.defaultdict(list)
for k,v in R.items():
    g,n,s=k.split("|"); v["g"]=g; v["n"]=n; by[s].append(v)
print("strategy | fits | median q | share CI_hi<1 | median q_claimed/q | median cov1 | cov2 | median R(1) | median n_half/n_c")
for s,vs in sorted(by.items()):
    fits=[v for v in vs if v.get("fit_true") and v["fit_true"]["q"]]
    q=[v["fit_true"]["q"] for v in fits]; hi=[v["fit_true"]["q_ci"][1] for v in fits if v["fit_true"]["q_ci"]]
    rat=[v["fit_claimed"]["q"]/v["fit_true"]["q"] for v in fits if v["fit_claimed"] and v["fit_claimed"]["q"]]
    c1=[v["coverage_at_stop"]["within_1_all"] for v in vs]; c2=[v["coverage_at_stop"]["within_2_all"] for v in vs]
    r1=[v["R_true"]["1"] for v in vs if v["R_true"].get("1")]
    print(s, len(fits),"/",len(vs), f"{np.median(q):.3f}", f"{np.mean([h<1 for h in hi]):.2f}", f"{np.median(rat):.2f}", f"{np.median(c1):.2f}", f"{np.median(c2):.2f}", f"{np.median(r1):.2f}")
print("SNR>=5 only SBED/Smart-SBED/Smart-Sweep/Sweep3 q & q*R_inf for sweeps")
for s in ("SimpleSweep-2perFWHM","SimpleSweep-3perFWHM","SimpleSweep-4perFWHM"):
    vals=[]
    for v in by[s]:
        if v["fit_true"] and v["fit_true"]["q"]:
            rinf=27*v["window"]/(8*math.pi*v["fwhm"]); vals.append(v["fit_true"]["q"]*rinf)
    print(s,"median q*R_inf",f"{np.median(vals):.2f}" if vals else None,"n",len(vals), "(1 = claim)")
# cost predicted/measured
for s in ("Bayesian-SBED","SimpleSweep-3perFWHM","Smart-Sweep"):
    rr=[v["predicted_cost"]/v["passage"]["median"] for v in by[s] if v.get("predicted_cost") and v.get("predicted_cost")>0 and v.get("passage") and v["passage"]["median"]]
    print(s,"pred/measured passage",f"median {np.median(rr):.2f}", "n",len(rr), "range", f"{min(rr):.2f}-{max(rr):.2f}")
# n_half vs n_c
for s in ("Bayesian-SBED",):
    print("R(1) gap implied", [f"{1/math.sqrt(v['R_true']['1']):.2f}" for v in by[s] if v['R_true'].get('1')][:8], "article 1.81")
print("truth spread by span", {g.split('-')[2]: round(v['truth_spread_ratio'],2) for v in by['Bayesian-SBED'] for g in [v['g']]})
print("R at last alive grid point: share with R>1.01 (bound violated)")
viol=[(k,n,r) for k,v in R.items() for n,r in v["R_true"].items() if r and r>1.01]
print(len(viol), viol[:5])
