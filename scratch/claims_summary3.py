import json, math
R = json.load(open("scratch/claims_results.json"))
def f(x, d=3):
    return "—" if x is None else (f"{x:.{d}g}" if isinstance(x,(int,float)) else str(x))
print("COVERAGE (gated by refine_start) at n, and at stop; rms/median")
for k,v in R.items():
    g,n,s=k.split("|")
    if n not in("SNR(5)","SNR(10)","SNR(20)") or s not in ("Bayesian-SBED","Smart-Sweep","SimpleSweep-3perFWHM"): continue
    c=v["coverage_at_stop"]; cv=v["coverage"]
    print(g.split('-')[2],n,s[:14],"stop: refined",c["refined_runs"],"/",c["runs"],"1s all",f(c["within_1_all"]),"2s all",f(c["within_2_all"]),"| refined 1s",f(c["within_1_refined"]),"2s",f(c["within_2_refined"]),"| n=100 gated",cv.get("100",{}).get("runs"),f(cv.get("100",{}).get("within_1")),f(cv.get("100",{}).get("within_2")),"| rms/med",f(v.get("rms_over_median_at_stop")))
print("\nSPEEDUP: q_ratio (sweep fit q  vs adaptive) vs R_inf = 27W/(8 pi G)")
cases=sorted({tuple(k.split("|")[:2]) for k in R})
for g,n in cases:
    if n not in("SNR(5)","SNR(10)","SNR(20)"): continue
    def q(s):
        v=R.get(f"{g}|{n}|{s}"); 
        return (v.get("fit_true") or {}).get("q") if v else None
    v=R[f"{g}|{n}|Bayesian-SBED"]
    rinf=27*v["window"]/(8*math.pi*v["fwhm"])
    qs=q("SimpleSweep-3perFWHM"); qa=q("Bayesian-SBED")
    # direct information-rate ratio from the true signal not stored yet (stage_fisher_sweep absent): use window-average formula
    print(g.split('-')[2],n,"R_inf_pred",f(rinf),"1/R_inf",f(1/rinf),"q_sweep3",f(qs),"q_SBED",f(qa),"q_SmartSweep",f(q("Smart-Sweep")),"ratio SBED/sweep",f(qa/qs if qa and qs else None))
print("\nCOST: measured passage (true error stays <= tau) vs predicted n0+tau^-2/(q Imax)")
for k,v in R.items():
    g,n,s=k.split("|")
    if n not in("SNR(5)","SNR(10)","SNR(20)") or s not in ("Bayesian-SBED","Smart-Sweep","SimpleSweep-3perFWHM"): continue
    p=v.get("passage") or {}
    print(g.split('-')[2],n,s[:14],"tau",f(v["tau"]),"passage med",f(p.get("median")),"reached",f(p.get("reached")),"pred",f(v.get("predicted_cost")),"stop med",v["stopped_median"])
