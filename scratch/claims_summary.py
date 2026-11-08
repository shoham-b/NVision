import json, math
R = json.load(open("scratch/claims_results.json"))
def f(x, d=2):
    return "—" if x is None else (f"{x:.{d}g}" if isinstance(x,(int,float)) else str(x))
print("== prior gap / truth spread / alive90 (per use case, SBED rows) ==")
seen=set()
for k,v in R.items():
    g,n,s=k.split("|")
    if s!="Bayesian-SBED": continue
    print(g.split('-')[2], n, "gap",f(v["prior_gap"]),"truth/uniform",f(v["truth_spread_ratio"]),"sigma0/W_theta",f(v["sigma0"]*math.sqrt(v["J0"])/ (2*math.pi)*1) ,"alive90 until",v["alive90_until"],"max_n",v["max_n"], "stop med", v["stopped_median"], "censored", v["censored_share"])
print("== R true @n (SBED, Smart-Sweep, SimpleSweep-3) ==")
for k,v in R.items():
    g,n,s=k.split("|")
    if s not in ("Bayesian-SBED","Smart-Sweep","SimpleSweep-3perFWHM") or n not in("SNR(5)","SNR(20)"): continue
    print(g.split('-')[2], n, s, "R_true", {a:f(b) for a,b in v["R_true"].items() if b}, "R_cl", {a:f(b) for a,b in v["R_claimed"].items() if b})
