import json, math
R = json.load(open("scratch/claims_results.json"))
def f(x, d=3):
    return "—" if x is None else (f"{x:.{d}g}" if isinstance(x,(int,float)) else str(x))
print("use case | strat | n_c=J0/Imax | n_half plateau drop tailslope | q_true[CI] n0 | q_claim n0 | claim/true | senseratio vs 1/sqrt(q) | window")
for k,v in R.items():
    g,n,s=k.split("|")
    if n not in("SNR(5)","SNR(10)","SNR(20)"): continue
    if s not in ("Bayesian-SBED","Smart-SBED","Smart-Sweep","SimpleSweep-3perFWHM"): continue
    r=v["regimes"]; ft=v.get("fit_true") or {}; fc=v.get("fit_claimed") or {}
    nc=v["J0"]/v["Imax"]
    qci = ft.get("q_ci"); 
    print(g.split('-')[2],n,s[:14],"nc",f(nc),"|",r["n_half"],f(r["plateau_ratio"]),f(r["drop_factor"]),f(r["tail_slope"]),"|",
          f(ft.get("q")), (f"[{qci[0]:.2g},{qci[1]:.2g}]" if qci else ""), f(ft.get("n0")),ft.get("window"),"|",f(fc.get("q")),f(fc.get("n0")),"|",
          (f(fc["q"]/ft["q"]) if fc.get("q") and ft.get("q") else "—"),"|",f(v.get("sens_ratio")),(f(1/math.sqrt(ft["q"])) if ft.get("q") else ""), "|", (ft.get("why") or "")[:40], r["why"][:1])
