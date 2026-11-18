import json, math, statistics as st, re
R = json.load(open("scratch/claims_results_v2.json"))
rows = []
for k, o in R.items():
    g, n, s = k.split("|"); span = re.search(r"span([\d.]+)", g).group(1); snr = int(re.search(r"\((\d+)\)", n).group(1))
    rows.append((span, snr, s, o))
print("W/Gamma (drive window / fwhm) by span:", {sp: round(st.median(o["window_phys"]/o["fwhm_phys"] for sp2,_,_,o in rows if sp2==sp),2) for sp in sorted({r[0] for r in rows}, key=float)})
print("\nsweep q_fit / q_theory(8pi/27 * Gamma/W) per locator, median [min,max] over use cases")
for s in ["SimpleSweep-optimal", "SimpleSweep-optimal-x0.5", "SimpleSweep-optimal-x2", "Bayesian-SBED"]:
    v = [o["fit_true"]["q"] / (8*math.pi/27*o["fwhm_phys"]/o["window_phys"]) for sp,snr,s2,o in rows if s2==s and o.get("fit_true") and o["fit_true"]["q"]]
    print(f" {s}: {st.median(v):.2f} [{min(v):.2f},{max(v):.2f}]  n={len(v)}")
print("\nSBED: q, R_true@1000, cov1/cov2 at budget by SNR (median over spans)")
for snr in (1,2,5,10,20):
    L=[o for sp,sn,s,o in rows if s=="Bayesian-SBED" and sn==snr]
    if not L: continue
    print(f" SNR {snr} (n={len(L)}): q {st.median(o['fit_true']['q'] for o in L):.2f} Rtrue@1000 {st.median(o['R_true'].get('1000',0) for o in L):.2f}  R_claimed {st.median(o['R_claimed'].get('1000',0) for o in L):.2f} cov {st.median(o['coverage_at_budget']['within_1_all'] for o in L):.2f}/{st.median(o['coverage_at_budget']['within_2_all'] for o in L):.2f} z-std {st.median(o['coverage_z_at_budget']['std'] for o in L):.2f} n_c {st.median(o['n_c'] for o in L):.1f}")
print("\nSBED vs sweep-optimal q ratio per use case (median), and R_inf measured vs 1.07 W/Gamma")
by = {(sp,snr,s): o for sp,snr,s,o in rows}
rat=[]; pred=[]
for (sp,snr,s),o in by.items():
    if s=="Bayesian-SBED" and (sp,snr,"SimpleSweep-optimal") in by:
        sw=by[(sp,snr,"SimpleSweep-optimal")]
        rat.append(o["fit_true"]["q"]/sw["fit_true"]["q"]); pred.append(27/(8*math.pi)*o["window_phys"]/o["fwhm_phys"])
print(f" q_SBED/q_sweep median {st.median(rat):.2f} [{min(rat):.2f},{max(rat):.2f}]  vs predicted R_inf median {st.median(pred):.2f} [{min(pred):.2f},{max(pred):.2f}]")
print("\nregimes: n_half and plateau for SBED by span (median)")
for sp in sorted({r[0] for r in rows}, key=float):
    L=[o["regimes"] for s2,_,s,o in [(r[0],r[1],r[2],r[3]) for r in rows] if s=="Bayesian-SBED" and s2==sp]
    print(" span",sp, "n_half", st.median(r["n_half"] for r in L if r["n_half"]), "n_half=1 share", sum(r["n_half"]==1 for r in L), "of", len(L), "plateau", [round(r["plateau_ratio"],2) for r in L if r["plateau_ratio"]][:6], "tail", round(st.median(r["tail_slope"] for r in L if r["tail_slope"]),2))
print("\nfit window medians:", st.median(o["fit_window"][0] for *_,o in rows), st.median(o["fit_window"][1] for *_,o in rows))
print("share of SBED use cases with q CI upper bound <1:", sum(o["fit_true"]["q_ci"] and o["fit_true"]["q_ci"][1]<1 for sp,sn,s,o in rows if s=="Bayesian-SBED"),"/17")
print("\nq_SBED/q_sweep by span (positive q both):")
for sp in sorted({r[0] for r in rows}, key=float):
    v=[by[(s,snr,'Bayesian-SBED')]["fit_true"]["q"]/by[(s,snr,'SimpleSweep-optimal')]["fit_true"]["q"] for (s,snr,st_) in by if st_=='Bayesian-SBED' and s==sp and (s,snr,'SimpleSweep-optimal') in by and by[(s,snr,'SimpleSweep-optimal')]["fit_true"]["q"]>0]
    print(" span",sp, [round(x,2) for x in v], "predicted", round(27/(8*math.pi)*float(sp),2))
