import json, statistics as st
R = json.load(open("scratch/claims_results_v2.json"))
def med(v): v=[x for x in v if x is not None]; return round(st.median(v),3) if v else None
by = {}
for k, o in R.items():
    g, n, s = k.split("|"); by.setdefault(s, []).append(o)
print("n combos", len(R), {s: len(v) for s, v in by.items()})
for s, L in by.items():
    L = [o for o in L if "unavailable" not in o]
    print("\n==", s, len(L))
    print(" prior_gap", med([o["prior_gap"] for o in L]), "truth_spread", med([o["truth_spread_ratio"] for o in L]))
    print(" bound_violated", sum(o["bound_violated"] for o in L), "/", len(L))
    for n in (1, 10, 100, 1000):
        print(f" R_true@{n} med", med([o["R_true"].get(str(n)) for o in L]), " R_claimed med", med([o["R_claimed"].get(str(n)) for o in L]),
              " max R_true", max((o["R_true"].get(str(n)) or 0) for o in L))
    rg = [o["regimes"] for o in L]
    print(" plateau", med([r["plateau_ratio"] for r in rg]), "n_half none", sum(r["n_half"] is None for r in rg), "drop", med([r["drop_factor"] for r in rg]), "tail slope", med([r["tail_slope"] for r in rg]), "tail has", sum(r["tail_slope"] is not None for r in rg))
    ft = [o["fit_true"] for o in L if o["fit_true"]]
    print(" fits", len(ft), "linear", sum(f["linear"] is True for f in ft), "nonlin", sum(f["linear"] is False for f in ft), "q med", med([f["q"] for f in ft]), "q lin-only med", med([f["q"] for f in ft if f["linear"]]), "n0 med", med([f["n0"] for f in ft]), "r2", med([f["r2"] for f in ft]))
    print(" rate_ratio claimed/true med", med([o["rate_ratio_claimed_over_true"] for o in L]), "n", sum(o["rate_ratio_claimed_over_true"] is not None for o in L), "sens ratio", med([o["sensitivity_ratio"] for o in L]))
    for n in (100, 1000):
        c = [o["coverage"].get(str(n)) for o in L]
        print(f" cov@{n} gated 1/2", med([x and x["within_1"] for x in c]), med([x and x["within_2"] for x in c]), "all", med([x and x["within_1_all"] for x in c]), med([x and x["within_2_all"] for x in c]), "runs gated med", med([x and x["runs_gated"] for x in c]))
    cb = [o["coverage_at_budget"] for o in L]
    print(" cov budget all", med([c["within_1_all"] for c in cb]), med([c["within_2_all"] for c in cb]), "gated", med([c["within_1"] for c in cb]), med([c["within_2"] for c in cb]), "refined share", med([o["refined_share"] for o in L]))
    z = [o["coverage_z_at_budget"] for o in L if o["coverage_z_at_budget"]]
    print(" z cov", len(z), med([x["within_1"] for x in z]), med([x["within_2"] for x in z]), "z std", med([x["std"] for x in z]), "z mean", med([x["mean"] for x in z]))
