from pathlib import Path
from nvision.cache import CacheBridge
from nvision.cli.api_server import _combo_key
from nvision.sim.combinations import CombinationGrid
b = CacheBridge(Path(r"C:/Users/shoha/git/NVision/artifacts/cache"))
c = next(x for x in b.list_combinations_with_updated_at() if x["generator"]=="NVCenter-voigt-span1.6-gs0-hfunresolved" and x["noise"]=="SNR(10)" and x["strategy"]=="Bayesian-SBED")
repo = b.get_cache_for_category(CombinationGrid.generator_category(c["generator"]))
meta = repo._repeats.load_repeats_meta(_combo_key(c), int(c["repeats"]))
for ents,row in meta[:6]:
    scan = next(e for e in ents if e["type"]=="scan")
    s=scan["series"]; print(row.get("stage_true_passage_step"), row.get("stage_tau_phys"), s["e"][:6], scan["true_params"]["params"]["zeeman_shift"], scan["true_params"]["bounds"]["zeeman_shift"])
import numpy as np
ps=[m.get("stage_true_passage_step") for e,m in meta]; print("stored passage steps", sorted([p for p in ps if p is not None])[:30], "n None", sum(p is None for p in ps))
print("first-step errors / tau:", np.median([next(e for e in ents if e['type']=='scan')['series']['e'][0] for ents,m in meta]), "tau", meta[0][1]['stage_tau_phys'])
b.close()
