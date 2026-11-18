"""Read-only: article-claim numbers on the cached voigt SNR x window study -> scratch/claims_results_v2.json.
Usage: uv run python scratch/claims_v2_run.py <physics_fingerprint> <out.json>"""
import json, sys
from pathlib import Path
from nvision.cache import CacheBridge
from nvision.cli.api_server import _combo_key, _scan_field_row
from nvision.metrics.claims_report import analyse_use_case
from nvision.sim.combinations import CombinationGrid

fp, out_path = sys.argv[1], sys.argv[2]
b = CacheBridge(Path(r"C:/Users/shoha/git/NVision/artifacts/cache"))
res = {}
for combo in sorted(b.list_combinations_with_updated_at(), key=lambda c: (c["generator"], c["noise"], c["strategy"])):
    if combo["physics_fingerprint"] != fp or combo["repeats"] < 10:
        continue
    repo = b.get_cache_for_category(CombinationGrid.generator_category(combo["generator"]))
    meta = repo._repeats.load_repeats_meta(_combo_key(combo), int(combo["repeats"])) or repo._repeats.load_repeats(_combo_key(combo), int(combo["repeats"]))
    rows = [r for e, m in meta if (r := _scan_field_row(combo, e, m)) is not None]
    key = f'{combo["generator"]}|{combo["noise"]}|{combo["strategy"]}'
    res[key] = analyse_use_case(rows)
    print(key, len(rows), file=sys.stderr, flush=True)
b.close()
json.dump(res, open(out_path, "w"), indent=1, default=float)
