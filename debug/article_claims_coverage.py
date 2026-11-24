"""Item 4: the article's own coverage numbers (``/api/claims``: Gaussian-onset-gated coverage) for SBED and the sweep
at every span of the pure-Lorentzian article grid, computed in-process with FastAPI's TestClient against the main
cache (read-only: no server, no reload, nothing written).

Usage: uv run python debug/article_claims_coverage.py <artifacts dir>
"""

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from nvision.cli.api_server import build_app
from nvision.sim.presets import snr_study_noises

run_dir = Path(sys.argv[1])
client = TestClient(build_app(cache_dir=run_dir / "cache", run_dir=run_dir))
spans = ["1.6", "2", "3", "5", "10", "20", "50"]
gens = ",".join(f"NVCenter-voigt-span{s}-gs0-hfunresolved" for s in spans)
noises = ",".join(n for n, _ in snr_study_noises())
r = client.get("/api/claims", params={"generator": gens, "noise": noises})
r.raise_for_status()
rows = r.json()
Path("scratch").mkdir(exist_ok=True)
Path("scratch/article_claims.json").write_text(json.dumps(rows))
print("strategy  span  noise  n  coverage_at_budget(1s,2s)  coverage_z_at_budget  gaussian_share  coverage(n=...)")
for row in rows:
    if row["strategy"] not in ("Bayesian-SBED", "SimpleSweep-optimal"):
        continue
    span = row["generator"].split("span")[1].split("-")[0]
    cab = row.get("coverage_at_budget")
    print(
        row["strategy"],
        span,
        row["noise"],
        row.get("repeats"),
        json.dumps(cab),
        json.dumps(row.get("coverage_z_at_budget")),
        row.get("gaussian_share"),
        json.dumps(row.get("coverage"))[:200],
    )
