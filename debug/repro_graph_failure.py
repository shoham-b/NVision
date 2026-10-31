"""Re-run graph generation for a .failed graph-queue state read-only (graphs go to scratch, nothing is cached)."""

import sys
import traceback
from pathlib import Path

from nvision.runner.graph_queue import _load_job
from nvision.runner.plots import generate_attempt_plots
from nvision.viz import Viz

path = Path(sys.argv[1])
job = _load_job(path)
out = Path("scratch/graph_repro")
out.mkdir(parents=True, exist_ok=True)
kw = job.combo_kw
print("slug", job.slug, "rid", job.rid)
try:
    generate_attempt_plots(
        viz=Viz(out / "graphs"),
        entry_base=job.entry_base,
        attempt_idx_in_combo=job.rid,
        current_scan=job.current_scan,
        current_history_df=job.history_df,
        noise_obj=job.noise_obj,
        strat_obj=job.strat_obj,
        slug_base=job.slug,
        out_dir=out,
        scans_dir=out / "graphs" / "scans",
        bayes_dir=out / "graphs" / "bayes",
        run_result=job.run_result,
        defer=False,
    )
    print("OK (no failure)")
except Exception:
    traceback.print_exc()
