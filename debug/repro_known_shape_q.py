"""Item 3: does knowing the line shape bring SBED's refine placement q toward 1? (dry-run: nothing is cached)

usage: [NVISION_MAX_MEASUREMENTS=n] [KS_CONTRAST_SCALE=x] uv run python debug/repro_known_shape_q.py
       <generator> <noise> <strategy> <repeats> <out.json>

Runs ``nv run-single ... --dry-run --runners 1`` in-process and records, per repeat: every measurement's oracle
Fisher efficiency ``eta = g(x)^2 / max g^2`` (``nvision.metrics.stages.primary_sensitivity_at``, the stored metrics'
own definition), its offset from the true line in half widths, the dip-detection step, the true error and claimed
sigma of ``zeeman_shift`` after every step, and the belief's contrast/width bounds (to check the known shape took).
Analyse with ``debug/analyse_known_shape_q.py``.
"""

import contextlib
import json
import sys

import numpy as np

from nvision.metrics.refine_rates import nearest_centre_offsets
from nvision.metrics.refine_rates_runs import line_centres_phys
from nvision.metrics.stages import primary_sensitivity_at
from nvision.runner import executor as _ex

GEN, NOISE, STRAT, REPS, OUT = sys.argv[1:6]
RECORDS: list[dict] = []
_orig = _ex._TaskRunner._run_single_repeat


def _capture(self, **kw):
    history_df, finalize_record, stop_reason, result = _orig(self, **kw)
    ts = result.true_signal
    snaps = result.snapshots
    belief0 = snaps[0].belief
    lo, hi = belief0.drive_freq_bounds_phys
    x_phys = np.array([lo + float(s.obs.drive_freq_unit) * (hi - lo) for s in snaps])
    sens = primary_sensitivity_at(ts, "zeeman_shift", lo, hi, x_phys, [None] * len(x_phys))
    eta = sens.measured / max(sens.best, float(sens.measured.max()))
    centres = line_centres_phys(ts, lo, hi)
    values = ts.parameter_values()
    hwhm = float(values.get("hwhm", values.get("hwhm_lorentz")))
    true_z = float(values["zeeman_shift"])
    err = [abs(float(s.belief.estimates()["zeeman_shift"]) - true_z) for s in snaps]
    unc = [float(s.belief.uncertainty()["zeeman_shift"]) for s in snaps]
    bounds = {k: list(map(float, v)) for k, v in belief0.physical_param_bounds.items() if not k.startswith("_")}
    RECORDS.append(
        {
            "eta": eta.tolist(),
            "offset_hwhm": nearest_centre_offsets(x_phys, centres, hwhm).tolist(),
            "err": err,
            "unc": unc,
            "bounds": bounds,
            "true": {k: float(v) for k, v in values.items()},
            "window_phys": float(hi - lo),
            "hwhm_phys": hwhm,
            "finalize": {k: v for k, v in finalize_record.items() if isinstance(v, (int, float, str)) or v is None},
        }
    )
    return history_df, finalize_record, stop_reason, result


_ex._TaskRunner._run_single_repeat = _capture

# KS_CONTRAST_SCALE=x (item 2): the known shape's contrast is x times the truth (a mis-calibrated lab contrast), the
# width stays true. Debug-only: the shipped strategy always uses the truth.
_SCALE = float(__import__("os").environ.get("KS_CONTRAST_SCALE", "1"))
if _SCALE != 1.0:
    _orig_known = _ex._TaskRunner._known_shape_prior_phys

    def _scaled(experiment):
        out = dict(_orig_known(experiment))
        c, std = out["contrast_total"]
        out["contrast_total"] = (c * _SCALE, std * _SCALE)
        return out

    _ex._TaskRunner._known_shape_prior_phys = staticmethod(_scaled)
from nvision.__main__ import main  # noqa: E402

sys.argv = ["nv", "run-single", GEN, NOISE, STRAT, "--repeats", REPS, "--runners", "1", "--dry-run", "--no-progress"]
with contextlib.suppress(SystemExit):
    main()
with open(OUT, "w") as f:
    json.dump({"generator": GEN, "noise": NOISE, "strategy": STRAT, "repeats": RECORDS}, f)
print(f"wrote {len(RECORDS)} repeats to {OUT}")
