"""Log ESS / unique particles / primary sigma at every SMC resample of a dry-run repeat.

usage: uv run python debug/repro_resample_ess.py <generator> <noise> <strategy> [repeats]
Writes one row per resample to scratch/resample_log.csv (repeat = restart of obs count).
"""

import sys

import numpy as np

from nvision.belief import smc_marginal as sm

C = sm.SMCMarginalDistribution
orig = C._resample
ROWS = []
_rep = [0]
_last_n = [10**9]


def patched(self):
    n_obs = int(self._obs_count)
    if n_obs < _last_n[0]:
        _rep[0] += 1
    _last_n[0] = n_obs
    w = np.asarray(self._weights, dtype=np.float64)
    ess = float(1.0 / np.sum(w * w))
    n = self.num_particles
    pre_unique = len(np.unique(np.asarray(self._particles).round(9), axis=0))
    pre_sigma = float(self.uncertainty()["zeeman_shift"]) if "zeeman_shift" in self._param_names else float("nan")
    orig(self)
    post_unique = len(np.unique(np.asarray(self._particles).round(9), axis=0))
    post_sigma = float(self.uncertainty()["zeeman_shift"]) if "zeeman_shift" in self._param_names else float("nan")
    ROWS.append((_rep[0], n_obs, n, ess, pre_unique, pre_sigma, post_unique, post_sigma))


C._resample = patched
from nvision.__main__ import main  # noqa: E402


def _dump():
    with open("scratch/resample_log.csv", "w") as f:
        f.write("repeat,n_obs,n_particles,ess,pre_unique,pre_sigma,post_unique,post_sigma\n")
        for r in ROWS:
            f.write(",".join(str(x) for x in r) + "\n")


import atexit  # noqa: E402

atexit.register(_dump)
reps = sys.argv[4] if len(sys.argv) > 4 else "6"
sys.argv = ["nv", "run-single", sys.argv[1], sys.argv[2], sys.argv[3], "--repeats", reps, "--runners", "1", "--dry-run", "--no-progress"]
main()
