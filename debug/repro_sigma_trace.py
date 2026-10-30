"""Per-step trace of sigma, true error and the belief's own CRLB for the primary param (zeeman_shift).

usage: uv run python debug/repro_sigma_trace.py <generator> <noise> <strategy> [repeats]
Writes scratch/sigma_trace.csv (dry-run: nothing is cached).
"""

import atexit
import os
import sys
import traceback

from nvision.belief import smc_marginal as sm
from nvision.models import observer as _obs

C = sm.SMCMarginalDistribution
ROWS = []
TRUTH = []
_rep = [-1]
_oi = _obs.Observer.__init__


def _init(self, ts, *a, **k):
    TRUTH.append(ts)
    _rep[0] += 1
    _oi(self, ts, *a, **k)


_obs.Observer.__init__ = _init
orig = C.batch_update

# ORACLE=hwhm|both collapses the named line-width bounds onto the true value (+-0.1 %), so the belief cannot be
# wrong about them. ORACLE_OUT picks the csv path. Debug-only: this never touches the shipped algorithm.
ORACLE = os.environ.get("ORACLE", "none")
OUT = os.environ.get("ORACLE_OUT", "scratch/sigma_trace.csv")
if ORACLE != "none":
    from nvision.runner import executor as _ex

    _orig_bounds = _ex._TaskRunner._injected_parameter_bounds
    _fixed = ("hwhm_lorentz",) if ORACLE == "hwhm" else ("hwhm_lorentz", "sigma_inhom")

    def _oracle_bounds(experiment):
        bounds = dict(_orig_bounds(experiment))
        tp = experiment.true_signal.typed_parameters
        for name in _fixed:
            if name in bounds:
                v = float(getattr(tp, name))
                half = max(abs(v) * 1e-3, 1.0)
                bounds[name] = (v - half, v + half)
        return bounds

    _ex._TaskRunner._injected_parameter_bounds = staticmethod(_oracle_bounds)


_BELIEFS: dict[int, tuple[int, object, str, float]] = {}


def patched(self, observations):
    orig(self, observations)
    n = int(self._obs_count)
    key = id(self)
    if key not in _BELIEFS or n <= len(observations):
        stack = traceback.extract_stack(limit=25)
        tag = (
            "sbed"
            if any("sbed_locator" in f.filename for f in stack)
            else ("sweep" if any("generic_sweep_locator" in f.filename for f in stack) else "other")
        )
        _ts = TRUTH[-1]
        _pm, _ps = _ts.bounds["_priors"]["zeeman_shift"]  # the generator's stated prior (mean, std), physical Hz
        _off = (float(_pm) - float(getattr(_ts.typed_parameters, "zeeman_shift"))) / float(_ps)
        _BELIEFS[key] = (len(_BELIEFS), _ts, tag, _off)
    bid, ts, tag, prior_off = _BELIEFS[key]
    if n % 5 and n > 20:
        return
    true_z = float(getattr(ts.typed_parameters, "zeeman_shift"))
    est = self.estimates()["zeeman_shift"]
    sig = float(self.uncertainty()["zeeman_shift"])
    crlb = self.crlb_per_param().get("zeeman_shift", float("nan"))
    noise_mean = self.estimates()["noise_sigma"]
    noise_p90 = self.estimated_noise_std()
    ROWS.append((bid, tag, n, true_z, est, abs(est - true_z), sig, crlb, noise_mean, noise_p90, prior_off))


C.batch_update = patched
from nvision.__main__ import main  # noqa: E402


def _dump():
    with open(OUT, "w") as f:
        f.write("belief,tag,n_obs,true_z,est,abs_err,sigma,crlb,noise_mean,noise_p90,prior_off\n")
        for r in ROWS:
            f.write(",".join(str(x) for x in r) + "\n")


atexit.register(_dump)
reps = sys.argv[4] if len(sys.argv) > 4 else "8"
sys.argv = ["nv", "run-single", sys.argv[1], sys.argv[2], sys.argv[3], "--repeats", reps, "--runners", "1", "--dry-run", "--no-progress"]
main()
