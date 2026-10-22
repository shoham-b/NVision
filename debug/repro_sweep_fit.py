"""Repro: SimpleSweep's final fit lands at the zeeman_shift bound midpoint."""

import sys

import numpy as np

from nvision.sim.locs.coarse import generic_sweep_locator as g

G = g.GenericSweepLocator
orig_run = G._run_curve_fit_candidates
orig_seed = G._seed_zero_field_freq_and_splits
orig_fin = G.finalize


def seed(self, du, ys, sm, contrast, dlo, dw, zs_idx, hf_idx, lo, hi):
    out = orig_seed(self, du, ys, sm, contrast, dlo, dw, zs_idx, hf_idx, lo, hi)
    print("contrast", contrast, "argmin offset MHz", (du[np.argmin(ys)] * dw) / 1e6, "noise", self._noise_std)
    print("SEEDS", [(a - dlo, b, c) for a, b, c in out])
    return out


def run(curve_fn, x, ys, cands, make_p0, lo, hi, xtol, **kw):
    popt, pcov = orig_run(curve_fn, x, ys, cands, make_p0, lo, hi, xtol, **kw)
    print("n_cands", len(cands), "p0[0]", make_p0(*cands[0]), "popt", popt)
    return popt, pcov


G._seed_zero_field_freq_and_splits = seed
G._run_curve_fit_candidates = staticmethod(run)
from nvision.models import observer as _obs  # noqa: E402

TRUTH = []
_oi = _obs.Observer.__init__


def _init(self, ts, *a, **k):
    TRUTH.append(ts)
    _oi(self, ts, *a, **k)


_obs.Observer.__init__ = _init


def fin(self):
    orig_fin(self)
    tp = TRUTH[-1].typed_parameters
    est = self._fit_params_phys["zeeman_shift"]
    tr = float(tp.zeeman_shift)
    print(f"ERR zeeman est {est / 1e6:.3f} true {tr / 1e6:.3f} err {(est - tr) / 1e3:.1f} kHz")


G.finalize = fin
from nvision.__main__ import main  # noqa: E402

sys.argv = [
    "nv",
    "run-single",
    "NVCenter-voigt-w0.50MHz-c0.10-si0.30MHz-hfunresolved",
    "Gauss(0.004)",
    "SimpleSweep-3perFWHM",
    "--repeats",
    "8",
    "--runners",
    "1",
    "--dry-run",
    "--no-progress",
]
main()
