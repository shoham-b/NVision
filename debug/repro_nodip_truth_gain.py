"""Repro: chi2 gain over flat at the TRUE parameters for SNR(1) SimpleSweep fits that the gate rejects."""

import sys

import numpy as np

from nvision.models import observer as _obs
from nvision.sim.locs.coarse import generic_sweep_locator as g

G = g.GenericSweepLocator
orig_req = G._require_fit_beats_flat_baseline
TRUTH = []
_oi = _obs.Observer.__init__


def _init(self, ts, *a, **k):
    TRUTH.append(ts)
    _oi(self, ts, *a, **k)


_obs.Observer.__init__ = _init


def req(self, model_ys, ys, popt, names):
    ts = TRUTH[-1]
    n = len(ys)
    du = self.history.drive_freqs_unit[:n]
    xs = self._domain_lo + du * (self._domain_hi - self._domain_lo)
    truth = np.asarray([float(ts(float(x), ts.baseline_drive)) for x in xs])
    noise = max(float(self._noise_std), 1e-12)
    flat = float(np.sum(((ys - ys.mean()) / noise) ** 2))
    chi_truth = float(np.sum(((truth - ys) / noise) ** 2))
    chi_fit = float(np.sum(((model_ys - ys) / noise) ** 2))
    print(f"GAIN truth={flat - chi_truth:7.1f} fit={flat - chi_fit:7.1f} n={n} (gate {g._FIT_MIN_CHI2_GAIN})", flush=True)
    return orig_req(self, model_ys, ys, popt, names)


G._require_fit_beats_flat_baseline = req
from nvision.__main__ import main  # noqa: E402

sys.argv = ["nv", "run-single", sys.argv[1], sys.argv[2], sys.argv[3], "--repeats", "10", "--runners", "1",
            "--dry-run", "--no-progress"]
main()
