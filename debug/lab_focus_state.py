"""Final focus-test state of the lab runs per window (read-only): why the focus window never locks in wide windows."""

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

folder = Path(sys.argv[1])
F = [
    "focusLocked",
    "focusJointLogTail",
    "focusMass",
    "focusBinomialTail",
    "focusAdjustedTail",
    "focusCandidateMass",
    "focusSupport",
    "noiseVariance",
    "ess",
]
for p in sorted(folder.glob("Run_*MHz_1.mat")) + sorted(folder.glob("Run_*MHz_2.mat")):
    esr = sio.loadmat(p, squeeze_me=True, struct_as_record=False, variable_names=["myStruct"])["myStruct"].BayesianESR
    s = esr.state
    out = {f: np.round(np.asarray(getattr(s, f), dtype=float).ravel(), 5).tolist() for f in F}
    cands = s.focusCandidatesHistoryMHz
    ncand = [np.size(c) // 2 for c in cands]
    print(p.name, out, "ncand_last20:", ncand[-20:], "n_with_cand:", sum(n > 0 for n in ncand))
