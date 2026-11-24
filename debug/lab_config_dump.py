"""Dump the lab MATLAB BayesianESR config/state fields of one run (read-only)."""

import sys

import numpy as np
import scipy.io as sio

p = sys.argv[1]
mat = sio.loadmat(p, squeeze_me=True, struct_as_record=False, variable_names=["myStruct"])
esr = mat["myStruct"].BayesianESR
print("TOP fields:", esr._fieldnames)
for name in ("config",):
    c = getattr(esr, name)
    for f in c._fieldnames:
        v = getattr(c, f)
        if hasattr(v, "_fieldnames"):
            print(f"config.{f}: struct {v._fieldnames}")
            for g in v._fieldnames:
                print(f"   .{g} = {np.asarray(getattr(v, g)).ravel()[:6]}")
        else:
            a = np.asarray(v)
            print(f"config.{f} = {a.ravel()[:8] if a.size > 1 else v}")
s = esr.state
print("STATE fields:", s._fieldnames)
for f in s._fieldnames:
    a = np.asarray(getattr(s, f), dtype=object)
    print(f"state.{f}: shape {np.shape(getattr(s, f))}")
for f in esr._fieldnames:
    if f not in ("config", "state"):
        print("esr.", f, "=", str(getattr(esr, f))[:200])
