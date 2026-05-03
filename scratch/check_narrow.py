
import random
import numpy as np
from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator

def check_narrow():
    # Narrow variant parameters from updated presets.py
    x_min=2.86e9
    x_max=2.88e9
    gen = NVCenterCoreGenerator(x_min=x_min, x_max=x_max, variant="lorentzian", center_freq_fraction=0.1, narrow_signal=True)

    rng = random.Random(42)
    signal = gen.generate(rng)

    print(f"Domain width: {(x_max - x_min)/1e6} MHz")
    print(f"Generated params: {signal.typed_parameters}")

    for param, (lo, hi) in signal.bounds.items():
        if param.startswith("_"): continue
        val = getattr(signal.typed_parameters, param)
        if val < lo or val > hi:
            print(f"WARNING: {param} value {val/1e6:.3f} MHz is outside bounds [{lo/1e6:.3f}, {hi/1e6:.3f}] MHz")
        else:
            print(f"OK: {param} value {val/1e6:.3f} MHz is within bounds [{lo/1e6:.3f}, {hi/1e6:.3f}] MHz")

if __name__ == "__main__":
    check_narrow()
