import numpy as np
import time
from nvision.spectra.numba_kernels import nv_center_lorentzian_vectorized_many
from numba import njit, prange

def test_perf():
    n_x = 8
    n_freq = 1024

    xs = np.random.rand(n_x)
    freq = np.random.rand(n_freq)
    linewidth = np.random.rand(n_freq)
    split = np.random.rand(n_freq)
    k_np = np.random.rand(n_freq)
    dip_depth = np.random.rand(n_freq)
    background = np.ones(n_freq)
    out = np.empty((n_x, n_freq))

    # warm up
    nv_center_lorentzian_vectorized_many(xs, freq, linewidth, split, k_np, dip_depth, background, out)

    t0 = time.time()
    for _ in range(1000):
        nv_center_lorentzian_vectorized_many(xs, freq, linewidth, split, k_np, dip_depth, background, out)
    t1 = time.time()

    print(f"Time for nv_center_lorentzian_vectorized_many: {(t1 - t0) * 1000:.2f} ms")

if __name__ == "__main__":
    test_perf()
