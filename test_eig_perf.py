import numpy as np
import time
from nvision.belief.smc_marginal import _weighted_variance_axis1
from numba import njit, prange

def test_perf():
    n_candidates = 8
    n_particles = 1024

    predictions = np.random.rand(n_candidates, n_particles)
    weights = np.random.rand(n_particles)
    weights /= weights.sum()

    # warm up
    _weighted_variance_axis1(predictions, weights)

    t0 = time.time()
    for _ in range(1000):
        _weighted_variance_axis1(predictions, weights)
    t1 = time.time()

    print(f"Time for _weighted_variance_axis1: {(t1 - t0) * 1000:.2f} ms")

if __name__ == "__main__":
    test_perf()
