import numpy as np
import time
from numba import njit, prange

n_cands = 2000
n_parts = 10000
pred = np.random.rand(n_cands, n_parts).astype(np.float32)
w = (np.ones(n_parts) / n_parts).astype(np.float32)

@njit(parallel=True, cache=True, fastmath=True)
def weighted_var_numba(pred, w):
    m = pred.shape[0]
    n = pred.shape[1]
    out = np.empty(m, dtype=pred.dtype)
    for i in prange(m):
        sum_p = 0.0
        sum_p2 = 0.0
        for j in range(n):
            val = pred[i, j]
            wi = w[j]
            sum_p += wi * val
            sum_p2 += wi * val * val
        # Clip negative values due to numerical precision
        v = sum_p2 - sum_p * sum_p
        out[i] = v if v > 0.0 else 0.0
    return out

# Warm up
weighted_var_numba(pred, w)

# Numba time
t0 = time.perf_counter()
res = weighted_var_numba(pred, w)
t_numba = time.perf_counter() - t0
print(f"Numba approach took:    {t_numba:.6f} s")

# Identity time
t0 = time.perf_counter()
mean = pred @ w
res2 = (pred**2) @ w - mean**2
res2 = np.maximum(res2, 0.0)
t_identity = time.perf_counter() - t0
print(f"Identity approach took: {t_identity:.6f} s")

print(f"Max difference:        {np.max(np.abs(res - res2)):.2e}")
