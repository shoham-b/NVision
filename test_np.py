import numpy as np
import time
from numba import njit
from nvision.belief.smc_marginal import _weighted_mean_variance_1d as numba_2_pass

# Let's benchmark numpy dot vs numba 2-pass
x = np.ones(10000) * 1000000.0 + np.random.rand(10000) * 0.0001
w = np.random.rand(10000)

@njit(cache=True)
def _weighted_mean_variance_1d_np(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    n = x.shape[0]
    if n == 0:
        return 0.0, 0.0

    sum_weight = np.sum(w)

    if sum_weight <= 0.0:
        return 0.0, 0.0

    mean = np.dot(x, w) / sum_weight

    var_sum = 0.0
    for i in range(n):
        diff = x[i] - mean
        var_sum += w[i] * diff * diff

    return float(mean), float(var_sum / sum_weight)

@njit(cache=True)
def _weighted_mean_variance_1d_np2(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    n = x.shape[0]
    if n == 0:
        return 0.0, 0.0

    sum_weight = np.sum(w)

    if sum_weight <= 0.0:
        return 0.0, 0.0

    mean = np.dot(x, w) / sum_weight

    # can we do np.dot for variance too?
    diff = x - mean
    var_sum = np.dot(diff * diff, w)

    return float(mean), float(var_sum / sum_weight)

@njit(cache=True)
def _weighted_mean_variance_1d_np3(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    n = x.shape[0]
    if n == 0:
        return 0.0, 0.0

    sum_weight = np.sum(w)

    if sum_weight <= 0.0:
        return 0.0, 0.0

    mean = np.dot(x, w) / sum_weight

    diff = x - mean

    var_sum = 0.0
    for i in range(n):
        var_sum += w[i] * diff[i] * diff[i]

    return float(mean), float(var_sum / sum_weight)


numba_2_pass(x, w)
_weighted_mean_variance_1d_np(x, w)
_weighted_mean_variance_1d_np2(x, w)
_weighted_mean_variance_1d_np3(x, w)

times = {"numba_2_pass": 0.0, "np": 0.0, "np2": 0.0, "np3": 0.0}
for _ in range(10000):
    t0 = time.perf_counter()
    numba_2_pass(x, w)
    times["numba_2_pass"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    _weighted_mean_variance_1d_np(x, w)
    times["np"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    _weighted_mean_variance_1d_np2(x, w)
    times["np2"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    _weighted_mean_variance_1d_np3(x, w)
    times["np3"] += time.perf_counter() - t0


print("welford_res:", numba_2_pass(x, w))
print("np:", _weighted_mean_variance_1d_np(x, w))
print("np2:", _weighted_mean_variance_1d_np2(x, w))
print("np3:", _weighted_mean_variance_1d_np3(x, w))

print("Times:")
print(times)
