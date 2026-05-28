import numpy as np
import time
from numba import njit

@njit(cache=True)
def welford_1d(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    n = x.shape[0]
    if n == 0:
        return 0.0, 0.0

    mean = 0.0
    S = 0.0  # noqa: N806
    sum_weight = 0.0
    for i in range(n):
        wi = w[i]
        xi = x[i]
        sum_weight += wi
        if sum_weight > 0.0:
            delta = xi - mean
            mean += (wi / sum_weight) * delta
            S += wi * delta * (xi - mean)  # noqa: N806

    if sum_weight <= 0.0:
        return 0.0, 0.0

    return float(mean), float(S / sum_weight)


@njit(cache=True)
def twopass_1d(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    n = x.shape[0]
    if n == 0:
        return 0.0, 0.0

    mean = 0.0
    sum_weight = 0.0
    for i in range(n):
        wi = w[i]
        mean += wi * x[i]
        sum_weight += wi

    if sum_weight <= 0.0:
        return 0.0, 0.0

    mean /= sum_weight

    S = 0.0  # noqa: N806
    for i in range(n):
        delta = x[i] - mean
        S += w[i] * delta * delta

    return float(mean), float(S / sum_weight)

# JIT compile
np.random.seed(0)
x = np.random.rand(1000)
w = np.random.rand(1000)
welford_1d(x, w)
twopass_1d(x, w)

N = 1000000
x = np.random.rand(N)
w = np.random.rand(N)

t0 = time.perf_counter()
for _ in range(100):
    m1, v1 = welford_1d(x, w)
t1 = time.perf_counter()
print(f"Welford: {t1 - t0:.5f}s (mean={m1}, var={v1})")

t0 = time.perf_counter()
for _ in range(100):
    m2, v2 = twopass_1d(x, w)
t1 = time.perf_counter()
print(f"2-Pass:  {t1 - t0:.5f}s (mean={m2}, var={v2})")

assert np.allclose(m1, m2)
assert np.allclose(v1, v2)
