
## 2024-05-24 - 2-Pass algorithm is faster than Welford's in Numba
**Learning:** When calculating weighted variance inside tight Numba `@njit` loops over fully in-memory arrays (like particle sets), a 2-pass algorithm (calculate mean, then variance) is significantly faster than 1-pass algorithms (like Welford's). This is because the 2-pass approach eliminates loop-carried data dependencies, allowing Numba to heavily optimize and SIMD-vectorize the operations. It also avoids the catastrophic cancellation that plagues the naive 1-pass sum-of-squares formula.
**Action:** Always prefer 2-pass variance calculations over Welford's algorithm inside Numba-jitted functions for fully in-memory arrays.
