## 2024-05-18 - 2-pass weighted variance avoids data dependencies

**Learning:** In Numba @njit tight loops, Welford's algorithm and single-pass sum of squares variants for weighted variance create loop-carried dependencies that block vectorization and optimization. Given that SMC particles are fully in memory, a 2-pass algorithm (calculate mean, then calculate variance) is substantially faster (~10-20x) and maintains good numerical stability for small variances compared to the naive single-pass implementation.
**Action:** Always prefer 2-pass mean and variance calculations inside Numba loops over in-memory particle arrays for Bayesian locators.
