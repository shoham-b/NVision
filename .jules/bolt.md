## 2026-05-15 - Shifted 1-pass variance calculation in Numba
**Learning:** When calculating weighted variance inside tight Numba `@njit` loops over particle sets, a naive 1-pass algorithm ($V = E[X^2] - (E[X])^2$) is highly susceptible to catastrophic cancellation and loss of precision when the variance is small relative to the mean.
**Action:** Use a shifted 1-pass algorithm (shifting by the first element, e.g., `x[0]`) to maintain a stable 1-pass calculation without the overhead of the full 2-pass algorithm.
