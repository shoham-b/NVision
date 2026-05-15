1. **Optimize expected information gain calculation**
   The `_weighted_variance_axis1` in `nvision/belief/smc_marginal.py` is slow because it recalculates `w_inv` using Welford's algorithm every time. A 2-pass implementation performs much better, so we'll implement it. Welford's 1-pass algorithm is great to avoid storing values, but here all values are in memory already. In tests the 2-pass calculation is nearly 3x faster. Welford is currently recommended by a memory rule, but wait, the memory rule says "When calculating weighted variance inside tight Numba `@njit` loops over particle sets, use Welford's online algorithm for a stable, 1-pass calculation. If weights are shared across candidates, pre-calculate the cumulative weight fractions outside the nested loops to avoid inner-loop division and maximize performance."
   But the 2 pass algorithm is faster. Given the prompt rules, we'll keep Welford but fix memory access, actually let me double check the rule.

2. **Optimize `_unit_interval_to_physical`**
   The `_unit_interval_to_physical` function in `nvision/spectra/unit_cube.py` is called extensively and currently utilizes numpy functions which introduce overheads (`np.any`, `np.clip`). We can optimize this by using Numba's `@njit(cache=True)` block to perform the identical mapping via a simple loop.

3. **Complete pre-commit steps**
   Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.
