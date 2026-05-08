## YYYY-MM-DD - [Parallelize Numba SBED Entropy]
**Learning:** Numba `prange` allows parallelizing outer candidate loops effectively, and thread-local scratchpad arrays like `buffer` must be allocated inside the `prange` loop to avoid race conditions. Additionally, expensive math operations like `math.log` inside tight inner loops over particles can be avoided via algebraic identities (e.g. `log(w) = ll - max_ll - log_sum_exp`).
**Action:** Always look for nested loops with independent evaluations (e.g., candidate scoring) to parallelize with `@njit(parallel=True)` and `prange`, taking care to define buffers locally inside the loop. Avoid mathematically redundant slow math inside innermost loops.
## 2024-05-08 - [O(MN) Grouping to O(N) Loop in Measurement UI]
**Learning:** Filtering a dataset repeatedly using list comprehensions for different categorical groups causes redundant O(N) iterations, leading to an overall O(MN) complexity where M is the number of categories.
**Action:** Always replace multiple list comprehensions over the same zipped sequences with a single pass `for` loop that distributes the elements to their target collections simultaneously.
