## 2024-05-18 - Avoid Polars .filter in Loops
**Learning:** Polars `.filter()` inside loops leads to O(N^2) complexity, significantly degrading performance on large datasets.
**Action:** Use `.partition_by(..., as_dict=True)` to convert the DataFrame into O(1) hash map lookups, turning the O(N^2) operation into O(N). Note that `partition_by` always returns a dictionary where keys are tuples, even if partitioning by a single column.
