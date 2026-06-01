## 2024-05-18 - Avoid O(N^2) complexity with repeated Polars DataFrame filtering
**Learning:** When separating a Polars DataFrame into sub-dataframes by unique values in a column, repeated `.filter()` operations inside a loop result in O(N^2) time complexity.
**Action:** Use `.partition_by(columns, as_dict=True)` to partition the dataframe into a dictionary mapping keys (tuples) to dataframes, achieving O(N) performance. Ensure that unused variables in loop comprehensions are prefixed with an underscore to avoid linting errors.
