1. **Optimize O(N^2) Polars Dataframe filtering inside `Viz.plot_all_metrics`**
   - The method currently filters a polars DataFrame by unique combinations of `generator`, `noise`, and `strategy` inside a loop iterating over combinations. This results in O(N^2) complexity.
   - Replace it with a single `.partition_by(["generator", "noise", "strategy"], as_dict=True)` call outside the loop, reducing the complexity to O(N).

2. **Optimize O(N^2) Polars Dataframe filtering in `generate_attempt_metrics` and `_run_task` inside `executor.py`**
   - Similar to the visualization step, `executor.py` iterates over the repeats for a combination and delegates to `generate_attempt_metrics`. `generate_attempt_metrics` was running a `.filter()` to find rows associated with `repeat_id` from large histories and results DataFrames.
   - Extract `.partition_by("repeat_id", as_dict=True)` into `executor.py` outside the repeat loop, and fetch the corresponding partitions by `repeat_id` in O(1) time to pass down to `generate_attempt_metrics`.
   - `generate_attempt_metrics` will now check if `repeat_id` is present in the DataFrame before filtering to remain backwards-compatible, but take advantage of already-partitioned DataFrames when provided.

3. **Complete pre commit steps**
   - Run `pre_commit_instructions` tool to make sure proper formatting, testing, verifications, reviews and reflections are done.

4. **Submit PR**
   - Submit the PR with "⚡ Bolt: [performance improvement]" naming convention.
