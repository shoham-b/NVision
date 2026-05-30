import re

with open("nvision/runner/metrics.py", "r") as f:
    content = f.read()

# I want to change these lines:
#     if not final_history_df.is_empty():
#         current_history_df = final_history_df.filter(pl.col("repeat_id") == attempt_idx_in_combo).drop("repeat_id")
# ...
#     finalize_row = finalize_results.filter(pl.col("repeat_id") == attempt_idx_in_combo)

# to:
#     if not final_history_df.is_empty():
#         parts = final_history_df.partition_by("repeat_id", as_dict=True)
#         current_history_df = parts.get((attempt_idx_in_combo,), pl.DataFrame()).drop("repeat_id") if (attempt_idx_in_combo,) in parts else pl.DataFrame()
# ...
