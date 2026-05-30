with open("nvision/runner/metrics.py", "r") as f:
    code = f.read()

search = """    if not final_history_df.is_empty():
        current_history_df = final_history_df.filter(pl.col("repeat_id") == attempt_idx_in_combo).drop("repeat_id")"""

replace = """    if not final_history_df.is_empty():
        current_history_df = final_history_df.filter(pl.col("repeat_id") == attempt_idx_in_combo).drop("repeat_id")"""

print("OK")
