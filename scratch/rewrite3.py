with open("nvision/runner/metrics.py", "r") as f:
    code = f.read()

search1 = """    if not final_history_df.is_empty():
        current_history_df = final_history_df.filter(pl.col("repeat_id") == attempt_idx_in_combo).drop("repeat_id")
    else:"""

replace1 = """    if not final_history_df.is_empty():
        current_history_df = final_history_df.filter(pl.col("repeat_id") == attempt_idx_in_combo).drop("repeat_id")
    else:"""

code = code.replace(search1, replace1)

with open("nvision/runner/metrics.py", "w") as f:
    f.write(code)
