with open("nvision/runner/task_builder.py", "r") as f:
    code = f.read()

search = """        stats = df.group_by(["generator", "noise", "strategy"]).agg(pl.col("duration_ms").mean())
        per_combo = {(row["generator"], row["noise"], row["strategy"]): row["duration_ms"] for row in stats.to_dicts()}"""

replace = """        stats = df.group_by(["generator", "noise", "strategy"]).agg(pl.col("duration_ms").mean())
        per_combo = {(row["generator"], row["noise"], row["strategy"]): row["duration_ms"] for row in stats.to_dicts()}"""

# wait, I don't see any other filter calls in runner. I'll just keep the changes in executor.py, metrics.py and viz/__init__.py
