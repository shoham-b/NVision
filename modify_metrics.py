with open("nvision/runner/metrics.py") as f:
    content = f.read()

func_code = """
def recalculate_cached_metrics(row: dict[str, Any]) -> None:
    \"\"\"Recalculate derived metrics (like pair_rmse) from existing cached fields.\"\"\"
    if "abs_err_x1" in row and "abs_err_x2" in row:
        err1 = float(row["abs_err_x1"])
        err2 = float(row["abs_err_x2"])
        row["pair_rmse"] = math.sqrt(0.5 * (err1 * err1 + err2 * err2))


def _scan_attempt_metrics(truth_positions: Sequence[float], estimate: dict[str, object]) -> dict[str, float]:
"""

content = content.replace(
    "def _scan_attempt_metrics(truth_positions: Sequence[float], estimate: dict[str, object]) -> dict[str, float]:",
    func_code,
)

with open("nvision/runner/metrics.py", "w") as f:
    f.write(content)
