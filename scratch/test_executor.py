import nvision.runner.executor as ex
with open("nvision/runner/executor.py", "r") as f:
    code = f.read()

import re
matches = re.findall(r'def generate_attempt_metrics', code)
print(matches)
