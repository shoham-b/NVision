import re

with open("tests/test_runner_cache.py", "r") as f:
    code = f.read()

search = """datetime.datetime.now().isoformat()"""

replace = """datetime.datetime.now(datetime.timezone.utc).isoformat()"""

code = code.replace(search, replace)
with open("tests/test_runner_cache.py", "w") as f:
    f.write(code)
print("runner_cache OK")
