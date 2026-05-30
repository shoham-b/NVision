import re

with open("nvision/runner/executor.py", "r") as f:
    code = f.read()

search = """import datetime
import logging
import random"""

replace = """import datetime
import logging
import math
import random"""

code = code.replace(search, replace)

with open("nvision/runner/executor.py", "w") as f:
    f.write(code)
print("executor OK")
