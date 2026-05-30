import re

with open("nvision/runner/executor.py", "r") as f:
    code = f.read()

search = """import datetime
import logging
import queue
import random"""

replace = """import datetime
import logging
import math
import queue
import random"""

code = code.replace(search, replace)

search2 = """import logging
import math
import queue
import random
import sys"""

replace2 = """import logging
import math
import queue
import random
import sys"""

code = code.replace(search2, replace2)

with open("nvision/runner/executor.py", "w") as f:
    f.write(code)
print("executor OK")
