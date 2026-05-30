with open("nvision/viz/__init__.py", "r") as f:
    code = f.read()

import re

search = """        for (gen, noise, strat), subset in partitions.items():"""

replace = """        for (gen, noise, _strat), subset in partitions.items():"""

code = code.replace(search, replace)

with open("nvision/viz/__init__.py", "w") as f:
    f.write(code)
