import re

with open("nvision/runner/executor.py", "r") as f:
    code = f.read()

search = """            else:
                min_linewidth = 200e3  # fallback
            return int(np.ceil(domain_width / min_linewidth))"""

replace = """            else:
                min_linewidth = 200e3  # fallback
            return int(math.ceil(domain_width / min_linewidth))"""

code = code.replace(search, replace)
with open("nvision/runner/executor.py", "w") as f:
    f.write(code)
print("executor OK")
