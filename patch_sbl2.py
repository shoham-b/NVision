import re

with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "r") as f:
    content = f.read()

# SequentialBayesianLocator used to set _full_domain_lo
# Let's check where it used to be set by running git blame or git show
