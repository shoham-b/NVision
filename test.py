import re
with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "r") as f:
    print([line for line in f if "staged_sobol" in line])
