import re

files_to_update = [
    "nvision/sim/locs/bayesian/sbed_locator.py",
    "nvision/sim/locs/bayesian/maximum_likelihood_locator.py",
    "nvision/sim/locs/bayesian/utility_sampling_locator.py",
    "nvision/sim/locs/bayesian/students_t_locator.py",
    "nvision/sim/locs/bayesian/sequential_bayesian_locator.py",
    "nvision/models/locator.py"
]
# I will use diff blocks to do this.
