with open("debug/benchmark_run.py", "r") as f:
    code = f.read()

search = """from nvision import (
    CoreExperiment,
    NVCenterCoreGenerator,
    nv_center_smc_belief,
)
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator

# Trace imports
t0 = time.perf_counter()
t_import = time.perf_counter() - t0"""

replace = """# Trace imports
t0 = time.perf_counter()

from nvision import (
    CoreExperiment,
    NVCenterCoreGenerator,
    nv_center_smc_belief,
)
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator

t_import = time.perf_counter() - t0"""

code = code.replace(search, replace)
with open("debug/benchmark_run.py", "w") as f:
    f.write(code)
