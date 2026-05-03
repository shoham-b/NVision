import re

with open("tests/test_locator_benchmarks.py", "r") as f:
    content = f.read()

s1 = """    loc = locator_class.create(
        builder=builder,
        parameter_bounds=None,
        max_steps=12,
        initial_sweep_steps=4,
        noise_std=0.02,
        n_grid_freq=16,
        n_grid_width=8,
        n_grid_depth=8,
        n_grid_background=4,
        **extra,
    )"""

r1 = """    loc = locator_class.create(
        config=LocatorConfig(max_steps=12, initial_sweep_steps=4, noise_std=0.02),
        builder=builder,
        parameter_bounds=None,
        n_grid_freq=16,
        n_grid_width=8,
        n_grid_depth=8,
        n_grid_background=4,
        **extra,
    )"""

content = content.replace(s1, r1)
if "from nvision.models.locator import LocatorConfig" not in content:
    content = "from nvision.models.locator import LocatorConfig, ConvergenceConfig\n" + content

with open("tests/test_locator_benchmarks.py", "w") as f:
    f.write(content)

with open("tests/test_unit_cube_belief.py", "r") as f:
    content = f.read()

s2 = """    locator = SequentialBayesianExperimentDesignLocator.create(
        builder=nv_center_smc_belief,
        max_steps=100,
        convergence_threshold=0.01,
        scan_param="frequency",
        parameter_bounds=bounds,
        initial_sweep_steps=0,
        noise_std=0.01,
        signal_model=model,
    )"""
r2 = """    locator = SequentialBayesianExperimentDesignLocator.create(
        config=LocatorConfig(max_steps=100, noise_std=0.01, initial_sweep_steps=0, convergence=ConvergenceConfig(threshold=0.01)),
        builder=nv_center_smc_belief,
        scan_param="frequency",
        parameter_bounds=bounds,
        signal_model=model,
    )"""
content = content.replace(s2, r2)
if "from nvision.models.locator import LocatorConfig" not in content:
    content = "from nvision.models.locator import LocatorConfig, ConvergenceConfig\n" + content

with open("tests/test_unit_cube_belief.py", "w") as f:
    f.write(content)
