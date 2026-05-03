import os
import glob
import re

test_files = glob.glob("tests/test_*.py")

for filename in test_files:
    with open(filename, "r") as f:
        content = f.read()

    modified = False

    # 1. loc = locator_class.create(**config) -> loc = locator_class.create(config=LocatorConfig(), **config)
    if "locator_class.create(**config)" in content:
        content = content.replace("locator_class.create(**config)", "locator_class.create(config=LocatorConfig(), **config)")
        modified = True

    # 2. loc = locator_class.create(\n        builder=builder,\n        parameter_bounds=None,\n        max_steps=12,\n        initial_sweep_steps=4,\n        noise_std=0.02,\n        **extra,\n    )
    # let's do this specifically for test_locator_benchmarks.py
    if filename.endswith("test_locator_benchmarks.py"):
        search = """    loc = locator_class.create(
        builder=builder,
        parameter_bounds=None,
        max_steps=12,
        initial_sweep_steps=4,
        noise_std=0.02,
        **extra,
    )"""
        replace = """    loc = locator_class.create(
        config=LocatorConfig(max_steps=12, initial_sweep_steps=4, noise_std=0.02),
        builder=builder,
        parameter_bounds=None,
        **extra,
    )"""
        if search in content:
            content = content.replace(search, replace)
            modified = True

    # 3. SimpleSweepLocator.create(belief=belief, signal_model=model, max_steps=10)
    search2 = "SimpleSweepLocator.create(belief=belief, signal_model=model, max_steps=10)"
    replace2 = "SimpleSweepLocator.create(config=LocatorConfig(max_steps=10), belief=belief, signal_model=model)"
    if search2 in content:
        content = content.replace(search2, replace2)
        modified = True

    # 4. SobolSweepLocator.create(
    #             belief=belief,
    #             signal_model=model,
    #             max_steps=max_steps,
    #             domain_lo=domain_lo,
    #             domain_hi=domain_hi,
    #         )
    search3 = """        locator = SobolSweepLocator.create(
            belief=belief,
            signal_model=model,
            max_steps=max_steps,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )"""
    replace3 = """        locator = SobolSweepLocator.create(
            config=LocatorConfig(max_steps=max_steps),
            belief=belief,
            signal_model=model,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )"""
    if search3 in content:
        content = content.replace(search3, replace3)
        modified = True

    if modified:
        if "from nvision.models.locator import LocatorConfig" not in content:
            content = "from nvision.models.locator import LocatorConfig\n" + content
        with open(filename, "w") as f:
            f.write(content)
        print(f"Updated {filename}")
