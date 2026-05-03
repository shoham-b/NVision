import re

with open("tests/test_focus_window.py", "r") as f:
    content = f.read()

s1 = """        locator = SobolSweepLocator.create(
            belief=_dummy_belief(),
            signal_model=DummyModel(),
            max_steps=60,
            noise_std=0.001,
            domain_lo=0.0,
            domain_hi=1.0,
        )"""

r1 = """        locator = SobolSweepLocator.create(
            config=LocatorConfig(max_steps=60, noise_std=0.001),
            belief=_dummy_belief(),
            signal_model=DummyModel(),
            domain_lo=0.0,
            domain_hi=1.0,
        )"""
content = content.replace(s1, r1)

with open("tests/test_focus_window.py", "w") as f:
    f.write(content)

with open("tests/test_locators.py", "r") as f:
    content = f.read()

content = content.replace("SimpleSweepLocator.create(belief=belief, signal_model=model, max_steps=10)", "SimpleSweepLocator.create(config=LocatorConfig(max_steps=10), belief=belief, signal_model=model)")

s2 = """    loc = SweepingLocator.create(
        belief=_dummy_belief(model), signal_model=model, max_steps=10, noise_std=0.02
    )"""

r2 = """    loc = SweepingLocator.create(
        config=LocatorConfig(max_steps=10, noise_std=0.02),
        belief=_dummy_belief(model), signal_model=model
    )"""

content = content.replace(s2, r2)
with open("tests/test_locators.py", "w") as f:
    f.write(content)

with open("tests/test_integration.py", "r") as f:
    content = f.read()

# in run_batch
#         for _ in run_loop(SimpleSweepLocator, experiment, rng, max_steps=max_steps):
# run_loop handles **locator_config

# Actually, if run_loop handles locator_config, it should work fine, but let's check
