import re

with open("tests/test_new_locators.py", "r") as f:
    content = f.read()

# Replace SimpleSweepLocator.create in test_new_locators.py
search = "loc = SimpleSweepLocator.create(config=LocatorConfig(max_steps=10), belief=belief, signal_model=model)"
replace = "loc = SimpleSweepLocator.create(config=LocatorConfig(max_steps=10), belief=belief, signal_model=model)"
if search in content:
    # Actually wait, maybe SimpleSweepLocator.create needs some specific argument?
    # SimpleSweepLocator is an alias for GenericSweepLocator
    pass
