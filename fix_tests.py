import glob
import re

test_files = glob.glob("tests/test_*.py")

# Some tests might be failing because of missing imports or incorrect config.
# "NameError: name 'LocatorConfig' is not defined"
# Let's fix that first
for filename in test_files:
    with open(filename, "r") as f:
        content = f.read()

    modified = False

    if "LocatorConfig" in content and "from nvision.models.locator import LocatorConfig" not in content:
        # insert it after imports
        if "from nvision import" in content:
            content = content.replace("from nvision import", "from nvision.models.locator import LocatorConfig, ConvergenceConfig\nfrom nvision import")
        else:
            content = "from nvision.models.locator import LocatorConfig, ConvergenceConfig\n" + content
        modified = True

    if modified:
        with open(filename, "w") as f:
            f.write(content)
        print(f"Fixed {filename}")
