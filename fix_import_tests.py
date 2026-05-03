import glob

test_files = glob.glob("tests/test_*.py")

for filename in test_files:
    with open(filename, "r") as f:
        content = f.read()

    modified = False

    if "NameError: name 'LocatorConfig' is not defined" or "NameError: name 'ConvergenceConfig' is not defined" or True:
        # Check if LocatorConfig is in the file but not imported
        if "LocatorConfig" in content and "from nvision.models.locator import" not in content:
            # We already tried this but maybe it was placed wrongly
            content = "from nvision.models.locator import LocatorConfig, ConvergenceConfig\n" + content
            modified = True

        elif "LocatorConfig" in content and "ConvergenceConfig" in content:
            if "from nvision.models.locator import LocatorConfig\n" in content:
                content = content.replace("from nvision.models.locator import LocatorConfig\n", "from nvision.models.locator import LocatorConfig, ConvergenceConfig\n")
                modified = True

    if modified:
        with open(filename, "w") as f:
            f.write(content)
