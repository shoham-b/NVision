import glob

test_files = glob.glob("tests/test_*.py")
for filename in test_files:
    with open(filename, "r") as f:
        content = f.read()

    # ensure ConvergenceConfig is imported if used
    if "ConvergenceConfig" in content and "ConvergenceConfig" not in content[:content.find("\n\n\n")]:
        content = content.replace("from nvision.models.locator import LocatorConfig", "from nvision.models.locator import LocatorConfig, ConvergenceConfig")
        with open(filename, "w") as f:
            f.write(content)
