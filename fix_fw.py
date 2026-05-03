with open("tests/test_focus_window.py", "r") as f:
    content = f.read()

if "LocatorConfig" in content and "from nvision.models.locator import" not in content:
    content = "from nvision.models.locator import LocatorConfig, ConvergenceConfig\n" + content

with open("tests/test_focus_window.py", "w") as f:
    f.write(content)
