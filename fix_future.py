import glob

test_files = glob.glob("tests/test_*.py")
for filename in test_files:
    with open(filename, "r") as f:
        lines = f.readlines()

    future_line = None
    for i, line in enumerate(lines):
        if line.startswith("from __future__ import annotations"):
            future_line = line
            lines.pop(i)
            break

    if future_line:
        lines.insert(0, future_line)
        with open(filename, "w") as f:
            f.writelines(lines)
        print(f"Fixed {filename}")
