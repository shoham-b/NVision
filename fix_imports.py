"""Fix ScanBatch imports across the codebase."""

import re
from pathlib import Path


def fix_file(file_path: Path):
    """Fix ScanBatch import in a single file."""
    content = file_path.read_text(encoding="utf-8")
    original = content

    # Pattern: from nvision.sim.locs.base import ... ScanBatch ...
    pattern = r"from nvision\.sim\.locs\.base import ([^;\n]*ScanBatch[^;\n]*)"

    def replace_import(match):
        imports = match.group(1)
        # Split by comma, strip whitespace, filter out ScanBatch and empty strings
        parts = [p.strip().rstrip(",") for p in imports.split(",")]
        other_imports = [p for p in parts if p and p != "ScanBatch"]

        lines = []
        lines.append("from nvision.sim.scan_batch import ScanBatch")
        if other_imports:
            lines.append(f"from nvision.sim.locs.base import {', '.join(other_imports)}")
        return "\n".join(lines)

    content = re.sub(pattern, replace_import, content)

    if content != original:
        file_path.write_text(content, encoding="utf-8")
        print(f"Fixed: {file_path}")
        return True
    return False


# Find all Python files
root = Path("src/nvision")
py_files = list(root.rglob("*.py"))

fixed_count = 0
for py_file in py_files:
    if fix_file(py_file):
        fixed_count += 1

print(f"\nFixed {fixed_count} files")
