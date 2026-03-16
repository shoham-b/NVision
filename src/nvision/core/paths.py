from __future__ import annotations

import re
from pathlib import Path


def ensure_out_dir(path: Path) -> None:
    """Ensure that the output directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    """Convert a string to a slug."""
    value = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def find_project_root() -> Path:
    """Find the project root by looking for a .git directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").is_dir():
            return parent
    raise FileNotFoundError("Could not find the project root (no .git directory found).")


# Define PROJECT_ROOT as the root directory of the project
PROJECT_ROOT = find_project_root()
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
