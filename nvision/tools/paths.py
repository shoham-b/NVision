from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


def ensure_out_dir(path: Path) -> None:
    """Ensure that the output directory exists."""
    path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=4096)
def slugify(value: str) -> str:
    """Convert a string to a slug.

    Cached: task slugs reuse the same generator/noise/strategy labels many times per run.
    """
    value = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def find_project_root() -> Path:
    """Find the project root by looking for a .git directory.

    Falls back to the parent directory of 'nvision' if .git is not found (e.g. in Docker).
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").is_dir() or (parent / "pyproject.toml").is_file() or (parent / "nvision").is_dir():
            return parent
    # Fallback if somehow completely isolated (e.g. site-packages)
    return Path.cwd()


# Define PROJECT_ROOT as the root directory of the project
PROJECT_ROOT = find_project_root()
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
LOGS_ROOT = PROJECT_ROOT / "logs"
