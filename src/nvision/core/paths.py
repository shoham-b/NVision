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
