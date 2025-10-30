from __future__ import annotations

import re
from pathlib import Path


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value)
    slug = slug.strip("_")
    return slug or "item"


__all__ = ["ensure_out_dir", "slugify"]
