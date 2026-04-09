"""Stable hashing for cache keys — no domain logic."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_config_hash(config: dict[str, Any]) -> str:
    """MD5 of canonical JSON (sorted keys) for SQLite row keys."""
    canon = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.md5(canon.encode("utf-8")).hexdigest()
