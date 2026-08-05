"""Shared JSON-serialization safety helper.

math.nan/math.inf are valid Python floats used throughout this codebase as
"unconverged/no estimate" sentinels (see nvision/runner/metrics.py), but
json.dumps's default allow_nan=True emits them as literal NaN/Infinity
tokens -- valid for Python's own json module, not valid per the JSON spec.
Browsers' JSON.parse rejects those tokens outright, silently breaking any
endpoint or static export that ships one.
"""

from __future__ import annotations

import math
from typing import Any


def sanitize_non_finite(obj: Any) -> Any:
    """Recursively replace NaN/Infinity/-Infinity floats with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_non_finite(v) for v in obj]
    return obj
