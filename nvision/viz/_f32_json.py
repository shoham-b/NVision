"""Compact Plotly figure serialization using base64-encoded Float32 typed arrays.

Numeric arrays are stored as little-endian float32 binary, base64-encoded under
the ``__f32__`` key.  The JS decoder in app.js converts these back to Float32Array
objects that Plotly.js accepts natively — faster to parse and ~3× smaller on disk
than JSON text numbers.

None / null values in source arrays become NaN in the float32 stream.

Files are written as gzip-compressed JSON (``compresslevel=1`` for speed) which
gives an additional 3-5× reduction over the Float32 encoding alone.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


_MIN_ARRAY_LEN: int = 4


def _is_numeric_list(obj: list) -> bool:
    """True when every element is a plain number or None (not bool, not str)."""
    for v in obj:
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
    return True


def _encode_f32(lst: list) -> dict[str, str]:
    arr = np.asarray(
        [float("nan") if v is None else float(v) for v in lst],
        dtype=np.float32,
    )
    return {"__f32__": base64.b64encode(arr.tobytes()).decode("ascii")}


def _encode_arrays(obj: Any) -> Any:
    """Recursively replace eligible numeric lists with ``{__f32__: base64}``."""
    if isinstance(obj, list):
        if len(obj) >= _MIN_ARRAY_LEN and _is_numeric_list(obj):
            return _encode_f32(obj)
        return [_encode_arrays(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _encode_arrays(v) for k, v in obj.items()}
    return obj


def fig_to_f32_json(fig: Any) -> str:
    """Serialize a Plotly figure to JSON with numeric arrays as Float32.

    Drop-in replacement for ``fig.to_json()``.
    """
    d = fig.to_plotly_json()
    return json.dumps(_encode_arrays(d), separators=(",", ":"))


def to_gz_bytes(fig: Any) -> bytes:
    """Serialize a Plotly figure to gzip-compressed Float32 JSON bytes (no disk I/O)."""
    return gzip.compress(fig_to_f32_json(fig).encode("utf-8"), compresslevel=1)


def _sanitize_non_finite(obj: Any) -> Any:
    """Recursively replace non-finite floats (Infinity, -Infinity, NaN) with None."""
    if isinstance(obj, float):
        return None if not (obj == obj and obj != float("inf") and obj != float("-inf")) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


def payload_to_gz_bytes(payload: Any) -> bytes:
    """Serialize any JSON-serializable payload to gzip-compressed Float32 JSON bytes."""
    encoded = _encode_arrays(_sanitize_non_finite(payload))
    return gzip.compress(json.dumps(encoded, separators=(",", ":")).encode("utf-8"), compresslevel=1)


def write_plotly_gz(fig: Any, out_path: Path | None = None) -> bytes:
    """Serialize a Plotly figure to gzip-compressed Float32 JSON.

    Returns the raw bytes.  If *out_path* is given the bytes are also written
    to disk (path should end in ``.json.gz``).
    """
    data = to_gz_bytes(fig)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
    return data


def dump_gz(payload: Any, out_path: Path | None = None) -> bytes:
    """Serialize any payload to gzip-compressed Float32 JSON.

    Returns the raw bytes.  If *out_path* is given the bytes are also written
    to disk (path should end in ``.json.gz``).
    """
    data = payload_to_gz_bytes(payload)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
    return data


def _decode_arrays(obj: Any) -> Any:
    """Recursively decode ``{__f32__: base64}`` back to plain Python lists."""
    if isinstance(obj, dict):
        if "__f32__" in obj and len(obj) == 1:
            data = base64.b64decode(obj["__f32__"])
            arr = np.frombuffer(data, dtype=np.float32)
            return arr.tolist()
        return {k: _decode_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_arrays(v) for v in obj]
    return obj


def from_gz_bytes(data: bytes) -> Any:
    """Deserialize gzip-compressed Float32 JSON bytes back to a Python dict/list.

    Decodes ``{__f32__: base64}`` arrays back to plain float lists so the
    result can be passed directly to ``plotly.io.from_json`` or ``go.Figure``.
    """
    raw = json.loads(gzip.decompress(data).decode("utf-8"))
    return _decode_arrays(raw)


def figure_from_gz_bytes(data: bytes) -> Any:
    """Deserialize gzip-compressed Float32 JSON bytes to a ``go.Figure``."""
    import plotly.io as pio
    decoded = from_gz_bytes(data)
    return pio.from_json(json.dumps(decoded))
