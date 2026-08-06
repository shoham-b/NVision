"""Compact Plotly figure serialization using base64-encoded Float16/Float32 typed arrays.

Numeric arrays are stored as little-endian binary, base64-encoded, under one of:

- ``__f16__``  — raw float16, when that loses no distinguishable information
- ``__f16s__`` — ``[offset, scale, base64]``, float16 rescaled into the array's own
  [0, 1] range first, tried when raw float16 would overflow or be too coarse
- ``__f32__``  — float32 fallback, used whenever float16 (raw or rescaled) would
  merge two originally-distinguishable values (e.g. a dense, narrow-range
  frequency axis where adjacent points differ by far less than the array's span)

The choice is made per-array by actually attempting float16 and checking the
round-trip error against the array's own finest gap between distinct values —
not by guessing from the field name. This keeps arrays like probability
weights, particle clouds, and signal curves small while leaving arrays that
need it (e.g. a physical-Hz frequency grid) at full float32 precision.

The JS decoder in plotly-utils.js converts all three encodings back to a
Float32Array that Plotly.js accepts natively.

None / null values in source arrays become NaN in the encoded stream.

Files are written as gzip-compressed JSON (``compresslevel=9``: this is a
disk-space-bound cache, not a request-latency-bound service, so max
compression is worth the extra write-time CPU) which gives an additional
3-5x reduction over the typed-array encoding alone.
"""

from __future__ import annotations

import base64
import gzip
import json
import math
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


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


_VALUE_REL_TOL = 2e-3  # relative-to-span tolerance for arrays that already fit float16's range


def _f16_tolerance(finite: np.ndarray, lo: float, hi: float, overflowed: bool) -> float:
    """Max reconstruction error we're willing to accept for this array.

    ``overflowed`` means raw float16 can't even hold the array's magnitude (e.g. a
    frequency axis in raw Hz, ~1e9) -- that's the signature of an unscaled physical
    coordinate, not a plotted value (this codebase always pre-scales physical
    parameters like linewidth/zeeman_split down to O(1-100) before serializing them,
    so anything that still overflows float16 is coordinate-scale by construction).
    Those get a strict tolerance: the finest gap between distinct values actually
    present, so no two originally-distinguishable points -- e.g. two adjacent
    frequency-grid samples -- can be quantized onto the same value. This is what
    catches both a dense sorted grid (x_dense) and an unsorted one (measurements.x,
    picked adaptively over time) the same way, since it doesn't depend on sort order.

    Everything else gets a tolerance relative to the array's own span (or
    magnitude, if constant) -- plenty for anything that's only ever going to be
    looked at on a chart (signal curves, particle weights/positions, covariances).
    """
    span = hi - lo
    magnitude = max(abs(lo), abs(hi))
    # Floor scaled to the array's own magnitude (not a fixed constant): a fixed
    # absolute floor like 1e-6 is meaningless noise for a ~1e-8-scale array (it
    # would happily accept float16 flushing the whole array to zero) and overly
    # strict for a ~1e6-scale one. Tie it to float32's own precision instead.
    eps_floor = magnitude * np.finfo(np.float32).eps * 4
    if overflowed:
        uniq = np.unique(finite)
        if uniq.size >= 2:
            return float(np.diff(uniq).min())
        return max(magnitude * _VALUE_REL_TOL, eps_floor)
    if span > 0:
        return max(span * _VALUE_REL_TOL, eps_floor)
    return max(magnitude * _VALUE_REL_TOL, eps_floor)


def _pick_f16_or_f32(out: np.ndarray) -> dict[str, Any]:
    """Encode a float32 ndarray (already NaN-sanitized) as f16 when lossless-enough, else f32.

    See ``_f16_tolerance`` for what "lossless-enough" means.
    """
    finite = out[np.isfinite(out)]
    if finite.size < 2:
        with np.errstate(over="ignore"):
            raw16 = out.astype(np.float16)
        if np.isinf(raw16).any() and not np.isinf(out).any():
            return {"__f32__": _b64(out)}
        return {"__f16__": _b64(raw16)}

    lo, hi = float(finite.min()), float(finite.max())

    # Try raw float16 first — cheapest, no extra scalars.
    with np.errstate(over="ignore"):
        raw16 = out.astype(np.float16)
    overflowed = bool(np.isinf(raw16).any() and not np.isinf(out).any())
    tol = _f16_tolerance(finite, lo, hi, overflowed)

    if not overflowed:
        err = np.abs(raw16.astype(np.float64) - out.astype(np.float64))
        err = err[np.isfinite(err)]
        if err.size == 0 or err.max() <= tol:
            return {"__f16__": _b64(raw16)}

    # Try rescaled float16 (offset + scale, two extra float64 scalars).
    span = hi - lo
    if span > 0:
        scaled16 = ((out - lo) / span).astype(np.float16)
        recon = scaled16.astype(np.float64) * span + lo
        err = np.abs(recon - out.astype(np.float64))
        err = err[np.isfinite(err)]
        if err.size == 0 or err.max() <= tol:
            return {"__f16s__": [lo, span, _b64(scaled16)]}

    return {"__f32__": _b64(out)}


def _encode_f32(lst: list) -> dict[str, Any]:
    arr = np.asarray(
        [float("nan") if v is None else float(v) for v in lst],
        dtype=np.float32,
    )
    return _pick_f16_or_f32(arr)


def _encode_arrays(obj: Any) -> Any:
    """Recursively replace eligible numeric lists with a float16/float32-encoded dict."""
    if isinstance(obj, list):
        if len(obj) >= _MIN_ARRAY_LEN and _is_numeric_list(obj):
            return _encode_f32(obj)
        return [_encode_arrays(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _encode_arrays(v) for k, v in obj.items()}
    return obj


def fig_to_f32_json(fig: Any) -> str:
    """Serialize a Plotly figure to JSON with numeric arrays as Float16/Float32.

    Drop-in replacement for ``fig.to_json()``.
    """
    d = fig.to_plotly_json()
    return json.dumps(_encode_arrays(d), separators=(",", ":"))


def to_gz_bytes(fig: Any) -> bytes:
    """Serialize a Plotly figure to gzip-compressed Float16/Float32 JSON bytes (no disk I/O)."""
    # mtime=0: gzip's header otherwise embeds the current wall-clock time, so two
    # calls encoding identical content a second (or more) apart produce different
    # bytes -- pinning it makes output byte-for-byte reproducible for identical
    # input, which content-addressed caching/serving and byte-identity tests rely on.
    return gzip.compress(fig_to_f32_json(fig).encode("utf-8"), compresslevel=9, mtime=0)


def _sanitize_non_finite(obj: Any) -> Any:
    """Recursively replace non-finite floats (Infinity, -Infinity, NaN) with None."""
    if isinstance(obj, float):
        return None if not (obj == obj and obj != float("inf") and obj != float("-inf")) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


def _encode_ndarray_f32(arr: np.ndarray) -> dict[str, Any]:
    """Encode a 1D numeric ndarray as float16/float32 (non-finite -> NaN)."""
    out = np.ascontiguousarray(arr, dtype=np.float32)
    if out.dtype.kind == "f" and not np.isfinite(out).all():
        out = np.where(np.isfinite(out), out, np.float32(np.nan))
    return _pick_f16_or_f32(out)


def _encode_payload(obj: Any) -> Any:
    """Single-pass sanitize + Float16/Float32 encode for JSON payloads.

    Equivalent to ``_encode_arrays(_sanitize_non_finite(obj))`` but walks the
    payload once and accepts numpy arrays directly, so writers can pass
    ndarrays without materializing intermediate Python lists (``.tolist()``).
    Non-finite values become NaN inside encoded arrays and ``None``
    elsewhere, matching the two-pass behavior.
    """
    if isinstance(obj, dict):
        return {k: _encode_payload(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return _encode_payload(obj.item())
        if obj.ndim == 1:
            if obj.shape[0] >= _MIN_ARRAY_LEN and obj.dtype.kind in "fiu":
                return _encode_ndarray_f32(obj)
            return [_encode_payload(v) for v in obj.tolist()]
        return [_encode_payload(row) for row in obj]
    if isinstance(obj, list):
        if len(obj) >= _MIN_ARRAY_LEN and _is_numeric_list(obj):
            # Via ndarray so non-finite values (inf as well as NaN) become NaN,
            # matching the sanitize-then-encode behavior of the two-pass path.
            arr = np.asarray([float("nan") if v is None else float(v) for v in obj], dtype=np.float32)
            return _encode_ndarray_f32(arr)
        return [_encode_payload(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def payload_to_gz_bytes(payload: Any) -> bytes:
    """Serialize any JSON-serializable payload (numpy arrays welcome) to gzip-compressed Float16/Float32 JSON bytes."""
    encoded = _encode_payload(payload)
    # mtime=0: see to_gz_bytes for why this is pinned rather than left to default
    # to the current wall-clock time.
    return gzip.compress(json.dumps(encoded, separators=(",", ":")).encode("utf-8"), compresslevel=9, mtime=0)


def write_plotly_gz(fig: Any, out_path: Path | None = None) -> bytes:
    """Serialize a Plotly figure to gzip-compressed Float16/Float32 JSON.

    Returns the raw bytes.  If *out_path* is given the bytes are also written
    to disk (path should end in ``.json.gz``).
    """
    data = to_gz_bytes(fig)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
    return data


def dump_gz(payload: Any, out_path: Path | None = None) -> bytes:
    """Serialize any payload to gzip-compressed Float16/Float32 JSON.

    Returns the raw bytes.  If *out_path* is given the bytes are also written
    to disk (path should end in ``.json.gz``).
    """
    data = payload_to_gz_bytes(payload)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
    return data


def _decode_arrays(obj: Any) -> Any:
    """Recursively decode ``{__f16__/__f16s__/__f32__: ...}`` back to plain Python lists."""
    if isinstance(obj, dict):
        if "__f32__" in obj and len(obj) == 1:
            data = base64.b64decode(obj["__f32__"])
            arr = np.frombuffer(data, dtype=np.float32)
            return arr.tolist()
        if "__f16__" in obj and len(obj) == 1:
            data = base64.b64decode(obj["__f16__"])
            arr = np.frombuffer(data, dtype=np.float16)
            return arr.astype(np.float32).tolist()
        if "__f16s__" in obj and len(obj) == 1:
            lo, span, b64 = obj["__f16s__"]
            data = base64.b64decode(b64)
            arr = np.frombuffer(data, dtype=np.float16).astype(np.float64)
            return (arr * span + lo).tolist()
        return {k: _decode_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_arrays(v) for v in obj]
    return obj


def from_gz_bytes(data: bytes) -> Any:
    """Deserialize gzip-compressed Float16/Float32 JSON bytes back to a Python dict/list.

    Decodes typed-array tags back to plain float lists so the result can be
    passed directly to ``plotly.io.from_json`` or ``go.Figure``.
    """
    raw = json.loads(gzip.decompress(data).decode("utf-8"))
    return _decode_arrays(raw)


def figure_from_gz_bytes(data: bytes) -> Any:
    """Deserialize gzip-compressed Float16/Float32 JSON bytes to a ``go.Figure``."""
    import plotly.io as pio

    decoded = from_gz_bytes(data)
    return pio.from_json(json.dumps(decoded))
