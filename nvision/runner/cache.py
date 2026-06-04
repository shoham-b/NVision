"""Graph file embed/restore helpers for locator runs."""

from __future__ import annotations

import base64
import copy
import json
import logging
import zlib
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = [
    "embed_graph_content",
    "restore_graphs",
    "strip_heavy_fields",
]


_HEAVY_FIELDS = frozenset({"content", "content_bin", "plot_data", "_bytes"})


def strip_heavy_fields(entry: dict) -> dict:
    """Return a copy of *entry* with heavy fields removed.

    Stripped fields: ``content``, ``content_bin``, ``plot_data``, ``_bytes``.
    These are used for cache storage or in-memory transport but must not
    appear in ``plots_manifest.json``.
    """
    return {k: v for k, v in entry.items() if k not in _HEAVY_FIELDS}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decompress_text(entry: dict) -> str:
    """Return the text content from *entry* after zlib+base85 decompression."""
    return zlib.decompress(base64.b85decode(entry["content"])).decode("utf-8")


def _compress_text(text: str) -> str:
    """Compress *text* with zlib+base85 (level 6)."""
    return base64.b85encode(zlib.compress(text.encode("utf-8"), level=6)).decode("ascii")


def _encode_binary(data: bytes) -> str:
    """Encode binary *data* as base85 for cache storage."""
    return base64.b85encode(data).decode("ascii")


def _decode_binary(entry: dict) -> bytes:
    """Decode base85-encoded binary content from *entry*."""
    return base64.b85decode(entry["content_bin"])


def _upgrade_json_to_gz(text: str, out_path: Path) -> bool:
    """Convert old float64 JSON text to new Float32+gzip and write to *out_path*.

    Returns True on success.  Falls back gracefully on any error.
    """
    try:
        from nvision.viz._f32_json import dump_gz

        payload = json.loads(text)
        dump_gz(payload, out_path)
        return True
    except Exception as exc:
        log.warning("Could not upgrade cached JSON to gz format (%s): %s", out_path.name, exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def restore_graphs(cached_results: list, out_dir: Path) -> None:
    """Write cached graph content back to disk.

    Handles two storage formats:
    - ``content_bin`` — new binary format for ``.json.gz`` files (base85-encoded bytes)
    - ``content``     — old text format for ``.json`` files (zlib+base85 text)

    Old ``.json`` entries are transparently upgraded to ``.json.gz`` on restore:
    the file is written in the new Float32+gzip format and ``entry["path"]`` is
    updated in-place so the manifest gets the correct new path.

    Skips files that already exist on disk.
    """
    count = 0
    try:
        for entries, _ in cached_results:
            for entry in entries:
                if "path" not in entry:
                    continue

                if "content_bin" in entry:
                    # ---- New binary path (.json.gz stored as raw bytes) ----
                    file_path = out_dir / entry["path"]
                    if not file_path.exists():
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_bytes(_decode_binary(entry))
                        count += 1

                elif "content" in entry:
                    orig_path = entry["path"]

                    # ---- Old text path: upgrade .json -> .json.gz on the fly ----
                    if orig_path.endswith(".json") and not orig_path.endswith(".json.gz"):
                        new_path = orig_path + ".gz"
                        file_path = out_dir / new_path
                        if not file_path.exists():
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            old_text = _decompress_text(entry)
                            if _upgrade_json_to_gz(old_text, file_path):
                                count += 1
                            else:
                                # Fallback: write old-format file at original path
                                fallback = out_dir / orig_path
                                if not fallback.exists():
                                    fallback.parent.mkdir(parents=True, exist_ok=True)
                                    fallback.write_text(old_text, encoding="utf-8")
                                    count += 1
                                new_path = orig_path  # keep old path in manifest
                        # Update entry path so the manifest points to the right file
                        entry["path"] = new_path

                    else:
                        # Plain text file that isn't a .json graph (shouldn't occur normally)
                        file_path = out_dir / orig_path
                        if not file_path.exists():
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            file_path.write_text(_decompress_text(entry), encoding="utf-8")
                            count += 1

        if count:
            log.debug("Restored %s graph files from cache.", count)
        else:
            log.debug("No graph content found in cache entries to restore.")
    except Exception as exc:
        log.warning("Failed to restore cached graphs: %s", exc)


def embed_graph_content(
    entries: list[dict],
    out_dir: Path,
    content_map: dict[str, bytes] | None = None,
) -> list[dict]:
    """Embed graph content into *entries* for cache storage.

    Prefers bytes from *content_map* (in-memory, no disk I/O) when available.
    Falls back to reading from disk for entries not in *content_map*.

    Storage format:
    - ``.json.gz`` (binary) → ``"content_bin"`` (base85-encoded raw bytes)
    - other text files      → ``"content"`` (zlib+base85, legacy format)
    """
    entries_with_content = copy.deepcopy(entries)
    count = 0
    for entry in entries_with_content:
        rel_path = entry.get("path")
        if not rel_path:
            continue
        try:
            # Priority 1: in-memory bytes injected by generate_attempt_plots
            raw: bytes | None = entry.pop("_bytes", None)

            # Priority 2: caller-supplied content_map
            if raw is None and content_map:
                raw = content_map.get(rel_path)

            # Priority 3: disk fallback
            if raw is None:
                file_path = out_dir / rel_path
                if file_path.exists():
                    raw = (
                        file_path.read_bytes()
                        if file_path.suffix == ".gz"
                        else file_path.read_text(encoding="utf-8").encode("utf-8")
                    )

            if raw is not None:
                if rel_path.endswith(".gz"):
                    entry["content_bin"] = _encode_binary(raw)
                else:
                    entry["content"] = _compress_text(raw.decode("utf-8"))
                count += 1
        except Exception as exc:
            log.warning("Failed to embed graph content for %s: %s", rel_path, exc)
    if count:
        log.debug("Embedded content for %s graph files.", count)
    return entries_with_content
