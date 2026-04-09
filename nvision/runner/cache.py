"""Graph file embed/restore helpers for locator runs."""

from __future__ import annotations

import base64
import copy
import logging
import zlib
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = [
    "embed_graph_content",
    "restore_graphs",
]


def _decompress_content(entry: dict) -> str:
    """Return the text content from *entry* after zlib+base85 decompression."""
    return zlib.decompress(base64.b85decode(entry["content"])).decode("utf-8")


def _compress_content(text: str) -> str:
    """Compress *text* with zlib+base85."""
    return base64.b85encode(zlib.compress(text.encode("utf-8"), level=6)).decode("ascii")


def restore_graphs(cached_results: list, out_dir: Path) -> None:
    """Write cached graph content back to disk.

    Skips files that already exist. Logs the number of files restored.
    """
    count = 0
    try:
        for entries, _ in cached_results:
            for entry in entries:
                if "path" in entry and "content" in entry:
                    file_path = out_dir / entry["path"]
                    if not file_path.exists():
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(_decompress_content(entry), encoding="utf-8")
                        count += 1
        if count:
            log.debug("Restored %s graph files from cache.", count)
        else:
            log.debug("No graph content found in cache entries to restore.")
    except Exception as exc:
        log.warning("Failed to restore cached graphs: %s", exc)


def embed_graph_content(entries: list[dict], out_dir: Path) -> list[dict]:
    """Read graph files from disk, compress, and embed into entries.

    Returns a deep copy of ``entries`` with a compressed ``"content"`` key
    added for each entry whose file exists on disk.
    """
    entries_with_content = copy.deepcopy(entries)
    count = 0
    for entry in entries_with_content:
        if "path" in entry:
            try:
                file_path = out_dir / entry["path"]
                if file_path.exists():
                    text = file_path.read_text(encoding="utf-8")
                    entry["content"] = _compress_content(text)
                    count += 1
            except Exception as exc:
                log.warning("Failed to read graph content for caching: %s", exc)
    if count:
        log.debug("Embedded content for %s graph files (compressed).", count)
    return entries_with_content
