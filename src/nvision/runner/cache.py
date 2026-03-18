"""Cache helpers — embed graph file content into entries and restore from cache."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

log = logging.getLogger(__name__)


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
                        file_path.write_text(entry["content"], encoding="utf-8")
                        count += 1
        if count:
            log.debug("Restored %s graph files from cache.", count)
        else:
            log.debug("No graph content found in cache entries to restore.")
    except Exception as exc:
        log.warning("Failed to restore cached graphs: %s", exc)


def embed_graph_content(entries: list[dict], out_dir: Path) -> list[dict]:
    """Read graph files from disk and embed their text content into entries.

    Returns a deep copy of ``entries`` with a ``"content"`` key added for
    each entry whose file exists on disk.
    """
    entries_with_content = copy.deepcopy(entries)
    count = 0
    for entry in entries_with_content:
        if "path" in entry:
            try:
                file_path = out_dir / entry["path"]
                if file_path.exists():
                    entry["content"] = file_path.read_text(encoding="utf-8")
                    count += 1
            except Exception as exc:
                log.warning("Failed to read graph content for caching: %s", exc)
    if count:
        log.debug("Embedded content for %s graph files.", count)
    return entries_with_content
