from __future__ import annotations

import copy
import logging
from pathlib import Path


def restore_graphs(cached_results: list, out_dir: Path, log: logging.Logger) -> None:
    """Restore cached graph content to files."""
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
        if count > 0:
            log.debug(f"Restored {count} graph files from cache.")
        else:
            log.debug("No graph content found in cache entries to restore.")
    except Exception as e:
        log.warning(f"Failed to restore cached graphs: {e}")


def embed_graph_content(entries: list[dict], out_dir: Path, log: logging.Logger) -> list[dict]:
    """Embed graph file content into entries for caching."""
    entries_with_content = copy.deepcopy(entries)
    count = 0
    for entry in entries_with_content:
        if "path" in entry:
            try:
                file_path = out_dir / entry["path"]
                if file_path.exists():
                    entry["content"] = file_path.read_text(encoding="utf-8")
                    count += 1
            except Exception as e:
                log.warning(f"Failed to read graph content for caching: {e}")
    if count > 0:
        log.debug(f"Embedded content for {count} graph files.")
    return entries_with_content
