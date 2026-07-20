from __future__ import annotations

from pathlib import Path
from typing import Any

from nvision.tools.paths import ensure_out_dir


class VizBase:
    def __init__(self, out_dir: Path, *, no_disk_write: bool = False) -> None:
        self.out_dir = out_dir
        self.no_disk_write = no_disk_write
        # Populated by _emit() when no_disk_write is set: relative path (as used in
        # manifest entries) -> gzip-compressed bytes, for the API layer to serve
        # instead of a static file.
        self.collected_bytes: dict[str, bytes] = {}
        if not no_disk_write:
            ensure_out_dir(self.out_dir)

    def _emit(self, data: Any, rel_name: str, *, is_figure: bool = False) -> Path:
        """Serialize *data* to ``out_dir/rel_name`` and return that path.

        When ``no_disk_write`` is set, nothing touches disk: the gzip bytes are
        stashed in ``self.collected_bytes`` (keyed by the same path, as a posix
        string) instead, for an API endpoint to serve on demand.
        """
        from nvision.viz._f32_json import dump_gz, write_plotly_gz

        out_path = self.out_dir / rel_name
        serialize = write_plotly_gz if is_figure else dump_gz
        if self.no_disk_write:
            self.collected_bytes[out_path.as_posix()] = serialize(data, None)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            serialize(data, out_path)
        return out_path
