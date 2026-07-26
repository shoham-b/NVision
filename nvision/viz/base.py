from __future__ import annotations

from pathlib import Path
from typing import Any


class VizBase:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        # Populated by _emit(): relative path (as used in manifest entries) ->
        # gzip-compressed bytes, for the API layer to serve instead of a static file.
        self.collected_bytes: dict[str, bytes] = {}

    def _emit(self, data: Any, rel_name: str, *, is_figure: bool = False) -> Path:
        """Serialize *data*, stash the gzip bytes in ``self.collected_bytes``, and return the path.

        Nothing touches disk: bytes are keyed by ``out_dir/rel_name`` (as a posix
        string) for an API endpoint to serve on demand.
        """
        from nvision.viz._f32_json import dump_gz, write_plotly_gz

        out_path = self.out_dir / rel_name
        serialize = write_plotly_gz if is_figure else dump_gz
        self.collected_bytes[out_path.as_posix()] = serialize(data, None)
        return out_path
