from __future__ import annotations

from pathlib import Path

from nvision.core.paths import ensure_out_dir


class VizBase:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        ensure_out_dir(self.out_dir)
