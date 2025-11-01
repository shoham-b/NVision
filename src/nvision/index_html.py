from __future__ import annotations

import html
from pathlib import Path
from typing import Final

_TEMPLATE_CACHE: str | None = None
_TEMPLATE_PATH: Final = Path(__file__).with_name("templates") / "index.html"


def _load_template() -> str:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return _TEMPLATE_CACHE


def compile_html_index(out_dir: Path) -> Path:
    """Create an interactive ``index.html`` to browse generated plots."""
    template = _load_template()

    manifest_path = out_dir / "plots_manifest.json"
    manifest_json = "[]"
    if manifest_path.exists():
        manifest_json = manifest_path.read_text(encoding="utf-8") or "[]"
    manifest_json = manifest_json.replace("</", "<\\/")

    html_content = template.replace("${out_dir_display}", html.escape(out_dir.as_posix())).replace(
        "${manifest_json}", manifest_json
    )

    index_path = out_dir / "index.html"
    index_path.write_text(html_content, encoding="utf-8")
    return index_path


__all__ = ["compile_html_index"]
