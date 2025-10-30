from __future__ import annotations

import html
from pathlib import Path
from string import Template
from typing import Final

_TEMPLATE_CACHE: Template | None = None
_TEMPLATE_PATH: Final = Path(__file__).with_name("templates") / "index.html"


def _load_template() -> Template:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        template_text = _TEMPLATE_PATH.read_text(encoding="utf-8")
        _TEMPLATE_CACHE = Template(template_text)
    return _TEMPLATE_CACHE


def compile_html_index(out_dir: Path) -> Path:
    """Create an interactive ``index.html`` to browse generated plots."""
    template = _load_template()

    manifest_path = out_dir / "plots_manifest.json"
    manifest_json = "[]"
    if manifest_path.exists():
        manifest_json = manifest_path.read_text(encoding="utf-8") or "[]"
    manifest_json = manifest_json.replace("</", "<\\/")

    html_content = template.substitute(
        out_dir_display=html.escape(out_dir.as_posix()),
        manifest_json=manifest_json,
    )

    index_path = out_dir / "index.html"
    index_path.write_text(html_content, encoding="utf-8")
    return index_path


__all__ = ["compile_html_index"]
