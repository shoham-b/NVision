from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Final

_STATIC_INDEX_PATH: Final = Path(__file__).parents[3] / "static" / "index.html"


def _read_manifest_json(out_dir: Path) -> str:
    manifest_path = out_dir / "plots_manifest.json"
    if not manifest_path.exists():
        return "[]"
    return manifest_path.read_text(encoding="utf-8") or "[]"


def _write_js_data_file(path: Path, var_name: str, value_json: str) -> None:
    # Prevent accidental HTML/script termination if someone embeds this in a <script> tag.
    safe_json = value_json.replace("</", "<\\/")
    path.write_text(f"window.{var_name} = {safe_json};\n", encoding="utf-8")


def compile_html_index(out_dir: Path) -> Path:
    """Create an interactive ``index.html`` to browse generated plots.

    The HTML UI is a static file copied into ``out_dir`` unchanged. Plot data is written
    as adjacent JS files to support ``file://`` browsing without fetch/XHR.
    """
    import shutil

    if not _STATIC_INDEX_PATH.exists():
        msg = f"Static UI not found: {_STATIC_INDEX_PATH}"
        raise FileNotFoundError(msg)

    manifest_json = _read_manifest_json(out_dir)
    _write_js_data_file(out_dir / "manifest.js", "MANIFEST", manifest_json)

    settings_json = json.dumps(
        {
            "out_dir": out_dir.as_posix(),
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        },
        indent=2,
    )
    _write_js_data_file(out_dir / "settings.js", "SETTINGS", settings_json)

    index_path = out_dir / "index.html"
    shutil.copy(_STATIC_INDEX_PATH, index_path)

    gif_source = _STATIC_INDEX_PATH.parent / "locator_progress.gif"
    if gif_source.exists():
        shutil.copy(gif_source, out_dir / "locator_progress.gif")

    return index_path
