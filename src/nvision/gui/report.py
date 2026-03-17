from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Final

_STATIC_DIR: Final = Path(__file__).parents[3] / "static"
_STATIC_INDEX_PATH: Final = _STATIC_DIR / "index.html"


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
    """Prepare data files consumed by the static HTML UI.

    The UI source remains immutable at ``static/index.html``. This function writes only
    data files into ``out_dir`` (for example ``artifacts/``), so the static page can
    render results via ``window.MANIFEST`` and ``window.SETTINGS``.
    """

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

    # Return the immutable static UI entrypoint (repo-root static page).
    return _STATIC_INDEX_PATH
