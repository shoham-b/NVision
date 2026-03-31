from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Final

from nvision.tools.artifacts import plots_manifest_path

_STATIC_DIR: Final = Path(__file__).parents[3] / "static"
_STATIC_INDEX_PATH: Final = _STATIC_DIR / "index.html"


def _read_manifest_json(out_dir: Path) -> str:
    manifest_path = plots_manifest_path(out_dir)
    if not manifest_path.exists():
        return "[]"
    return manifest_path.read_text(encoding="utf-8") or "[]"


def _write_js_data_file(path: Path, var_name: str, value_json: str) -> None:
    # Prevent accidental HTML/script termination if someone embeds this in a <script> tag.
    safe_json = value_json.replace("</", "<\\/")
    path.write_text(f"window.{var_name} = {safe_json};\n", encoding="utf-8")


_MAX_INLINE_MANIFEST_BYTES: int = 50 * 1024 * 1024  # 50 MB threshold


def prepare_static_ui_data(out_dir: Path) -> Path:
    """Prepare data files consumed by the static HTML UI.

    The UI source remains immutable at ``static/index.html``. This function writes only
    data files into ``out_dir`` (for example ``artifacts/``), so the static page can
    render results via ``window.MANIFEST`` and ``window.SETTINGS``.
    """

    if not _STATIC_INDEX_PATH.exists():
        msg = f"Static UI not found: {_STATIC_INDEX_PATH}"
        raise FileNotFoundError(msg)

    manifest_json = _read_manifest_json(out_dir)
    manifest_bytes = len(manifest_json.encode("utf-8"))

    # If manifest is too large, don't inline it as JS - the UI will fetch it as JSON
    if manifest_bytes > _MAX_INLINE_MANIFEST_BYTES:
        # Write a minimal manifest.js that signals the UI to fetch the JSON
        _write_js_data_file(out_dir / "manifest.js", "MANIFEST", "null")
        # Keep the JSON file for fetch()-based loading
    else:
        _write_js_data_file(out_dir / "manifest.js", "MANIFEST", manifest_json)

    settings_json = json.dumps(
        {
            "out_dir": out_dir.as_posix(),
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        },
        indent=2,
    )
    _write_js_data_file(out_dir / "settings.js", "SETTINGS", settings_json)

    # Keep a synced copy of the static UI assets in out_dir so opening
    # `artifacts/index.html` always uses the same JS/CSS as the latest code.
    for name in ("index.html", "app.js", "styles.css"):
        src = _STATIC_DIR / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    # Cache-bust the app bundle reference so browsers don't keep an old app.js.
    index_path = out_dir / "index.html"
    if index_path.exists():
        index_html = index_path.read_text(encoding="utf-8")
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")
        index_html = index_html.replace(
            '<script src="app.js"></script>',
            f'<script src="app.js?v={stamp}"></script>',
        )
        index_path.write_text(index_html, encoding="utf-8")

    # Return the generated out-dir entrypoint users typically open.
    return out_dir / "index.html"
