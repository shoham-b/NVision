from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Final

from nvision.tools.artifacts import plots_manifest_path

_STATIC_DIR: Final = Path(__file__).parents[2] / "static"
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

    Generates a standalone index.html in out_dir with all assets and data inline
    or properly referenced for both HTTP serving and local file opening.
    """

    if not _STATIC_INDEX_PATH.exists():
        msg = f"Static UI not found: {_STATIC_INDEX_PATH}"
        raise FileNotFoundError(msg)

    # Read the static HTML template
    index_html = _STATIC_INDEX_PATH.read_text(encoding="utf-8")

    # Read manifest data
    manifest_json = _read_manifest_json(out_dir)
    manifest_bytes = len(manifest_json.encode("utf-8"))

    # Cache-bust timestamp
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")

    # Build the data scripts that will be injected into the HTML
    data_scripts = []

    # If manifest is too large, don't inline it - app.js will fetch it
    if manifest_bytes > _MAX_INLINE_MANIFEST_BYTES:
        data_scripts.append('<script>window.MANIFEST = null;</script>')
    else:
        # Inline the manifest
        safe_manifest = manifest_json.replace("</", "<\\/")
        data_scripts.append(f'<script>window.MANIFEST = {safe_manifest};</script>')

    # Inline settings
    settings_json = json.dumps(
        {
            "out_dir": out_dir.as_posix(),
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        },
        indent=2,
    )
    safe_settings = settings_json.replace("</", "<\\/")
    data_scripts.append(f'<script>window.SETTINGS = {safe_settings};</script>')

    # Add asset prefix for resolving relative paths
    data_scripts.append('<script>window.NVISION_ASSET_PREFIX = "./";</script>')

    # Inject data scripts before </head>
    data_block = '\n'.join(data_scripts)
    index_html = index_html.replace('</head>', f'{data_block}\n</head>')

    # Replace the iframe loader with direct script tag
    iframe_pattern = '<iframe src="../artifacts/loader.html" style="display:none"></iframe>'
    script_tag = f'<script src="app.js?v={stamp}"></script>'
    index_html = index_html.replace(iframe_pattern, script_tag)

    # Also handle case where path might be different
    iframe_pattern2 = '<iframe src="loader.html" style="display:none"></iframe>'
    index_html = index_html.replace(iframe_pattern2, script_tag)

    # Copy static assets to out_dir
    import shutil
    shutil.copy2(_STATIC_DIR / "styles.css", out_dir / "styles.css")

    # Copy app.js with cache-busting in the URL (content unchanged)
    app_js_src = _STATIC_DIR / "app.js"
    app_js_dest = out_dir / "app.js"
    if app_js_src.exists():
        shutil.copy2(app_js_src, app_js_dest)

    # If manifest was too large to inline, write it as JSON for fetching
    if manifest_bytes > _MAX_INLINE_MANIFEST_BYTES:
        # Also write the JSON file
        (out_dir / "plots_manifest.json").write_text(manifest_json, encoding="utf-8")

    # Write the final index.html
    index_path = out_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")

    return index_path
