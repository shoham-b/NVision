from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Final

from nvision.sim.combinations import CombinationGrid
from nvision.tools.artifacts import plots_manifest_path, read_run_status

_STATIC_DIR: Final = Path(__file__).parents[2] / "static"
_STATIC_INDEX_PATH: Final = _STATIC_DIR / "index.html"


def _strategy_grid_json() -> str:
    """Locator strategy names per generator for UI controls (includes not-yet-run strategies)."""
    grid = CombinationGrid()
    strategy_grid = {gen_name: [name for name, _ in grid.strategies_for(gen_name)] for gen_name in grid.generators}
    return json.dumps(strategy_grid, indent=2)


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

    # Build the data scripts that will be injected into the HTML
    data_scripts = []

    # If manifest is too large, don't inline it - app.js will fetch it
    if manifest_bytes > _MAX_INLINE_MANIFEST_BYTES:
        data_scripts.append("<script>window.MANIFEST = null;</script>")
    else:
        # Inline the manifest
        safe_manifest = manifest_json.replace("</", "<\\/")
        data_scripts.append(f"<script>window.MANIFEST = {safe_manifest};</script>")

    # Inline settings
    settings_json = json.dumps(
        {
            "out_dir": out_dir.as_posix(),
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        },
        indent=2,
    )
    safe_settings = settings_json.replace("</", "<\\/")
    data_scripts.append(f"<script>window.SETTINGS = {safe_settings};</script>")

    strategy_grid_json = _strategy_grid_json()
    safe_strategy_grid = strategy_grid_json.replace("</", "<\\/")
    data_scripts.append(f"<script>window.STRATEGY_GRID = {safe_strategy_grid};</script>")

    # Inline run status so the UI can show banners even before polling
    run_status = read_run_status(out_dir)
    if run_status is not None:
        run_status_json = json.dumps(run_status, indent=2)
        safe_run_status = run_status_json.replace("</", "<\\/")
        data_scripts.append(f"<script>window.RUN_STATUS = {safe_run_status};</script>")
    else:
        data_scripts.append("<script>window.RUN_STATUS = null;</script>")

    # Add asset prefix for resolving relative paths
    data_scripts.append('<script>window.NVISION_ASSET_PREFIX = "./";</script>')

    # Inject data scripts at the very start of <head> so window.MANIFEST / window.SETTINGS
    # are defined before any split JS file (bootstrap.js, etc.) executes.
    data_block = "\n".join(data_scripts)
    index_html = index_html.replace("<head>", f"<head>\n{data_block}", 1)

    # Inline CSS to avoid file:// URL caching issues
    css_src = _STATIC_DIR / "styles.css"
    if css_src.exists():
        css_content = css_src.read_text(encoding="utf-8")
        css_inline = f"<style>\n{css_content}\n</style>"
        css_pattern = '<link rel="stylesheet" href="styles.css">'
        index_html = index_html.replace(css_pattern, css_inline)

    # Copy all split JS files to out_dir
    import shutil

    _JS_FILES = [
        "app.js", "bootstrap.js", "format-utils.js",
        "plotly-utils.js", "run-status.js", "reload.js",
    ]
    for js_file in _JS_FILES:
        js_src = _STATIC_DIR / js_file
        if js_src.exists():
            shutil.copy2(js_src, out_dir / js_file)

    # If manifest was too large to inline, write it as JSON for fetching
    if manifest_bytes > _MAX_INLINE_MANIFEST_BYTES:
        # Also write the JSON file
        (out_dir / "plots_manifest.json").write_text(manifest_json, encoding="utf-8")

    # Write the final index.html
    index_path = out_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")

    return index_path
