"""FastAPI app serving NVision results entirely from the SQLite cache — no static
manifest, no static graph files, no disk writes of any kind.

Replaces the old "restore missing graphs to disk, then serve as static files" model
(``nvision/cli/serve.py``'s former ``_restore_missing_graphs``) with genuine on-demand
serving: every request reads straight from ``artifacts/cache``.
"""

from __future__ import annotations

import base64
import gzip
import logging
import threading
import urllib.parse
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from nvision.cache import CacheBridge
from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_keys import combination_base_cache_config
from nvision.gui.report import _STATIC_DIR, render_index_html
from nvision.runner.cache import _decompress_text
from nvision.sim.combinations import CombinationGrid
from nvision.tools.artifacts import _slim_manifest_entry
from nvision.viz import Viz

# JS/CSS assets served straight from the repo's static/ directory — no per-run
# copy needed, since these never vary by run (see task discussion: nv run used
# to copy these into every out_dir via prepare_static_ui_data; nv serve doesn't
# need that indirection at all, only the static-export path does).
_STATIC_ASSET_NAMES = (
    "app.js", "bootstrap.js", "format-utils.js",
    "plotly-utils.js", "run-status.js", "reload.js", "styles.css",
)

log = logging.getLogger("nvision.api_server")


def _graph_bytes_for_entry(entry: dict) -> bytes | None:
    """Return already-gzip-compressed JSON bytes for one graph entry, or None."""
    content_bin = entry.get("content_bin")
    if content_bin:
        return base64.b85decode(content_bin)
    if entry.get("content"):
        # Legacy zlib+base85 text format — re-gzip so Content-Encoding: gzip stays honest.
        text = _decompress_text(entry)
        return gzip.compress(text.encode("utf-8"))
    return None


# Fields present on every graph-manifest entry that are large and only needed in
# bulk by specific UI features (the per-plot metrics panel, the Highlights
# cross-repeat view) rather than by the picker/dropdown/global manifest scan.
# Dropped from the bulk /api/manifest response; served instead by
# /api/repeat-meta (single repeat) and /api/scan-fields (Highlights, scoped by
# the user's current generator/noise selection) — see nvision/cli/api_server.py
# module docstring history / task_91f913fb for why this split exists.
_BULK_STRIP_FIELDS = frozenset({"series", "true_params", "metrics"})


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that always disables caching — artifacts change on every run."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


class ReloadState:
    def __init__(self) -> None:
        self.running = False
        self.last_output = ""


def build_app(cache_dir: Path, run_dir: Path) -> FastAPI:
    """Build the FastAPI app serving the repo's frontend against *cache_dir*.

    *run_dir* is only needed for run-specific files that genuinely vary per
    run (``locator_results.csv``, ``run_status.json``) — index.html, JS, CSS,
    and graph-def templates are generated/served straight from the repo's
    ``static/`` directory, never copied into *run_dir*.
    """
    app = FastAPI()
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    reload_state = ReloadState()
    # In-process caches, invalidated by /api/reload. Rebuilding the manifest means
    # walking every cached combination's :meta sidecars — cheap per call, but not
    # cheap enough to redo on every page load, so we cache it until told otherwise.
    cache: dict[str, Any] = {"manifest": None, "aggregate_bytes": {}, "combos": None}
    lock = threading.Lock()

    def _bridge() -> CacheBridge:
        return CacheBridge(cache_dir)

    def _combo_key(combo: dict) -> str:
        ptr_config = combination_base_cache_config(
            generator=combo["generator"],
            noise=combo["noise"],
            strategy=combo["strategy"],
            seed=combo["seed"],
            max_steps=combo["max_steps"],
            timeout_s=combo["timeout_s"],
        )
        return stable_config_hash(ptr_config)

    def _load_combo(bridge: CacheBridge, combo: dict) -> tuple[list[dict], list[dict]]:
        """One combo's (graph manifest entries, flat locator-result rows) — a single
        cache read per combo feeds both, instead of scanning the whole cache twice."""
        category = CombinationGrid.generator_category(str(combo["generator"]))
        repo = bridge.get_cache_for_category(category)
        combo_key = _combo_key(combo)
        achieved = int(combo.get("repeats", 0))
        if achieved <= 0:
            return [], []
        meta = repo._repeats.load_repeats_meta(combo_key, achieved)
        if meta is None:
            meta = repo._repeats.load_repeats(combo_key, achieved)
        graph_entries: list[dict] = []
        result_rows: list[dict] = []
        for repeat_idx, (repeat_entries, main_row) in enumerate(meta):
            result_rows.append(main_row)
            for entry in repeat_entries:
                gtype = entry.get("type", "unknown")
                slim = _slim_manifest_entry(entry)
                for field in _BULK_STRIP_FIELDS:
                    slim.pop(field, None)
                slim["path"] = f"/api/graph/{combo_key}/{repeat_idx}/{gtype}.json.gz"
                graph_entries.append(slim)
        return graph_entries, result_rows

    def _build_manifest() -> list[dict]:
        import concurrent.futures
        import os

        import polars as pl

        bridge = _bridge()
        try:
            combos = bridge.list_combinations()
            cache["combos"] = combos
            entries: list[dict] = []
            result_rows: list[dict] = []
            n_workers = min(os.cpu_count() or 4, max(len(combos), 1), 12)
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                for combo_entries, combo_rows in pool.map(lambda c: _load_combo(bridge, c), combos):
                    entries.extend(combo_entries)
                    result_rows.extend(combo_rows)

            # Aggregate views (comparisons/grid-study/experiment summaries) — computed
            # once from the same cache, bytes collected in-memory rather than written.
            # Best-effort: a column-shape surprise in one aggregate view (e.g. a
            # locator that never recorded acquisition_hi) must not take down the
            # per-repeat graph entries above, same as render.py's _run_summary guard.
            viz = Viz(Path("."), no_disk_write=True)
            try:
                df = pl.from_dicts(result_rows, infer_schema_length=None) if result_rows else pl.DataFrame()
                aggregate_entries = viz.plot_locator_summary(df) if not df.is_empty() else []
            except Exception:
                log.exception("Failed to build aggregate views; serving per-repeat graphs only.")
                aggregate_entries = []
            for entry in aggregate_entries:
                original_path = entry["path"]
                entry["path"] = "/api/aggregate/" + urllib.parse.quote(original_path, safe="")
            # Caller (_get_manifest / reload) already holds `lock` while this runs.
            cache["aggregate_bytes"] = dict(viz.collected_bytes)
            entries.extend(aggregate_entries)
            return entries
        finally:
            bridge.close()

    def _get_manifest() -> list[dict]:
        with lock:
            if cache["manifest"] is None:
                cache["manifest"] = _build_manifest()
            return cache["manifest"]

    @app.get("/api/status")
    def status() -> dict:
        return {"reload_running": reload_state.running, "last_output": reload_state.last_output}

    @app.get("/api/manifest")
    def manifest() -> list[dict]:
        return _get_manifest()

    @app.get("/api/graph/{combo_key}/{repeat_idx}/{gtype_with_ext}")
    def graph(combo_key: str, repeat_idx: int, gtype_with_ext: str) -> Response:
        gtype = gtype_with_ext.removesuffix(".json.gz")
        bridge = _bridge()
        try:
            category = None
            # combo_key doesn't carry the category; try both stores (cheap: one indexed lookup each).
            for cat_name in ("NVCenter", "Complementary"):
                repo = bridge.get_cache_for_category(cat_name)
                found = repo._repeats.load_repeat(combo_key, repeat_idx)
                if found is not None:
                    category = cat_name
                    break
            if category is None or found is None:
                raise HTTPException(status_code=404, detail="No such repeat")
            entries, _main_row = found
            entry = next((e for e in entries if e.get("type") == gtype), None)
            if entry is None:
                raise HTTPException(status_code=404, detail="No such graph type for this repeat")
            payload = _graph_bytes_for_entry(entry)
            if payload is None:
                raise HTTPException(status_code=404, detail="Graph entry has no content")
            return Response(
                content=payload,
                media_type="application/json",
                # "identity", not "gzip": the payload is already-gzipped bytes that
                # static/plotly-utils.js's _fetchJson decompresses itself via
                # DecompressionStream (URLs ending in .json.gz always did, back when
                # they were static files with no Content-Encoding at all). Labeling
                # this "gzip" made the browser transparently decompress it first,
                # so the client's own decompress then failed on already-plain JSON
                # (net::ERR_ABORTED). "identity" also stops GZipMiddleware from
                # compressing it a second time — it only checks that some
                # Content-Encoding header is present, not which one.
                headers={"Content-Encoding": "identity", "Cache-Control": "no-store"},
            )
        finally:
            bridge.close()

    @app.get("/api/repeat-meta/{combo_key}/{repeat_idx}")
    def repeat_meta(combo_key: str, repeat_idx: int) -> dict:
        """The bulk-stripped fields (metrics/true_params/series) for one repeat's
        scan entry — fetched on demand by the per-plot metrics panel, which only
        ever needs this for the single currently-selected repeat."""
        bridge = _bridge()
        try:
            found = None
            for cat_name in ("NVCenter", "Complementary"):
                repo = bridge.get_cache_for_category(cat_name)
                found = repo._repeats.load_repeat(combo_key, repeat_idx)
                if found is not None:
                    break
            if found is None:
                raise HTTPException(status_code=404, detail="No such repeat")
            entries, _main_row = found
            scan_entry = next((e for e in entries if e.get("type") == "scan"), None)
            if scan_entry is None:
                raise HTTPException(status_code=404, detail="No scan entry for this repeat")
            return {field: scan_entry.get(field) for field in _BULK_STRIP_FIELDS}
        finally:
            bridge.close()

    @app.get("/api/scan-fields")
    def scan_fields(generator: str, noise: str) -> list[dict]:
        """Bulk series/true_params for every scan repeat matching the given
        generator(s)/noise(s) — comma-separated. Powers the Highlights view's
        cross-repeat aggregation, scoped to the user's current selection instead
        of shipping these fields for the entire cache in /api/manifest."""
        wanted_generators = set(generator.split(","))
        wanted_noises = set(noise.split(","))

        with lock:
            combos = cache["combos"]
        bridge = _bridge()
        try:
            if combos is None:
                combos = bridge.list_combinations()
                with lock:
                    cache["combos"] = combos
            matching = [
                c for c in combos if c["generator"] in wanted_generators and c["noise"] in wanted_noises
            ]
            out: list[dict] = []
            for combo in matching:
                category = CombinationGrid.generator_category(str(combo["generator"]))
                repo = bridge.get_cache_for_category(category)
                combo_key = _combo_key(combo)
                achieved = int(combo.get("repeats", 0))
                if achieved <= 0:
                    continue
                meta = repo._repeats.load_repeats_meta(combo_key, achieved)
                if meta is None:
                    meta = repo._repeats.load_repeats(combo_key, achieved)
                for repeat_entries, main_row in meta:
                    scan_entry = next((e for e in repeat_entries if e.get("type") == "scan"), None)
                    if scan_entry is None:
                        continue
                    out.append({
                        "generator": combo["generator"],
                        "noise": combo["noise"],
                        "strategy": combo["strategy"],
                        "repeat": main_row.get("attempt"),
                        "failure_reason": main_row.get("failure_reason"),
                        "measurements": main_row.get("measurements"),
                        "freq_converged_step": main_row.get("freq_converged_step"),
                        "series": scan_entry.get("series"),
                        "true_params": scan_entry.get("true_params"),
                    })
            return out
        finally:
            bridge.close()

    @app.get("/api/aggregate/{encoded_path}")
    def aggregate(encoded_path: str) -> Response:
        key = urllib.parse.unquote(encoded_path)
        with lock:
            if cache["manifest"] is None:
                cache["manifest"] = _build_manifest()
            payload = cache["aggregate_bytes"].get(key)
        if payload is None:
            raise HTTPException(status_code=404, detail="No such aggregate view")
        return Response(
            content=payload,
            media_type="application/json",
            # See /api/graph's identical fix: "identity", not "gzip" — the
            # payload is already-gzipped and must reach the client's own
            # DecompressionStream untouched.
            headers={"Content-Encoding": "identity", "Cache-Control": "no-store"},
        )

    @app.post("/api/reload")
    def reload_endpoint() -> dict:
        if reload_state.running:
            return {"status": "already_running", "message": "Reload already in progress"}
        reload_state.running = True
        reload_state.last_output = ""

        def _run():
            try:
                with lock:
                    cache["manifest"] = None
                    cache["aggregate_bytes"] = {}
                    cache["combos"] = None
                # Force a rebuild now so the next /api/manifest hit is warm, and to
                # surface any errors in last_output immediately rather than lazily.
                _get_manifest()
                reload_state.last_output = "Reloaded from cache."
            except Exception as exc:
                reload_state.last_output = f"Error: {exc}"
            finally:
                reload_state.running = False

        threading.Thread(target=_run, daemon=True).start()
        return {"status": "started", "message": "Reload started"}

    @app.post("/api/stop")
    def stop() -> dict:
        server = getattr(app.state, "uvicorn_server", None)
        if server is not None:
            threading.Thread(target=lambda: setattr(server, "should_exit", True), daemon=True).start()
        return {"status": "stopping", "message": "Server shutting down"}

    @app.get("/")
    def index() -> HTMLResponse:
        return HTMLResponse(render_index_html(run_dir), headers={"Cache-Control": "no-store"})

    @app.get("/{asset_name}")
    def static_asset(asset_name: str) -> FileResponse:
        if asset_name not in _STATIC_ASSET_NAMES:
            raise HTTPException(status_code=404, detail="No such asset")
        return FileResponse(_STATIC_DIR / asset_name, headers={"Cache-Control": "no-store"})

    @app.get("/graphs/{def_name}")
    def graph_def(def_name: str) -> FileResponse:
        path = _STATIC_DIR / "graphs" / def_name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="No such graph definition")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    # Run-specific files (locator_results.csv, ...) last — anything not matched
    # by a route above falls through to whatever's physically in run_dir.
    app.mount("/", _NoCacheStaticFiles(directory=str(run_dir)), name="run_dir")

    return app
