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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from nvision.cache import CacheBridge
from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_keys import combination_base_cache_config
from nvision.gui.report import _STATIC_DIR, render_index_html
from nvision.runner.cache import _decompress_text
from nvision.sim.combinations import CombinationGrid
from nvision.tools.artifacts import _slim_manifest_entry
from nvision.tools.json_sanitize import sanitize_non_finite
from nvision.viz import Viz

# JS/CSS assets served straight from the repo's static/ directory — no per-run
# copy needed, since these never vary by run (see task discussion: nv run used
# to copy these into every out_dir via prepare_static_ui_data; nv serve doesn't
# need that indirection at all, only the static-export path does).
_STATIC_ASSET_NAMES = (
    "app.js",
    "bootstrap.js",
    "format-utils.js",
    "plotly-utils.js",
    "run-status.js",
    "reload.js",
    "styles.css",
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


def _dedupe_latest_combos(combos: list[dict]) -> list[dict]:
    """Keep only the most-recently-updated generation per (generator, noise, strategy).

    A long-lived cache accumulates multiple max_steps/timeout_s generations of
    what's effectively "the same" combo from iterative tuning -- each one fully
    counted in /api/manifest's response, even though the UI only ever displays
    one at a time. Dropping superseded generations from the response (nothing is
    deleted from the cache itself) is what keeps that response from growing
    without bound as the cache accumulates history. Ties broken arbitrarily
    (equal updated_at is not expected in practice). Takes
    CacheBridge.list_combinations_with_updated_at()'s output.
    """
    latest: dict[tuple[str, str, str], dict] = {}
    for combo in combos:
        key = (combo["generator"], combo["noise"], combo["strategy"])
        current = latest.get(key)
        if current is None or combo.get("updated_at", "") >= current.get("updated_at", ""):
            latest[key] = combo
    return list(latest.values())


def _build_aggregate_entries(result_rows: list[dict]) -> tuple[list[dict], dict[str, bytes]]:
    """Comparisons/grid-study/experiment summary views computed from every combo's flat
    result row -- (entries with /api/aggregate/... paths, their collected gzip bytes).

    Module-level (not a build_app closure) because it needs none of build_app's shared
    server state, only result_rows -- keeping it out of build_app's own body is what
    keeps that function's branch count (mccabe complexity) down as endpoints are added.
    Best-effort: a column-shape surprise in one aggregate view (e.g. a locator that
    never recorded acquisition_hi) must not take down the per-repeat graph entries this
    feeds into, same as render.py's _run_summary guard.
    """
    import polars as pl

    viz = Viz(Path("."))
    try:
        df = pl.from_dicts(result_rows, infer_schema_length=None) if result_rows else pl.DataFrame()
        aggregate_entries = viz.plot_locator_summary(df) if not df.is_empty() else []
    except Exception:
        log.exception("Failed to build aggregate views; serving per-repeat graphs only.")
        aggregate_entries = []
    for entry in aggregate_entries:
        original_path = entry["path"]
        entry["path"] = "/api/aggregate/" + urllib.parse.quote(original_path, safe="")
    return aggregate_entries, dict(viz.collected_bytes)


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that always disables caching — artifacts change on every run."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


class _NaNSafeJSONResponse(JSONResponse):
    """Default response class for all dict/list-returning routes below.

    Unconverged fits legitimately produce math.nan/math.inf (see
    nvision/runner/metrics.py's default estimate); sanitize_non_finite keeps
    those out of the wire format (see nvision/tools/json_sanitize.py for why).
    """

    def render(self, content: Any) -> bytes:
        return super().render(sanitize_non_finite(content))


class ReloadState:
    def __init__(self) -> None:
        self.running = False
        self.last_output = ""


def _combo_key(combo: dict) -> str:
    """The cache key this combo was stored under -- not the one the *current* code
    would write it under.

    PHYSICS_CONFIG_FINGERPRINT is folded into every combination key on purpose
    (see nvision/cache/locator_keys.py), so changing a physical constant or bound
    busts the cache for *runs* and old entries are never silently reused as if
    they were drawn under the new physics. Reading is the opposite case: results
    recorded under an older physics config are still perfectly valid results, and
    the viewer has to be able to open them. Recomputing the key from the current
    fingerprint instead of reusing the stored one made every pre-change combo
    load as an empty page -- listed by /api/combos (that index comes from the
    stored configs) but with no manifest entries behind it.

    `physics_fingerprint` comes from CacheBridge.list_combinations_with_updated_at;
    `None` means a v8-era config that predates the field, whose key was hashed
    without the field entirely rather than with some default value.
    """
    ptr_config = combination_base_cache_config(
        generator=combo["generator"],
        noise=combo["noise"],
        strategy=combo["strategy"],
        seed=combo["seed"],
        max_steps=combo["max_steps"],
        timeout_s=combo["timeout_s"],
    )
    if "physics_fingerprint" in combo:
        stored_fp = combo["physics_fingerprint"]
        if stored_fp is None:
            ptr_config.pop("physics_fingerprint", None)
        else:
            ptr_config["physics_fingerprint"] = stored_fp
    if combo.get("schema_version") is not None:
        ptr_config["schema_version"] = combo["schema_version"]
    return stable_config_hash(ptr_config)


def _find_repeat(bridge: CacheBridge, combo_key: str, repeat_idx: int) -> tuple[list[dict] | None, dict] | None:
    """The (entries, main_row) for one repeat, trying every category's store --
    combo_key alone doesn't carry which category it was recorded under (cheap:
    one indexed lookup each).

    Module-level, like _load_combo, to keep build_app's mccabe complexity down;
    shared by the /api/graph and /api/repeat-meta routes, which both need this
    same category-guessing lookup.
    """
    for cat_name in ("NVCenter", "Complementary"):
        repo = bridge.get_cache_for_category(cat_name)
        found = repo._repeats.load_repeat(combo_key, repeat_idx)
        if found is not None:
            return found
    return None


def _scan_field_row(combo: dict, repeat_entries: list[dict] | None, main_row: dict) -> dict | None:
    """One repeat's Highlights-view row (generator/noise/strategy plus its scan
    entry's series/true_params), or None if this repeat has no scan entry.

    Module-level, like _load_combo, to keep build_app's mccabe complexity down.
    """
    scan_entry = (
        next((e for e in repeat_entries if e.get("type") == "scan"), None) if repeat_entries is not None else None
    )
    if scan_entry is None:
        return None
    return {
        "generator": combo["generator"],
        "noise": combo["noise"],
        "strategy": combo["strategy"],
        "repeat": main_row.get("attempt"),
        "failure_reason": main_row.get("failure_reason"),
        "measurements": main_row.get("measurements"),
        "splitting_converged_step": main_row.get("splitting_converged_step"),
        "series": scan_entry.get("series"),
        "true_params": scan_entry.get("true_params"),
    }


def _generator_grid_info_for(bridge: CacheBridge, combo: dict) -> dict | None:
    """One generator's structural ``grid_*`` metrics from its repeat-0 scan entry --
    see ``_get_generator_grid_info``'s docstring for what these are and why repeat-0
    is enough.

    Module-level (not a build_app closure), like _load_combo: it only needs its own
    arguments, and keeping it out of build_app's body is what keeps that function's
    mccabe complexity down as more endpoints are added there.
    """
    category = CombinationGrid.generator_category(combo["generator"])
    repo = bridge.get_cache_for_category(category)
    found = repo._repeats.load_repeat(_combo_key(combo), 0)
    if found is None:
        return None
    entries, _main_row = found
    scan_entry = next((e for e in entries if e.get("type") == "scan"), None)
    metrics = (scan_entry or {}).get("metrics") or {}
    return {k: v for k, v in metrics.items() if k.startswith("grid_") and v is not None}


def _load_combo(bridge: CacheBridge, combo: dict) -> tuple[list[dict], list[dict]]:
    """One combo's (graph manifest entries, flat locator-result rows) — a single
    cache read per combo feeds both, instead of scanning the whole cache twice.

    Module-level (not a build_app closure), like _build_aggregate_entries: it only
    needs its own arguments, and keeping it out of build_app's body is what keeps
    that function's mccabe complexity down as more endpoints are added there.
    """
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
        if repeat_entries is None:
            continue
        for entry in repeat_entries:
            gtype = entry.get("type", "unknown")
            slim = _slim_manifest_entry(entry)
            for field in _BULK_STRIP_FIELDS:
                slim.pop(field, None)
            slim["path"] = f"/api/graph/{combo_key}/{repeat_idx}/{gtype}.json.gz"
            graph_entries.append(slim)
    return graph_entries, result_rows


def build_app(cache_dir: Path, run_dir: Path) -> FastAPI:
    """Build the FastAPI app serving the repo's frontend against *cache_dir*.

    *run_dir* is only needed for run-specific files that genuinely vary per
    run (``locator_results.parquet``, ``run_status.json``) — index.html, JS, CSS,
    and graph-def templates are generated/served straight from the repo's
    ``static/`` directory, never copied into *run_dir*.
    """
    app = FastAPI(default_response_class=_NaNSafeJSONResponse)
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    reload_state = ReloadState()
    # In-process caches, invalidated by /api/reload. Rebuilding the manifest means
    # walking every cached combination's :meta sidecars — cheap per call, but not
    # cheap enough to redo on every page load, so we cache it until told otherwise.
    cache: dict[str, Any] = {"manifest": None, "aggregate_bytes": {}, "combos": None, "generator_grid_info": None}
    # RLock, not Lock: _get_manifest/aggregate hold `lock` while calling
    # _build_manifest, which calls _get_combos -- itself a `with lock:` block.
    # A plain Lock would deadlock on that same-thread re-acquisition.
    lock = threading.RLock()

    def _bridge() -> CacheBridge:
        return CacheBridge(cache_dir)

    def _get_combos() -> list[dict]:
        """Cheap combo index (generator/noise/strategy/repeat count/updated_at) --
        no per-repeat scan. Cached alongside the manifest, but computable (and
        cached) independently of it: this is all the picker/generator-switcher UI
        needs, and all `/api/scan-fields` needs to resolve its own combo scope.
        """
        with lock:
            if cache["combos"] is None:
                bridge = _bridge()
                try:
                    cache["combos"] = _dedupe_latest_combos(bridge.list_combinations_with_updated_at())
                finally:
                    bridge.close()
            return cache["combos"]

    def _get_generator_grid_info() -> dict[str, dict]:
        """Structural swept-parameter coordinates (``grid_*`` -- variant, width,
        contrast, saturation, sigma_inhom, hyperfine, ...) per *distinct generator*,
        read straight from one representative repeat's own recorded metrics (see
        ``nvision/runner/metrics.py``'s grid_* extraction loop) rather than parsed
        back out of the generator name string. The UI's Study/facet pickers use
        this as their single source of truth for how to group and split generators
        into axes -- see nvision-ui's app.js, which used to duplicate this as a
        name-regex parser (and bootstrap.js duplicated a second, independent copy
        of that regex just to decide what to fetch) that could silently drift out
        of sync with whatever grid shape presets.py actually produces.

        One cheap repeat-0 read per distinct generator (bounded by the size of the
        parameter grid -- tens to low hundreds of generators -- not by repeat count
        or combo count), so this stays cheap even though /api/manifest itself is
        not. Cached alongside combos/manifest; invalidated by /api/reload.
        """
        with lock:
            if cache["generator_grid_info"] is None:
                bridge = _bridge()
                try:
                    info: dict[str, dict] = {}
                    seen: set[str] = set()
                    for combo in _get_combos():
                        gen = combo["generator"]
                        if gen in seen:
                            continue
                        seen.add(gen)
                        grid_fields = _generator_grid_info_for(bridge, combo)
                        if grid_fields:
                            info[gen] = grid_fields
                    cache["generator_grid_info"] = info
                finally:
                    bridge.close()
            return cache["generator_grid_info"]

    def _build_manifest(generators: frozenset[str] | None = None) -> list[dict]:
        """Build graph-manifest entries for the given *generators*, or every combo
        in the cache when *generators* is None.

        Scoping to a subset of generators is what keeps a page load on a large
        shared cache from having to scan every repeat of every combo up front
        (see nvision-ui's generator picker, which only loads the currently
        selected generator's data this way). Aggregate views (comparisons/
        grid-study/experiment summaries) need every combo's result rows to be
        meaningful, so they're only computed for the unscoped (full) build --
        callers that need them (the Dashboard tab) must request the full manifest.
        """
        import concurrent.futures
        import os

        bridge = _bridge()
        try:
            combos = _get_combos()
            if generators is not None:
                combos = [c for c in combos if c["generator"] in generators]
            entries: list[dict] = []
            result_rows: list[dict] = []
            n_workers = min(os.cpu_count() or 4, max(len(combos), 1), 12)
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                for combo_entries, combo_rows in pool.map(lambda c: _load_combo(bridge, c), combos):
                    entries.extend(combo_entries)
                    result_rows.extend(combo_rows)

            if generators is not None:
                return entries

            aggregate_entries, aggregate_bytes = _build_aggregate_entries(result_rows)
            # Caller (_get_manifest / reload) already holds `lock` while this runs.
            cache["aggregate_bytes"] = aggregate_bytes
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

    @app.get("/api/combos")
    def combos() -> list[dict]:
        """Cheap picker/generator-switcher data: one row per (generator, noise,
        strategy) combo with its repeat count -- no per-repeat scan. See
        nvision-ui's generator picker, which fetches this before ever asking
        for a scoped /api/manifest.
        """
        return _get_combos()

    @app.get("/api/generator-grid-info")
    def generator_grid_info() -> dict[str, dict]:
        """One cheap repeat-0 read's worth of ``grid_*`` metrics per distinct
        generator -- see ``_get_generator_grid_info``'s docstring. Fetched by
        bootstrap.js alongside /api/combos and used by both it and app.js as the
        structural source for family/facet grouping, instead of each parsing the
        generator name with its own regex."""
        return _get_generator_grid_info()

    @app.get("/api/manifest")
    def manifest(generators: str | None = None) -> list[dict]:
        """Full manifest by default; pass ``?generators=a,b`` to scope the build
        to only those generators (skips the aggregate views -- see _build_manifest).
        Scoped requests aren't cached: each one only touches a handful of combos,
        so recomputing per request is cheap and avoids per-scope cache invalidation.
        """
        if generators is None:
            return _get_manifest()
        wanted = frozenset(g for g in generators.split(",") if g)
        return _build_manifest(wanted)

    @app.get("/api/graph/{combo_key}/{repeat_idx}/{gtype_with_ext}")
    def graph(combo_key: str, repeat_idx: int, gtype_with_ext: str) -> Response:
        gtype = gtype_with_ext.removesuffix(".json.gz")
        bridge = _bridge()
        try:
            found = _find_repeat(bridge, combo_key, repeat_idx)
            if found is None:
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
            found = _find_repeat(bridge, combo_key, repeat_idx)
            if found is None:
                raise HTTPException(status_code=404, detail="No such repeat")
            entries, _main_row = found
            scan_entry = next((e for e in entries if e.get("type") == "scan"), None) if entries is not None else None
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

        combos = _get_combos()
        bridge = _bridge()
        try:
            matching = [c for c in combos if c["generator"] in wanted_generators and c["noise"] in wanted_noises]
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
                    row = _scan_field_row(combo, repeat_entries, main_row)
                    if row is not None:
                        out.append(row)
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
                    cache["generator_grid_info"] = None
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
        return HTMLResponse(render_index_html(run_dir, live=True), headers={"Cache-Control": "no-store"})

    @app.get("/{asset_name}")
    def static_asset(asset_name: str) -> FileResponse:
        if asset_name in _STATIC_ASSET_NAMES:
            return FileResponse(_STATIC_DIR / asset_name, headers={"Cache-Control": "no-store"})
        # Not one of the repo-wide JS/CSS assets — fall through to run-specific files
        # (locator_results.parquet, run_status.json, a stale plots_manifest.json.gz from an
        # older static export, ...). This route is registered before app.mount("/", ...)
        # below, so without this fallback it shadows the mount for every single-segment
        # path and any such file 404s even though it's sitting right there in run_dir.
        run_path = (run_dir / asset_name).resolve()
        if run_path.is_file() and run_path.is_relative_to(run_dir.resolve()):
            return FileResponse(run_path, headers={"Cache-Control": "no-store"})
        raise HTTPException(status_code=404, detail="No such asset")

    @app.get("/graphs/{def_name}")
    def graph_def(def_name: str) -> FileResponse:
        path = _STATIC_DIR / "graphs" / def_name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="No such graph definition")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    # Run-specific files (locator_results.parquet, ...) last — anything not matched
    # by a route above falls through to whatever's physically in run_dir.
    app.mount("/", _NoCacheStaticFiles(directory=str(run_dir)), name="run_dir")

    return app
