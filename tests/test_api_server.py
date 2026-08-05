import base64
import gzip

from fastapi.testclient import TestClient

from nvision.cache import CacheBridge
from nvision.cli.api_server import build_app
from nvision.sim.defaults import NVISION_SIMPLESWEEP_MAX_STEPS

_MAIN_RESULT_ROW = {
    "generator": "NVCenter-lorentzian",
    "noise": "Gauss(0.01)",
    "strategy": "SimpleSweep",
    "seed": 1,
    "attempt": 1,
    "abs_err_x": 123.0,
    "measurements": 10,
    "duration_ms": 5.0,
    "acquisition_hi": 2960000000.0,
    "acquisition_lo": 2780000000.0,
    "expected_uniform_points": 87.0,
}


def _seed_one_repeat(cache_dir, *, entry_extra: dict | None = None):
    bridge = CacheBridge(cache_dir)
    repo = bridge.get_cache_for_category("NVCenter")
    payload = gzip.compress(b'{"hello": "world"}')
    entry = {
        "path": "graphs/scans/x_r1.json.gz",
        "type": "scan",
        "content_bin": base64.b85encode(payload).decode("ascii"),
        "generator": "NVCenter-lorentzian",
        "noise": "Gauss(0.01)",
        "strategy": "SimpleSweep",
    }
    if entry_extra:
        entry.update(entry_extra)
    repo.save_cached_combination(
        generator="NVCenter-lorentzian",
        noise="Gauss(0.01)",
        strategy="SimpleSweep",
        repeats=1,
        seed=1,
        max_steps=NVISION_SIMPLESWEEP_MAX_STEPS,
        timeout_s=10,
        repeat_offset=0,
        results=[([entry], dict(_MAIN_RESULT_ROW))],
    )
    bridge.close()
    return payload


def _build_client(tmp_path):
    """(cache_dir, run_dir) — index.html/app.js/graph-defs are no longer per-run
    files; build_app serves them straight from the repo's static/ directory."""
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return cache_dir, run_dir


def test_status_endpoint(tmp_path):
    cache_dir, static_dir = _build_client(tmp_path)
    client = TestClient(build_app(cache_dir, static_dir))
    r = client.get("/api/status")
    assert r.status_code == 200
    assert r.json() == {"reload_running": False, "last_output": ""}


def test_graph_endpoint_returns_correct_gzip_bytes(tmp_path):
    cache_dir, static_dir = _build_client(tmp_path)
    _seed_one_repeat(cache_dir)
    client = TestClient(build_app(cache_dir, static_dir))

    manifest = client.get("/api/manifest").json()
    scan = next(e for e in manifest if e["type"] == "scan")

    r = client.get(scan["path"])
    assert r.status_code == 200
    # "identity", not "gzip": the payload must reach the client as opaque
    # raw bytes for its own DecompressionStream to handle (see api_server.py's
    # comment on this header) — if this were "gzip", httpx/the browser would
    # transparently gunzip it here, which is exactly the bug that shipped.
    assert r.headers["content-encoding"] == "identity"
    assert gzip.decompress(r.content) == b'{"hello": "world"}'


def test_graph_endpoint_404_for_unknown_combo(tmp_path):
    cache_dir, static_dir = _build_client(tmp_path)
    client = TestClient(build_app(cache_dir, static_dir))
    r = client.get("/api/graph/deadbeefdeadbeef/0/scan.json.gz")
    assert r.status_code == 404


def test_graph_endpoint_404_for_unknown_type(tmp_path):
    cache_dir, static_dir = _build_client(tmp_path)
    _seed_one_repeat(cache_dir)
    client = TestClient(build_app(cache_dir, static_dir))
    manifest = client.get("/api/manifest").json()
    scan = next(e for e in manifest if e["type"] == "scan")
    combo_key = scan["path"].split("/")[3]
    r = client.get(f"/api/graph/{combo_key}/0/bayesian_posterior_data.json.gz")
    assert r.status_code == 404


def test_manifest_does_not_write_any_files(tmp_path):
    cache_dir, static_dir = _build_client(tmp_path)
    _seed_one_repeat(cache_dir)
    client = TestClient(build_app(cache_dir, static_dir))

    r = client.get("/api/manifest")
    assert r.status_code == 200
    assert len(r.json()) >= 1

    # No plots_manifest.json, no graphs/ directory — this must all be in-memory.
    assert not (tmp_path / "plots_manifest.json").exists()
    assert not (static_dir / "plots_manifest.json").exists()
    assert not (static_dir / "graphs").exists()


def test_manifest_includes_aggregate_view_with_working_path(tmp_path):
    cache_dir, static_dir = _build_client(tmp_path)
    _seed_one_repeat(cache_dir)
    client = TestClient(build_app(cache_dir, static_dir))

    manifest = client.get("/api/manifest").json()
    agg = next((e for e in manifest if e["type"] != "scan"), None)
    assert agg is not None
    assert agg["path"].startswith("/api/aggregate/")

    r = client.get(agg["path"])
    assert r.status_code == 200
    assert len(r.content) > 0


def test_aggregate_endpoint_matches_direct_viz_call(tmp_path):
    """The API's aggregate bytes must be identical to calling the viz function
    directly — proof no aggregate computation changed."""
    import urllib.parse
    from pathlib import Path

    from nvision.cache.aggregate import build_locator_results_df
    from nvision.viz import Viz

    cache_dir, static_dir = _build_client(tmp_path)
    _seed_one_repeat(cache_dir)
    client = TestClient(build_app(cache_dir, static_dir))

    manifest = client.get("/api/manifest").json()
    agg = next((e for e in manifest if e["type"] != "scan"), None)
    assert agg is not None
    api_bytes = client.get(agg["path"]).content

    bridge = CacheBridge(cache_dir)
    try:
        df = build_locator_results_df(bridge)
    finally:
        bridge.close()
    # Viz never touches disk (see VizBase._emit's docstring) -- collected_bytes
    # is always the in-memory store, so no extra flag is needed here.
    viz = Viz(Path("."))
    direct_entries = viz.plot_locator_summary(df)
    original_key = urllib.parse.unquote(agg["path"].removeprefix("/api/aggregate/"))
    direct_entry = next(e for e in direct_entries if e["path"] == original_key)
    direct_bytes = viz.collected_bytes[direct_entry["path"]]

    # "identity" encoding means the test client does NOT auto-decompress —
    # api_bytes is the same raw gzip bytes dump_gz/write_plotly_gz produced.
    assert api_bytes == direct_bytes


def test_reload_invalidates_manifest_cache(tmp_path):
    cache_dir, static_dir = _build_client(tmp_path)
    client = TestClient(build_app(cache_dir, static_dir))

    r1 = client.get("/api/manifest")
    assert r1.json() == []

    _seed_one_repeat(cache_dir)

    # Without reload, the in-process cache is stale (still empty).
    r2 = client.get("/api/manifest")
    assert r2.json() == []

    r3 = client.post("/api/reload")
    assert r3.status_code == 200
    assert r3.json()["status"] == "started"

    import time

    for _ in range(50):
        if not client.get("/api/status").json()["reload_running"]:
            break
        time.sleep(0.05)

    r4 = client.get("/api/manifest")
    assert len(r4.json()) >= 1


def test_index_generated_live(tmp_path):
    """index.html is generated on the fly from the repo template, never a
    per-run file — window.MANIFEST must be null so the browser always fetches
    /api/manifest instead of expecting an inlined/static array."""
    cache_dir, run_dir = _build_client(tmp_path)
    client = TestClient(build_app(cache_dir, run_dir))
    r = client.get("/")
    assert r.status_code == 200
    assert "NVision" in r.text
    assert "window.MANIFEST = null;" in r.text
    assert not (run_dir / "index.html").exists()


def test_js_and_graph_defs_served_from_repo(tmp_path):
    """JS/CSS/graph-def templates come straight from the repo's static/
    directory — no per-run copy needed for nv serve to work."""
    cache_dir, run_dir = _build_client(tmp_path)
    client = TestClient(build_app(cache_dir, run_dir))

    r = client.get("/app.js")
    assert r.status_code == 200
    assert len(r.content) > 0

    r2 = client.get("/graphs/scan.json")
    assert r2.status_code == 200
    assert r2.json()["version"] == 1

    r3 = client.get("/no-such-asset.js")
    assert r3.status_code == 404


_BULK_FIELDS_EXTRA = {
    "series": {"s": [1, 2, 3], "e": [10.0, 5.0, 1.0], "u": [20.0, 8.0, 2.0], "tau": 1.0},
    "true_params": {"label": "True Signal Parameters", "params": {"frequency": 2.87e9}},
    "metrics": {"abs_err_x": 123.0, "splitting_converged_step": 2},
}


def test_manifest_strips_bulk_fields(tmp_path):
    """series/true_params/metrics must not appear in the bulk manifest — they're
    the fields driving the 653MB size blowup found against the real cache."""
    cache_dir, static_dir = _build_client(tmp_path)
    _seed_one_repeat(cache_dir, entry_extra=dict(_BULK_FIELDS_EXTRA))
    client = TestClient(build_app(cache_dir, static_dir))

    manifest = client.get("/api/manifest").json()
    scan = next(e for e in manifest if e["type"] == "scan")
    assert "series" not in scan
    assert "true_params" not in scan
    assert "metrics" not in scan
    # Everything else the picker/dropdown UI needs must still be present.
    assert scan["generator"] == "NVCenter-lorentzian"
    assert scan["path"].startswith("/api/graph/")


def test_repeat_meta_endpoint_returns_stripped_fields(tmp_path):
    cache_dir, static_dir = _build_client(tmp_path)
    _seed_one_repeat(cache_dir, entry_extra=dict(_BULK_FIELDS_EXTRA))
    client = TestClient(build_app(cache_dir, static_dir))

    manifest = client.get("/api/manifest").json()
    scan = next(e for e in manifest if e["type"] == "scan")
    combo_key, repeat_idx = scan["path"].split("/")[3], scan["path"].split("/")[4]

    r = client.get(f"/api/repeat-meta/{combo_key}/{repeat_idx}")
    assert r.status_code == 200
    data = r.json()
    assert data["series"] == _BULK_FIELDS_EXTRA["series"]
    assert data["true_params"] == _BULK_FIELDS_EXTRA["true_params"]
    assert data["metrics"] == _BULK_FIELDS_EXTRA["metrics"]


def test_repeat_meta_404_for_unknown_repeat(tmp_path):
    cache_dir, static_dir = _build_client(tmp_path)
    client = TestClient(build_app(cache_dir, static_dir))
    r = client.get("/api/repeat-meta/deadbeefdeadbeef/0")
    assert r.status_code == 404


def test_scan_fields_endpoint_scoped_by_selection(tmp_path):
    cache_dir, static_dir = _build_client(tmp_path)
    _seed_one_repeat(cache_dir, entry_extra=dict(_BULK_FIELDS_EXTRA))
    client = TestClient(build_app(cache_dir, static_dir))

    r = client.get("/api/scan-fields", params={"generator": "NVCenter-lorentzian", "noise": "Gauss(0.01)"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["series"] == _BULK_FIELDS_EXTRA["series"]
    assert data[0]["true_params"] == _BULK_FIELDS_EXTRA["true_params"]
    assert data[0]["generator"] == "NVCenter-lorentzian"
    assert data[0]["strategy"] == "SimpleSweep"

    # A selection matching nothing returns an empty list, not an error.
    r2 = client.get("/api/scan-fields", params={"generator": "NVCenter-voigt", "noise": "Gauss(0.02)"})
    assert r2.status_code == 200
    assert r2.json() == []
