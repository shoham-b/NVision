"""`nv cache clean` must not leave graph blob rows behind (see test_purge_cached_combination.py
and test_purge_artifacts_blob_leak.py for the same bug in the other purge call sites)."""

import base64

from typer.testing import CliRunner

from nvision.cache.bridge import CacheBridge
from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_keys import combination_base_cache_config
from nvision.cli.cache_cmd import cache_app
from nvision.sim.combinations import CombinationGrid

runner = CliRunner()

_IDENT = {
    "generator": "NVCenter-lorentzian",
    "noise": "Gauss(0.002)",
    "strategy": "Bayesian-SBED",
    "seed": 1,
    "max_steps": 10,
    "timeout_s": 10,
}


def test_cache_clean_removes_blob_rows(tmp_path):
    out = tmp_path / "artifacts"
    bridge = CacheBridge(out / "cache")
    category = CombinationGrid.generator_category(_IDENT["generator"])
    repo = bridge.get_cache_for_category(category)
    entries = [{"type": "scan", "content_bin": base64.b85encode(b"plot-bytes").decode("ascii")}]
    repo.append_cached_repeats(**_IDENT, new_results=[(entries, {"idx": 0})], start_idx=0)
    combo_key = stable_config_hash(combination_base_cache_config(**_IDENT))
    blob_key = f"blob:{combo_key}:0:scan"
    assert bridge.get_cache_for_category(category).backend.blob_get(blob_key) is not None
    bridge.close()

    result = runner.invoke(
        cache_app,
        ["clean", "--out", str(out), "--generator", _IDENT["generator"], "--strategy", _IDENT["strategy"], "--force"],
    )
    assert result.exit_code == 0, result.output

    bridge = CacheBridge(out / "cache")
    try:
        assert bridge.get_cache_for_category(category).backend.blob_get(blob_key) is None
    finally:
        bridge.close()
