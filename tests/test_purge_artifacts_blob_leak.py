"""Blob rows (graph/plot bytes) must not survive a --purge / --no-cache purge.

Mirrors tests/test_purge_cached_combination.py::test_purge_removes_blob_rows, but for the
batch purge helpers used by `nv run --purge` / `nv groups --purge` (purge_cache_and_artifacts_
for_combinations / _for_strategies), which had the same missing-blob-key omission.
"""

import base64
import logging

from nvision.cache.bridge import CacheBridge
from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_keys import combination_base_cache_config
from nvision.sim.combinations import CombinationGrid
from nvision.tools.artifacts import (
    purge_cache_and_artifacts_for_combinations,
    purge_cache_and_artifacts_for_strategies,
)

log = logging.getLogger("test_purge_artifacts_blob_leak")


def _ident(generator: str = "NVCenter-lorentzian") -> dict:
    return {
        "generator": generator,
        "noise": "Gauss(0.002)",
        "strategy": "Bayesian-SBED",
        "seed": 1,
        "max_steps": 10,
        "timeout_s": 10,
    }


def _save_with_blob(out_dir, ident: dict) -> str:
    bridge = CacheBridge(out_dir / "cache")
    try:
        category = CombinationGrid.generator_category(ident["generator"])
        repo = bridge.get_cache_for_category(category)
        entries = [{"type": "scan", "content_bin": base64.b85encode(b"plot-bytes").decode("ascii")}]
        repo.append_cached_repeats(**ident, new_results=[(entries, {"idx": 0})], start_idx=0)
        combo_key = stable_config_hash(combination_base_cache_config(**ident))
        assert bridge.get_cache_for_category(category).backend.blob_get(f"blob:{combo_key}:0:scan") is not None
        return combo_key
    finally:
        bridge.close()


def _blob_survives(out_dir, ident: dict, combo_key: str) -> bool:
    bridge = CacheBridge(out_dir / "cache")
    try:
        category = CombinationGrid.generator_category(ident["generator"])
        backend = bridge.get_cache_for_category(category).backend
        return backend.blob_get(f"blob:{combo_key}:0:scan") is not None
    finally:
        bridge.close()


def test_purge_by_combination_removes_blob_rows(tmp_path):
    ident = _ident()
    combo_key = _save_with_blob(tmp_path, ident)

    deleted = purge_cache_and_artifacts_for_combinations(
        tmp_path, {(ident["generator"], ident["noise"], ident["strategy"])}, log
    )

    assert deleted == 1
    assert not _blob_survives(tmp_path, ident, combo_key)


def test_purge_by_strategy_removes_blob_rows(tmp_path):
    ident = _ident()
    combo_key = _save_with_blob(tmp_path, ident)

    deleted = purge_cache_and_artifacts_for_strategies(tmp_path, {ident["strategy"]}, log)

    assert deleted == 1
    assert not _blob_survives(tmp_path, ident, combo_key)
