"""``purge_cached_combination`` must be a keyed operation, not a scan of the whole cache.

The scan version (``for k in backend: backend.get(k)``) took minutes per combination on a ~1M-key
cache and stalled every ``--no-cache`` runner at once; these tests pin both behaviour and the
absence of any whole-cache iteration.
"""

import base64
import logging

import polars as pl
import pytest

from nvision.cache.data_store import CategoryDataStore
from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_keys import combination_base_cache_config
from nvision.cache.locator_repository import LocatorResultsRepository
from nvision.cache.repeats_repository import RepeatsRepository

log = logging.getLogger("test_purge_cached_combination")


def _ident(name: str) -> dict:
    return {
        "generator": name,
        "noise": "Gauss(0.002)",
        "strategy": "Bayesian-SBED",
        "seed": 1,
        "max_steps": 10,
        "timeout_s": 10,
    }


@pytest.fixture
def store(tmp_path):
    return CategoryDataStore(tmp_path / "test.db")


@pytest.fixture
def repo(store):
    return LocatorResultsRepository(store)


def _save(repo: LocatorResultsRepository, name: str, n: int = 2) -> str:
    repo.append_cached_repeats(
        **_ident(name), new_results=[([{"type": "scan"}], {"idx": i}) for i in range(n)], start_idx=0
    )
    return stable_config_hash(combination_base_cache_config(**_ident(name)))


def _purge(repo: LocatorResultsRepository, name: str, repeats: int = 2) -> None:
    repo.purge_cached_combination(repeats=repeats, **_ident(name))


def _cached(repo: LocatorResultsRepository, name: str, repeats: int = 2):
    return repo.get_cached_combination(repeats=repeats, **_ident(name))


def test_purge_removes_live_combination_and_keeps_others(repo, store):
    key_a = _save(repo, "a")
    key_b = _save(repo, "b")
    assert _cached(repo, "a") is not None

    _purge(repo, "a")

    assert _cached(repo, "a") is None
    assert not store.backend.keys_exist_batch(
        [key_a, RepeatsRepository.make_repeat_key(key_a, 0), RepeatsRepository.make_repeat_key(key_a, 1) + ":meta"]
    )
    assert _cached(repo, "b") is not None
    assert key_b in store.backend.keys_exist_batch([key_b])


def test_purge_removes_archived_combination(repo, store):
    key = _save(repo, "a")
    assert repo.archive_if_complete(**_ident("a"), target_repeats=2, log=log)
    assert store.backend.archive.has_combo(key)

    _purge(repo, "a")

    assert not store.backend.archive.has_combo(key)  # stale archived repeats must not show through
    assert _cached(repo, "a") is None


def test_purge_removes_legacy_v8_pointer(repo, store):
    cfg = combination_base_cache_config(**_ident("a"))
    cfg["schema_version"] = 8
    cfg.pop("physics_fingerprint", None)
    key = stable_config_hash(cfg)
    store.save_df(pl.DataFrame({"achieved_repeats": [1], "streaming": [True]}), key, metadata={"config": cfg})
    assert store.backend.keys_exist_batch([key]) == {key}

    _purge(repo, "a")

    assert store.backend.keys_exist_batch([key]) == set()


def test_purge_removes_blob_rows(repo, store):
    """A repeat's graph blobs (RepeatsRepository._extract_blobs) must not outlive the purge."""
    entries = [{"type": "scan", "content_bin": base64.b85encode(b"plot-bytes").decode("ascii")}]
    repo.append_cached_repeats(**_ident("a"), new_results=[(entries, {"idx": 0})], start_idx=0)
    combo_key = stable_config_hash(combination_base_cache_config(**_ident("a")))
    blob_key = f"blob:{combo_key}:0:scan"
    assert store.backend.blob_get(blob_key) is not None

    _purge(repo, "a", repeats=1)

    assert store.backend.blob_get(blob_key) is None


def test_purge_of_unknown_combination_is_a_noop(repo):
    _save(repo, "b")
    _purge(repo, "never-saved")
    assert _cached(repo, "b") is not None


def test_purge_never_scans_the_whole_cache(repo, store, monkeypatch):
    _save(repo, "a")
    _save(repo, "b")

    backend_type = type(store.backend)

    def _boom(*_a, **_kw):
        raise AssertionError("purge_cached_combination scanned the whole cache")

    monkeypatch.setattr(backend_type, "__iter__", _boom)
    monkeypatch.setattr(backend_type, "get", _boom)
    monkeypatch.setattr(backend_type, "list_keys_excluding_prefixes", _boom)

    _purge(repo, "a")

    monkeypatch.undo()
    assert _cached(repo, "a") is None
    assert _cached(repo, "b") is not None
