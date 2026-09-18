import base64
import logging

import polars as pl
import pytest

from nvision.cache.data_store import CategoryDataStore
from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_keys import combination_base_cache_config
from nvision.cache.locator_repository import LocatorResultsRepository
from nvision.cache.parquet_archive import ComboArchive, archive_combination
from nvision.cache.repeats_repository import RepeatsRepository

log = logging.getLogger("test_parquet_archive")


@pytest.fixture
def store(tmp_path):
    return CategoryDataStore(tmp_path / "test.db")


@pytest.fixture
def repo(store):
    return LocatorResultsRepository(store)


def _ptr_key(**kw):
    cfg = combination_base_cache_config(**kw)
    return stable_config_hash(cfg)


# -- ComboArchive direct round-trip --------------------------------------------------


def test_combo_archive_write_and_read_roundtrip(tmp_path):
    archive = ComboArchive(tmp_path / "archive")
    combo_key = "abc123"
    rows = [
        (combo_key, "json", b'{"achieved_repeats": 2}'),
        (f"repeat:{combo_key}:0", "json", b'{"entries": []}'),
        (f"blob:{combo_key}:0:scan", "blob", b"\x1f\x8bgzipped-bytes"),
    ]
    archive.write_combo(combo_key, rows)

    assert archive.has_combo(combo_key)
    assert archive.get(combo_key) == {"achieved_repeats": 2}
    assert archive.get(f"repeat:{combo_key}:0") == {"entries": []}
    assert archive.blob_get(f"blob:{combo_key}:0:scan") == b"\x1f\x8bgzipped-bytes"
    assert archive.get(f"blob:{combo_key}:0:scan") is None  # wrong accessor for a blob row
    assert archive.blob_get(combo_key) is None  # wrong accessor for a json row


def test_combo_archive_batch_and_listing(tmp_path):
    archive = ComboArchive(tmp_path / "archive")
    archive.write_combo("c1", [("c1", "json", b"{}"), (f"repeat:c1:0", "json", b'{"a": 1}')])
    archive.write_combo("c2", [("c2", "json", b"{}"), (f"repeat:c2:0", "json", b'{"a": 2}')])

    batch = archive.batch_get(["repeat:c1:0", "repeat:c2:0", "repeat:missing:0"])
    assert batch == {"repeat:c1:0": {"a": 1}, "repeat:c2:0": {"a": 2}}

    exist = archive.keys_exist_batch(["c1", "c2", "nope"])
    assert exist == {"c1", "c2"}

    pointer_keys = archive.list_keys_excluding_prefixes(["repeat:", "blob:"])
    assert set(pointer_keys) == {"c1", "c2"}

    assert set(archive.iter_keys()) == {"c1", "repeat:c1:0", "c2", "repeat:c2:0"}


def test_combo_archive_caches_loaded_df_and_invalidates_on_write(tmp_path, monkeypatch):
    archive = ComboArchive(tmp_path / "archive")
    archive.write_combo("c1", [("c1", "json", b'{"n": 1}')])

    read_calls = []
    real_read_parquet = pl.read_parquet

    def _counting_read_parquet(path, *a, **kw):
        read_calls.append(path)
        return real_read_parquet(path, *a, **kw)

    import nvision.cache.parquet_archive as mod

    monkeypatch.setattr(mod.pl, "read_parquet", _counting_read_parquet)

    # Several lookups against the same combo should only hit disk once.
    for _ in range(5):
        assert archive.get("c1") == {"n": 1}
    assert len(read_calls) == 1

    # Overwriting the combo invalidates the cached DataFrame.
    archive.write_combo("c1", [("c1", "json", b'{"n": 2}')])
    assert archive.get("c1") == {"n": 2}
    assert len(read_calls) == 2


def test_combo_archive_cache_is_bounded(tmp_path):
    archive = ComboArchive(tmp_path / "archive")
    from nvision.cache.parquet_archive import _DF_CACHE_MAX_COMBOS

    for i in range(_DF_CACHE_MAX_COMBOS + 10):
        key = f"c{i}"
        archive.write_combo(key, [(key, "json", b"{}")])
        archive.get(key)  # populate the cache

    assert len(archive._df_cache) <= _DF_CACHE_MAX_COMBOS


def test_combo_archive_delete_combo(tmp_path):
    archive = ComboArchive(tmp_path / "archive")
    archive.write_combo("c1", [("c1", "json", b"{}")])
    assert archive.has_combo("c1")
    archive.delete_combo("c1")
    assert not archive.has_combo("c1")
    # Deleting an already-absent combo is a no-op, not an error.
    archive.delete_combo("c1")


# -- archive_combination() end-to-end via the real cache stack ------------------------


def test_archive_combination_moves_data_and_stays_readable(repo, store):
    entries = [{"type": "scan", "content_bin": base64.b85encode(b"plot-bytes-0").decode("ascii")}]
    repo.append_cached_repeats(
        generator="gen",
        noise="noise",
        strategy="strat",
        seed=1,
        max_steps=10,
        timeout_s=10,
        new_results=[(entries, {"abs_err_x": 0.1})],
        start_idx=0,
    )

    ptr_key = _ptr_key(generator="gen", noise="noise", strategy="strat", seed=1, max_steps=10, timeout_s=10)

    ok = archive_combination(store, ptr_key, achieved_repeats=1, log=log)
    assert ok

    # Live SQLite no longer has any of this combo's rows.
    live = store.backend._live
    assert live.get(ptr_key) is None
    assert live.get(RepeatsRepository.make_repeat_key(ptr_key, 0)) is None
    assert live.blob_get(f"blob:{ptr_key}:0:scan") is None

    # But every existing read path still works transparently.
    assert store.load_df(ptr_key) is not None
    loaded = repo.get_cached_combination(
        generator="gen", noise="noise", strategy="strat", repeats=1, seed=1, max_steps=10, timeout_s=10
    )
    assert loaded is not None
    assert len(loaded) == 1
    loaded_entries, loaded_row = loaded[0]
    assert loaded_row == {"abs_err_x": 0.1}
    assert base64.b85decode(loaded_entries[0]["content_bin"]) == b"plot-bytes-0"


def test_archive_combination_is_idempotent_and_mergeable(repo, store):
    """Archiving, then appending more repeats, then archiving again re-archives the merged set."""
    repo.append_cached_repeats(
        generator="gen",
        noise="noise",
        strategy="strat",
        seed=1,
        max_steps=10,
        timeout_s=10,
        new_results=[([{"type": "scan"}], {"idx": 0})],
        start_idx=0,
    )
    ptr_key = _ptr_key(generator="gen", noise="noise", strategy="strat", seed=1, max_steps=10, timeout_s=10)
    assert archive_combination(store, ptr_key, achieved_repeats=1, log=log)

    # More repeats land in live storage after the first archive pass.
    repo.append_cached_repeats(
        generator="gen",
        noise="noise",
        strategy="strat",
        seed=1,
        max_steps=10,
        timeout_s=10,
        new_results=[([{"type": "scan"}], {"idx": 1})],
        start_idx=1,
    )

    # Reading both old (archived) and new (live) repeats works before re-archiving.
    loaded = repo.get_cached_combination(
        generator="gen", noise="noise", strategy="strat", repeats=2, seed=1, max_steps=10, timeout_s=10
    )
    assert len(loaded) == 2

    assert archive_combination(store, ptr_key, achieved_repeats=2, log=log)
    live = store.backend._live
    assert live.get(RepeatsRepository.make_repeat_key(ptr_key, 1)) is None

    loaded_again = repo.get_cached_combination(
        generator="gen", noise="noise", strategy="strat", repeats=2, seed=1, max_steps=10, timeout_s=10
    )
    assert len(loaded_again) == 2
    assert loaded_again[0][1]["idx"] == 0
    assert loaded_again[1][1]["idx"] == 1


def test_archive_combination_skips_when_pointer_missing(store):
    assert archive_combination(store, "nonexistent", achieved_repeats=3, log=log) is False


def test_archive_if_complete_below_and_at_target(repo, store):
    repo.append_cached_repeats(
        generator="gen",
        noise="noise",
        strategy="strat",
        seed=1,
        max_steps=10,
        timeout_s=10,
        new_results=[([{"type": "scan"}], {"idx": 0})],
        start_idx=0,
    )

    # Below target: no-op, nothing archived.
    archived = repo.archive_if_complete(
        generator="gen", noise="noise", strategy="strat", seed=1, max_steps=10, timeout_s=10, target_repeats=2
    )
    assert archived is False
    ptr_key = _ptr_key(generator="gen", noise="noise", strategy="strat", seed=1, max_steps=10, timeout_s=10)
    assert not store.backend.archive.has_combo(ptr_key)

    # At target: archives.
    archived = repo.archive_if_complete(
        generator="gen", noise="noise", strategy="strat", seed=1, max_steps=10, timeout_s=10, target_repeats=1
    )
    assert archived is True
    assert store.backend.archive.has_combo(ptr_key)

    # Still readable normally afterward.
    loaded = repo.get_cached_combination(
        generator="gen", noise="noise", strategy="strat", repeats=1, seed=1, max_steps=10, timeout_s=10
    )
    assert len(loaded) == 1


def test_archive_if_complete_missing_pointer_is_noop(repo, store):
    archived = repo.archive_if_complete(
        generator="gen", noise="noise", strategy="strat", seed=1, max_steps=10, timeout_s=10, target_repeats=1
    )
    assert archived is False


def test_delete_many_purges_archived_combo(repo, store):
    repo.append_cached_repeats(
        generator="gen",
        noise="noise",
        strategy="strat",
        seed=1,
        max_steps=10,
        timeout_s=10,
        new_results=[([{"type": "scan"}], {"idx": 0})],
        start_idx=0,
    )
    ptr_key = _ptr_key(generator="gen", noise="noise", strategy="strat", seed=1, max_steps=10, timeout_s=10)
    assert archive_combination(store, ptr_key, achieved_repeats=1, log=log)
    assert store.backend.archive.has_combo(ptr_key)

    # Mirrors nv cache clean / purge_cache_and_artifacts_for_combinations: delete the
    # pointer plus every speculative repeat/meta index in one delete_many call.
    keys = [ptr_key] + [RepeatsRepository.make_repeat_key(ptr_key, i) for i in range(5)]
    keys += [k + ":meta" for k in keys[1:]]
    store.backend.delete_many(keys)

    assert not store.backend.archive.has_combo(ptr_key)
    assert store.load_df(ptr_key) is None
