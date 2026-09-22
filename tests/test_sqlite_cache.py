import sqlite3
from pathlib import Path

import pytest

from nvision.cache.sqlite import _init_schema_with_recovery


def test_sqlite_cache_normal_init(tmp_path: Path):
    db_path = tmp_path / "normal.db"
    schema_sql = "CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT);"

    _init_schema_with_recovery(db_path, schema_sql)

    assert db_path.exists()

    # Verify the schema was created
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table';")
        assert cursor.fetchone() is not None


def test_sqlite_cache_recovery_from_corrupted_file(tmp_path: Path):
    db_path = tmp_path / "corrupt.db"

    # Write garbage data to simulate a corrupted database file
    db_path.write_text("This is definitely not a sqlite database. Garbage data!")

    schema_sql = "CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT);"

    # This should recover by unlinking the corrupted file and creating a new DB
    _init_schema_with_recovery(db_path, schema_sql)

    assert db_path.exists()

    # Verify the schema was created
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table';")
        assert cursor.fetchone() is not None


def test_sqlite_cache_recovery_other_db_error_raises(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "error.db"
    schema_sql = "CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT);"

    # We want to patch sqlite3.connect to raise a DatabaseError that DOES NOT say "file is not a database"
    def mock_connect(*args, **kwargs):
        raise sqlite3.DatabaseError("Some other database error occurred")

    monkeypatch.setattr("nvision.cache.sqlite.sqlite3.connect", mock_connect)

    with pytest.raises(sqlite3.DatabaseError, match="Some other database error occurred"):
        _init_schema_with_recovery(db_path, schema_sql)


def test_sharded_sqlite_cache_iter_index_error(tmp_path, monkeypatch):
    from nvision.cache.sqlite import ShardedSqliteCache

    cache = ShardedSqliteCache(tmp_path / "base.db")

    def mock_get_index_conn(*args, **kwargs):
        raise Exception("Simulated index DB error")

    monkeypatch.setattr(cache, "_get_index_conn", mock_get_index_conn)

    keys = list(cache)
    assert keys == []


def test_sharded_sqlite_cache_iter_legacy_error(tmp_path, monkeypatch):
    from nvision.cache.sqlite import ShardedSqliteCache

    # Create a dummy base db to trigger legacy path behavior
    base_db = tmp_path / "legacy.db"
    base_db.write_text("dummy")

    cache = ShardedSqliteCache(base_db)

    def mock_get_conn_for_path(*args, **kwargs):
        raise Exception("Simulated legacy DB error")

    monkeypatch.setattr(cache, "_get_conn_for_path", mock_get_conn_for_path)

    keys = list(cache)
    assert keys == []


def test_sharded_sqlite_cache_iter_yields_legacy_if_index_fails(tmp_path, monkeypatch):
    import sqlite3

    from nvision.cache.sqlite import ShardedSqliteCache

    # Create a dummy base db to trigger legacy path behavior
    base_db = tmp_path / "legacy_working.db"
    base_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(base_db) as conn:
        conn.execute("CREATE TABLE cache (key TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO cache (key) VALUES ('legacy_key_1')")

    cache = ShardedSqliteCache(base_db)

    def mock_get_index_conn(*args, **kwargs):
        raise Exception("Simulated index DB error")

    monkeypatch.setattr(cache, "_get_index_conn", mock_get_index_conn)

    keys = list(cache)
    assert keys == ["legacy_key_1"]


def test_list_keys_excluding_prefixes_filters_in_sql(tmp_path):
    """Same result set as filtering `list(cache)` in Python, but computed via the
    index DB's own WHERE clause -- this is what keeps /api/manifest from having to
    stream every repeat:/blob: key in a large cache into Python just to discard it."""
    from nvision.cache.sqlite import ShardedSqliteCache

    cache = ShardedSqliteCache(tmp_path / "base.db")
    cache.set("combo:aaa", {"config": {"kind": "locator_combination_pointer"}})
    cache.set("combo:bbb", {"config": {"kind": "locator_combination_pointer"}})
    cache.set("repeat:aaa:0", {"entries": []})
    cache.set("repeat:aaa:0:meta", {"entries": []})
    cache.blob_set("blob:aaa:0:scan", b"raw-bytes")

    kept = cache.list_keys_excluding_prefixes(["repeat:", "blob:"])

    assert sorted(kept) == ["combo:aaa", "combo:bbb"]


def test_list_keys_excluding_prefixes_includes_legacy_keys(tmp_path):
    import sqlite3

    from nvision.cache.sqlite import ShardedSqliteCache

    base_db = tmp_path / "legacy_working.db"
    base_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(base_db) as conn:
        conn.execute("CREATE TABLE cache (key TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO cache (key) VALUES ('combo:legacy')")
        conn.execute("INSERT INTO cache (key) VALUES ('repeat:legacy:0')")

    cache = ShardedSqliteCache(base_db)

    kept = cache.list_keys_excluding_prefixes(["repeat:", "blob:"])

    assert kept == ["combo:legacy"]


def test_list_keys_excluding_prefixes_empty_prefixes_returns_all(tmp_path):
    from nvision.cache.sqlite import ShardedSqliteCache

    cache = ShardedSqliteCache(tmp_path / "base.db")
    cache.set("a", {})
    cache.set("b", {})

    assert sorted(cache.list_keys_excluding_prefixes([])) == ["a", "b"]


def test_delete_removes_blob_row(tmp_path):
    """delete() must clear the `graphs` table too, not just `cache` -- otherwise blob
    payloads (the bulk of an archived combo's footprint) never actually leave the
    live SQLite shard files."""
    from nvision.cache.sqlite import ShardedSqliteCache

    cache = ShardedSqliteCache(tmp_path / "base.db")
    cache.blob_set("blob:aaa:0:scan", b"raw-bytes")
    assert cache.blob_get("blob:aaa:0:scan") == b"raw-bytes"

    cache.delete("blob:aaa:0:scan")

    assert cache.blob_get("blob:aaa:0:scan") is None
    assert "blob:aaa:0:scan" not in cache


def test_delete_many_removes_blob_rows(tmp_path):
    from nvision.cache.sqlite import ShardedSqliteCache

    cache = ShardedSqliteCache(tmp_path / "base.db")
    cache.set("combo:aaa", {"config": {"kind": "locator_combination_pointer"}})
    cache.blob_set("blob:aaa:0:scan", b"raw-bytes-0")
    cache.blob_set("blob:aaa:1:scan", b"raw-bytes-1")

    cache.delete_many(["combo:aaa", "blob:aaa:0:scan", "blob:aaa:1:scan"])

    assert cache.blob_batch_get(["blob:aaa:0:scan", "blob:aaa:1:scan"]) == {}
    assert list(cache) == []
