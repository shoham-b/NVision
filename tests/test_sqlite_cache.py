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
