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
