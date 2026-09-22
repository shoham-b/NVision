"""MySqlCache tests against an in-memory fake connection (no real MySQL server needed).

The fake only implements the small, fixed set of SQL shapes MySqlCache actually issues
(see nvision/cache/mysql.py) -- enough to exercise the delete/blob-cleanup logic without
a live Cloud SQL instance.
"""

import re

import pytest

from nvision.cache.mysql import MySqlCache


class _FakeCursor:
    def __init__(self, tables: dict[str, dict[str, object]]):
        self._tables = tables
        self._result: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql: str, params=()):
        params = params or ()
        self._result = []

        if sql.startswith("CREATE TABLE"):
            table = re.match(r"CREATE TABLE IF NOT EXISTS `(\w+)`", sql).group(1)
            self._tables.setdefault(table, {})
            return

        if sql.startswith("INSERT INTO"):
            table = re.match(r"INSERT INTO `(\w+)`", sql).group(1)
            key, value = params
            self._tables.setdefault(table, {})[key] = value
            return

        if sql.startswith("DELETE FROM"):
            table = re.match(r"DELETE FROM `(\w+)`", sql).group(1)
            store = self._tables.setdefault(table, {})
            if "IN (" in sql:
                for key in params:
                    store.pop(key, None)
            else:
                (key,) = params
                store.pop(key, None)
            return

        if sql.startswith("SELECT value FROM") or sql.startswith("SELECT data FROM"):
            table = re.search(r"FROM `(\w+)`", sql).group(1)
            (key,) = params
            store = self._tables.get(table, {})
            if key in store:
                self._result = [(store[key],)]
            return

        if sql.startswith("SELECT `key`, value FROM") or sql.startswith("SELECT `key`, data FROM"):
            table = re.search(r"FROM `(\w+)`", sql).group(1)
            store = self._tables.get(table, {})
            self._result = [(k, v) for k, v in store.items() if k in params]
            return

        if sql.startswith("SELECT 1 FROM"):
            table = re.search(r"FROM `(\w+)`", sql).group(1)
            (key,) = params
            store = self._tables.get(table, {})
            self._result = [(1,)] if key in store else []
            return

        if sql.startswith("SELECT `key` FROM") and "WHERE" not in sql:
            table = re.search(r"FROM `(\w+)`", sql).group(1)
            store = self._tables.get(table, {})
            self._result = [(k,) for k in store]
            return

        raise NotImplementedError(f"Fake cursor can't handle: {sql}")

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)


class _FakeConnection:
    def __init__(self):
        self.tables: dict[str, dict[str, object]] = {}

    def cursor(self):
        return _FakeCursor(self.tables)

    def commit(self):
        pass

    def close(self):
        pass


@pytest.fixture
def mysql_cache(monkeypatch):
    """A MySqlCache wired to an in-memory fake connection instead of a real pymysql one."""
    cache = object.__new__(MySqlCache)
    cache.table_prefix = "test"
    cache.shard_suffix = "0"  # non-None -- skips information_schema shard discovery
    cache._cache_table = "test_cache_shard0"
    cache._graphs_table = "test_graphs_shard0"
    cache._discovered_cache_tables = None
    cache._discovered_graphs_tables = None

    fake_conn = _FakeConnection()
    monkeypatch.setattr(cache, "_get_conn", lambda: fake_conn)
    return cache


def test_delete_removes_blob_row(mysql_cache):
    """delete() must clear the graphs table too, not just the cache table -- otherwise
    blob payloads (the bulk of an archived combo's footprint) never actually leave the
    live MySQL tables. Mirrors the equivalent ShardedSqliteCache regression test."""
    mysql_cache.blob_set("blob:aaa:0:scan", b"raw-bytes")
    assert mysql_cache.blob_get("blob:aaa:0:scan") == b"raw-bytes"

    mysql_cache.delete("blob:aaa:0:scan")

    assert mysql_cache.blob_get("blob:aaa:0:scan") is None
    assert "blob:aaa:0:scan" not in mysql_cache


def test_delete_many_removes_blob_rows(mysql_cache):
    mysql_cache.set("combo:aaa", {"config": {"kind": "locator_combination_pointer"}})
    mysql_cache.blob_set("blob:aaa:0:scan", b"raw-bytes-0")
    mysql_cache.blob_set("blob:aaa:1:scan", b"raw-bytes-1")

    mysql_cache.delete_many(["combo:aaa", "blob:aaa:0:scan", "blob:aaa:1:scan"])

    assert mysql_cache.blob_batch_get(["blob:aaa:0:scan", "blob:aaa:1:scan"]) == {}
    assert list(mysql_cache) == []


def test_delete_leaves_other_keys_intact(mysql_cache):
    mysql_cache.set("combo:aaa", {"a": 1})
    mysql_cache.set("combo:bbb", {"a": 2})
    mysql_cache.blob_set("blob:aaa:0:scan", b"raw-bytes")

    mysql_cache.delete("combo:aaa")

    assert mysql_cache.get("combo:aaa") is None
    assert mysql_cache.get("combo:bbb") == {"a": 2}
    assert mysql_cache.blob_get("blob:aaa:0:scan") == b"raw-bytes"
