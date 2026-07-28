"""MySQL-backed persistence for the cache layer (shared Cloud SQL DB layer only).

Duck-type compatible with ``ShardedSqliteCache`` (nvision/cache/sqlite.py) so
``CategoryDataStore`` can swap backends via ``NVISION_CACHE_BACKEND=mysql``.

Table naming encodes the sharding scheme directly, instead of routing individual
keys to shards the way ``ShardedSqliteCache`` does:

- Write target is always exactly one table: ``{prefix}_cache`` / ``{prefix}_graphs``
  when unsharded (``shard_suffix=None``), or ``{prefix}_cache_shard{N}`` /
  ``{prefix}_graphs_shard{N}`` for shard ``N``. Combinations are statically
  partitioned one-per-shard by the caller (see ``nvision/cli/run.py``'s shard
  slicing), so a given shard's table is never written by more than one process
  tree -- there is no cross-pod write contention to reason about.
- Reads fan out: an unsharded instance (``shard_suffix=None``) also transparently
  sees every ``{prefix}_cache_shard*`` table that exists, merging them with its own
  base table. Because keys never collide across shards (each combination's keys are
  only ever written by the one shard/process that owns it), the merge is an
  unambiguous union -- there's no last-writer-wins conflict.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from contextlib import suppress

import pymysql
import pymysql.cursors

_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")

# MySQL error codes treated as transient and worth retrying with backoff:
# 1213 = deadlock found when trying to get lock, 1205 = lock wait timeout exceeded.
_TRANSIENT_ERROR_CODES = {1213, 1205}


def _validate_name(name: str, what: str) -> str:
    if not _NAME_RE.match(name):
        raise ValueError(f"Invalid {what}: {name!r} (must match [a-zA-Z0-9_]+)")
    return name


def _retry_on_mysql_transient(fn, *, max_attempts: int = 10, base_delay: float = 0.2, max_delay: float = 5.0):
    """Retry ``fn`` on transient MySQL deadlock/lock-wait errors with backoff.

    Since each shard writes only to its own table, this is not needed for
    cross-pod correctness -- it just smooths over ordinary transient contention
    from a single pod's own internal process-pool concurrency, the same role
    ``_retry_on_locked`` plays for SQLite.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except pymysql.err.OperationalError as exc:
            code = exc.args[0] if exc.args else None
            if code not in _TRANSIENT_ERROR_CODES or attempt == max_attempts - 1:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            time.sleep(delay + random.uniform(0, delay * 0.25))


class MySqlCache:
    """A key-value cache backed by a shared MySQL (Cloud SQL) database."""

    def __init__(
        self,
        table_prefix: str,
        *,
        shard_suffix: str | None = None,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        self.table_prefix = _validate_name(table_prefix, "table_prefix")
        self.shard_suffix = _validate_name(str(shard_suffix), "shard_suffix") if shard_suffix is not None else None

        self.host = host or os.getenv("NVISION_MYSQL_HOST")
        self.port = int(port or os.getenv("NVISION_MYSQL_PORT", "3306"))
        self.user = user or os.getenv("NVISION_MYSQL_USER")
        self.password = password if password is not None else os.getenv("NVISION_MYSQL_PASSWORD", "")
        self.database = database or os.getenv("NVISION_MYSQL_DATABASE")
        self._ssl_disabled = os.getenv("NVISION_MYSQL_SSL_DISABLED", "False").lower() in ("true", "1", "yes")

        if not self.host or not self.user or not self.database:
            raise RuntimeError(
                "MySQL cache backend requires NVISION_MYSQL_HOST, NVISION_MYSQL_USER, and "
                "NVISION_MYSQL_DATABASE to be set (or passed explicitly)."
            )

        self._local = threading.local()

        suffix = f"_shard{self.shard_suffix}" if self.shard_suffix is not None else ""
        self._cache_table = f"{self.table_prefix}_cache{suffix}"
        self._graphs_table = f"{self.table_prefix}_graphs{suffix}"
        self._ensure_table(self._cache_table, is_blob=False)
        self._ensure_table(self._graphs_table, is_blob=True)

        self._discovered_cache_tables: list[str] | None = None
        self._discovered_graphs_tables: list[str] | None = None

    # -- connection ---------------------------------------------------------

    def _get_conn(self) -> pymysql.connections.Connection:
        if not hasattr(self._local, "conn"):
            connect_kwargs = dict(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                autocommit=False,
                cursorclass=pymysql.cursors.Cursor,
            )
            if self._ssl_disabled:
                # Without this, PyMySQL still attempts opportunistic ("PREFERRED") TLS
                # by default even with no `ssl=` kwarg -- explicitly disable it for local
                # dev DBs that aren't set up for client TLS.
                connect_kwargs["ssl_disabled"] = True
            else:
                # A non-empty `ssl` dict makes PyMySQL *require* TLS (vs. merely preferring
                # it and silently falling back to plaintext if the server allows it).
                # `check_hostname: False` with no `ca`/`capath` gives CERT_NONE verification
                # (encrypted, not cert-pinned) -- avoids bundling Cloud SQL's CA bundle into
                # the image. See pymysql.connections.Connection._create_ssl_ctx.
                connect_kwargs["ssl"] = {"check_hostname": False}
            self._local.conn = pymysql.connect(**connect_kwargs)
        return self._local.conn

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            with suppress(Exception):
                self._local.conn.close()
            del self._local.conn

    # -- schema ---------------------------------------------------------------

    def _ensure_table(self, table: str, *, is_blob: bool) -> None:
        conn = self._get_conn()
        column = "data LONGBLOB" if is_blob else "value LONGTEXT"
        with conn.cursor() as cur:
            cur.execute(f"CREATE TABLE IF NOT EXISTS `{table}` (`key` VARCHAR(191) PRIMARY KEY, {column})")
        conn.commit()

    # -- table resolution -------------------------------------------------------

    def _discover_tables(self, base_table: str, like_pattern: str) -> list[str]:
        """Base table (always present) plus every ``{prefix}_..._shard*`` table found."""
        conn = self._get_conn()
        tables = [base_table]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name LIKE %s AND table_name != %s",
                (self.database, like_pattern, base_table),
            )
            tables.extend(row[0] for row in cur.fetchall())
        return tables

    def _read_cache_tables(self) -> list[str]:
        if self.shard_suffix is not None:
            return [self._cache_table]
        if self._discovered_cache_tables is None:
            self._discovered_cache_tables = self._discover_tables(
                self._cache_table, f"{self.table_prefix}_cache_shard%"
            )
        return self._discovered_cache_tables

    def _read_graphs_tables(self) -> list[str]:
        if self.shard_suffix is not None:
            return [self._graphs_table]
        if self._discovered_graphs_tables is None:
            self._discovered_graphs_tables = self._discover_tables(
                self._graphs_table, f"{self.table_prefix}_graphs_shard%"
            )
        return self._discovered_graphs_tables

    # -- JSON cache -------------------------------------------------------------

    def get(self, key: str) -> dict | None:
        conn = self._get_conn()
        for table in self._read_cache_tables():
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT value FROM `{table}` WHERE `key` = %s", (key,))
                    row = cur.fetchone()
                if row:
                    return json.loads(row[0])
            except Exception:
                continue
        return None

    def set(self, key: str, value: dict) -> None:
        conn = self._get_conn()

        def _write():
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO `{self._cache_table}` (`key`, value) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE value = VALUES(value)",
                    (key, json.dumps(value)),
                )
            conn.commit()

        _retry_on_mysql_transient(_write)

    def delete(self, key: str) -> None:
        try:
            conn = self._get_conn()
            _retry_on_mysql_transient(
                lambda: self._exec_commit(conn, f"DELETE FROM `{self._cache_table}` WHERE `key` = %s", (key,))
            )
        except Exception:
            pass

    def __contains__(self, key: str) -> bool:
        conn = self._get_conn()
        for table in self._read_cache_tables():
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT 1 FROM `{table}` WHERE `key` = %s", (key,))
                    if cur.fetchone() is not None:
                        return True
            except Exception:
                continue
        return False

    def __iter__(self):
        conn = self._get_conn()
        yielded: set[str] = set()
        for table in self._read_cache_tables():
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT `key` FROM `{table}`")
                    for (k,) in cur.fetchall():
                        if k not in yielded:
                            yielded.add(k)
                            yield k
            except Exception:
                continue

    def keys_exist_batch(self, keys: list[str]) -> set[str]:
        if not keys:
            return set()
        conn = self._get_conn()
        found: set[str] = set()
        placeholders = ",".join(["%s"] * len(keys))
        for table in self._read_cache_tables():
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT `key` FROM `{table}` WHERE `key` IN ({placeholders})", keys)
                    found.update(row[0] for row in cur.fetchall())
            except Exception:
                continue
        return found

    def batch_get(self, keys: list[str]) -> dict[str, dict]:
        if not keys:
            return {}
        conn = self._get_conn()
        result: dict[str, dict] = {}
        placeholders = ",".join(["%s"] * len(keys))
        for table in self._read_cache_tables():
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT `key`, value FROM `{table}` WHERE `key` IN ({placeholders})", keys)
                    for k, v in cur.fetchall():
                        if k not in result:
                            with suppress(Exception):
                                result[k] = json.loads(v)
            except Exception:
                continue
        return result

    def batch_set(self, items: dict[str, dict]) -> None:
        if not items:
            return
        conn = self._get_conn()
        pairs = [(k, json.dumps(v)) for k, v in items.items()]

        def _write():
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO `{self._cache_table}` (`key`, value) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE value = VALUES(value)",
                    pairs,
                )
            conn.commit()

        _retry_on_mysql_transient(_write)

    # -- blob (graphs) store ------------------------------------------------------

    def blob_get(self, key: str) -> bytes | None:
        conn = self._get_conn()
        for table in self._read_graphs_tables():
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT data FROM `{table}` WHERE `key` = %s", (key,))
                    row = cur.fetchone()
                if row:
                    return bytes(row[0])
            except Exception:
                continue
        return None

    def blob_set(self, key: str, data: bytes) -> None:
        conn = self._get_conn()

        def _write():
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO `{self._graphs_table}` (`key`, data) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE data = VALUES(data)",
                    (key, data),
                )
            conn.commit()

        _retry_on_mysql_transient(_write)

    def blob_batch_get(self, keys: list[str]) -> dict[str, bytes]:
        if not keys:
            return {}
        conn = self._get_conn()
        result: dict[str, bytes] = {}
        placeholders = ",".join(["%s"] * len(keys))
        for table in self._read_graphs_tables():
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT `key`, data FROM `{table}` WHERE `key` IN ({placeholders})", keys)
                    for k, v in cur.fetchall():
                        if k not in result:
                            result[k] = bytes(v)
            except Exception:
                continue
        return result

    # -- helpers ------------------------------------------------------------------

    @staticmethod
    def _exec_commit(conn, sql: str, params: tuple) -> None:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
