from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import suppress
from pathlib import Path


class SqliteCache:
    """A thread-safe key-value cache backed by SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA synchronous=NORMAL;")
        return self._local.conn

    def close(self):
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn

    def _ensure_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache
                (
                    key
                    TEXT
                    PRIMARY
                    KEY,
                    value
                    TEXT
                )
                """
            )
            conn.commit()

    def get(self, key: str) -> dict | None:
        try:
            conn = self._get_conn()
            cur = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
        except Exception:
            pass
        return None

    def set(self, key: str, value: dict):
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            conn.commit()
        except Exception:
            pass

    def delete(self, key: str):
        try:
            conn = self._get_conn()
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
        except Exception:
            pass

    def __iter__(self):
        try:
            conn = self._get_conn()
            cur = conn.execute("SELECT key FROM cache")
            for row in cur:
                yield row[0]
        except Exception:
            pass


class ShardedSqliteCache:
    """A thread-safe key-value cache backed by multiple SQLite shard files.

    Motivation: keep each SQLite file under a hard size cap (e.g. 2GB), which is
    important on some platforms/filesystems and avoids SQLite file-size limits.
    """

    _INDEX_SCHEMA = """
        CREATE TABLE IF NOT EXISTS cache_index
        (
            key TEXT PRIMARY KEY,
            shard_id INTEGER NOT NULL
        );
    """

    def __init__(
        self,
        base_db_path: Path,
        *,
        max_shard_bytes: int = 2 * 1024 * 1024 * 1024 - 16 * 1024 * 1024,
        shard_digits: int = 4,
    ):
        self.base_db_path = base_db_path
        self.max_shard_bytes = max_shard_bytes
        self.shard_digits = shard_digits

        self.base_db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._index_path = self._make_index_path()
        self._ensure_index()

        # If a legacy single-file DB exists at base_db_path, treat it as shard 0
        # but do not write new entries there once it approaches the cap.
        self._legacy_path = self.base_db_path if self.base_db_path.exists() else None

    def _make_index_path(self) -> Path:
        # Keep the index adjacent to the base DB, but separate to avoid growing large.
        return self.base_db_path.with_name(f"{self.base_db_path.stem}_index{self.base_db_path.suffix}")

    def _shard_path(self, shard_id: int) -> Path:
        # Example: nv_center_shard0001.db
        suffix = self.base_db_path.suffix
        stem = self.base_db_path.stem
        return self.base_db_path.with_name(f"{stem}_shard{shard_id:0{self.shard_digits}d}{suffix}")

    def _get_conn_for_path(self, path: Path) -> sqlite3.Connection:
        conns = getattr(self._local, "conns", None)
        if conns is None:
            conns = {}
            self._local.conns = conns

        key = str(path)
        if key not in conns:
            conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conns[key] = conn
        return conns[key]

    def _ensure_cache_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache
            (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        conn.commit()

    def _get_index_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "index_conn"):
            self._local.index_conn = sqlite3.connect(self._index_path, timeout=30.0, check_same_thread=False)
            self._local.index_conn.execute("PRAGMA journal_mode=WAL;")
            self._local.index_conn.execute("PRAGMA synchronous=NORMAL;")
        return self._local.index_conn

    def _ensure_index(self) -> None:
        with sqlite3.connect(self._index_path) as conn:
            conn.execute(self._INDEX_SCHEMA)
            conn.commit()

    def close(self):
        with suppress(Exception):
            if hasattr(self._local, "index_conn"):
                self._local.index_conn.close()
                del self._local.index_conn

        conns = getattr(self._local, "conns", None)
        if conns:
            for conn in list(conns.values()):
                with suppress(Exception):
                    conn.close()
            self._local.conns = {}

        with suppress(Exception):
            if hasattr(self._local, "conn"):
                self._local.conn.close()
                del self._local.conn

    def _index_get_shard_id(self, key: str) -> int | None:
        try:
            conn = self._get_index_conn()
            cur = conn.execute("SELECT shard_id FROM cache_index WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                return int(row[0])
        except Exception:
            pass
        return None

    def _index_set_shard_id(self, key: str, shard_id: int) -> None:
        try:
            conn = self._get_index_conn()
            conn.execute(
                "INSERT OR REPLACE INTO cache_index (key, shard_id) VALUES (?, ?)",
                (key, int(shard_id)),
            )
            conn.commit()
        except Exception:
            pass

    def _index_delete_key(self, key: str) -> None:
        try:
            conn = self._get_index_conn()
            conn.execute("DELETE FROM cache_index WHERE key = ?", (key,))
            conn.commit()
        except Exception:
            pass

    def _choose_write_shard_id(self) -> int:
        # Prefer the newest shard file under the cap; otherwise create the next.
        try:
            shards = sorted(self.base_db_path.parent.glob(f"{self.base_db_path.stem}_shard*{self.base_db_path.suffix}"))
        except Exception:
            shards = []

        if shards:
            last = shards[-1]
            try:
                if last.stat().st_size < self.max_shard_bytes:
                    return int(last.stem.split("shard")[-1])
            except Exception:
                pass

        # No shards yet, or last shard too large. Decide whether legacy can be used.
        if self._legacy_path is not None:
            try:
                if self._legacy_path.stat().st_size < self.max_shard_bytes:
                    return 0  # legacy-as-shard0
            except Exception:
                pass

        # Create a fresh shard file.
        next_id = 1
        if shards:
            try:
                next_id = int(shards[-1].stem.split("shard")[-1]) + 1
            except Exception:
                next_id = len(shards) + 1
        return next_id

    def _path_for_shard_id(self, shard_id: int) -> Path:
        # shard_id==0 can be the legacy base path (if it exists), otherwise a shard file.
        if shard_id == 0 and self._legacy_path is not None:
            return self._legacy_path
        return self._shard_path(shard_id)

    def get(self, key: str) -> dict | None:
        # Fast path: consult index DB.
        shard_id = self._index_get_shard_id(key)
        if shard_id is not None:
            try:
                db_path = self._path_for_shard_id(shard_id)
                conn = self._get_conn_for_path(db_path)
                self._ensure_cache_table(conn)
                cur = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
                row = cur.fetchone()
                if row:
                    return json.loads(row[0])
                return None
            except Exception:
                return None

        # Compatibility path: if legacy exists and key is not indexed yet, look there.
        if self._legacy_path is not None:
            try:
                conn = self._get_conn_for_path(self._legacy_path)
                self._ensure_cache_table(conn)
                cur = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
                row = cur.fetchone()
                if row:
                    # Backfill index so future lookups are fast.
                    self._index_set_shard_id(key, 0)
                    return json.loads(row[0])
            except Exception:
                pass
        return None

    def set(self, key: str, value: dict):
        try:
            # If already assigned, keep writing to that shard to avoid duplication.
            shard_id = self._index_get_shard_id(key)
            if shard_id is None:
                shard_id = self._choose_write_shard_id()

            db_path = self._path_for_shard_id(shard_id)
            conn = self._get_conn_for_path(db_path)
            self._ensure_cache_table(conn)
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            conn.commit()
            self._index_set_shard_id(key, shard_id)
        except Exception:
            pass

    def delete(self, key: str):
        try:
            shard_id = self._index_get_shard_id(key)
            if shard_id is not None:
                db_path = self._path_for_shard_id(shard_id)
                conn = self._get_conn_for_path(db_path)
                self._ensure_cache_table(conn)
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                self._index_delete_key(key)
                return

            # If unknown, try deleting from legacy DB (and ignore errors).
            if self._legacy_path is not None:
                conn = self._get_conn_for_path(self._legacy_path)
                self._ensure_cache_table(conn)
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
        except Exception:
            pass

    def __iter__(self):
        # Primary source of truth is the index DB.
        yielded: set[str] = set()
        try:
            conn = self._get_index_conn()
            cur = conn.execute("SELECT key FROM cache_index")
            for (k,) in cur.fetchall():
                if isinstance(k, str) and k not in yielded:
                    yielded.add(k)
                    yield k
        except Exception:
            pass

        # Also include any legacy keys not yet indexed (best-effort).
        if self._legacy_path is not None:
            try:
                conn = self._get_conn_for_path(self._legacy_path)
                self._ensure_cache_table(conn)
                cur = conn.execute("SELECT key FROM cache")
                for (k,) in cur.fetchall():
                    if isinstance(k, str) and k not in yielded:
                        yield k
            except Exception:
                pass
