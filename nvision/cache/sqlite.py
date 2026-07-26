from __future__ import annotations

import json
import random
import sqlite3
import threading
import time
from contextlib import suppress
from pathlib import Path


def _retry_on_locked(fn, *, max_attempts: int = 10, base_delay: float = 0.2, max_delay: float = 5.0):
    """Retry ``fn`` on transient 'database is locked' SQLite errors with backoff.

    Several worker processes can write to the same SQLite shard concurrently
    (one per combination). SQLite's own busy_timeout already retries for the
    connection's ``timeout`` seconds, but under heavy contention on Windows
    that isn't always enough — retry the whole operation on top of it rather
    than losing an entire combination's results.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower() or attempt == max_attempts - 1:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            time.sleep(delay + random.uniform(0, delay * 0.25))


def _init_schema_with_recovery(db_path: Path, schema_sql: str) -> None:
    """Initialize SQLite schema and recover from invalid DB files."""
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path)
        with conn:
            conn.execute(schema_sql)
            conn.commit()
    except sqlite3.DatabaseError as exc:
        if "file is not a database" not in str(exc).lower():
            if conn is not None:
                conn.close()
            raise
        # Corrupted/invalid file at this path; reset it so cache can recover.
        if conn is not None:
            conn.close()
        with suppress(FileNotFoundError):
            db_path.unlink()
        with sqlite3.connect(db_path) as conn2:
            conn2.execute(schema_sql)
            conn2.commit()


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
        _init_schema_with_recovery(
            self.db_path,
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
            """,
        )

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
        conn = self._get_conn()

        def _write():
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            conn.commit()

        _retry_on_locked(_write)

    def delete(self, key: str):
        try:
            conn = self._get_conn()
            _retry_on_locked(lambda: (conn.execute("DELETE FROM cache WHERE key = ?", (key,)), conn.commit()))
        except Exception:
            pass

    def __contains__(self, key: str) -> bool:
        try:
            conn = self._get_conn()
            cur = conn.execute("SELECT 1 FROM cache WHERE key = ?", (key,))
            return cur.fetchone() is not None
        except Exception:
            return False

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
                        key
                        TEXT
                        PRIMARY
                        KEY,
                        shard_id
                        INTEGER
                        NOT
                        NULL
                    ); \
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

    def _ensure_graphs_table(self, conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE IF NOT EXISTS graphs (key TEXT PRIMARY KEY, data BLOB)")
        conn.commit()

    def _get_index_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "index_conn"):
            self._local.index_conn = sqlite3.connect(self._index_path, timeout=30.0, check_same_thread=False)
            self._local.index_conn.execute("PRAGMA journal_mode=WAL;")
            self._local.index_conn.execute("PRAGMA synchronous=NORMAL;")
        return self._local.index_conn

    def _ensure_index(self) -> None:
        _init_schema_with_recovery(self._index_path, self._INDEX_SCHEMA)

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

    def _get_shard_cache(self) -> dict[str, int]:
        """Thread-local in-memory cache: key → shard_id.  Avoids SQLite round-trips."""
        cache = getattr(self._local, "shard_cache", None)
        if cache is None:
            self._local.shard_cache = {}
            cache = self._local.shard_cache
        return cache

    def _index_get_shard_id(self, key: str) -> int | None:
        cache = self._get_shard_cache()
        if key in cache:
            return cache[key]
        try:
            conn = self._get_index_conn()
            cur = conn.execute("SELECT shard_id FROM cache_index WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                shard_id = int(row[0])
                cache[key] = shard_id
                return shard_id
        except Exception:
            pass
        return None

    def _index_set_shard_id(self, key: str, shard_id: int) -> None:
        self._get_shard_cache()[key] = shard_id
        try:
            conn = self._get_index_conn()
            _retry_on_locked(
                lambda: (
                    conn.execute(
                        "INSERT OR REPLACE INTO cache_index (key, shard_id) VALUES (?, ?)",
                        (key, int(shard_id)),
                    ),
                    conn.commit(),
                )
            )
        except Exception:
            pass

    def _index_delete_key(self, key: str) -> None:
        self._get_shard_cache().pop(key, None)
        try:
            conn = self._get_index_conn()
            _retry_on_locked(lambda: (conn.execute("DELETE FROM cache_index WHERE key = ?", (key,)), conn.commit()))
        except Exception:
            pass

    def _choose_write_shard_id(self) -> int:
        # Use thread-local cache to avoid repeated filesystem globs.
        # Re-evaluate only when the cached shard has grown past the cap.
        cached = getattr(self._local, "write_shard_id", None)
        if cached is not None:
            shard_path = self._path_for_shard_id(cached)
            try:
                if not shard_path.exists() or shard_path.stat().st_size < self.max_shard_bytes:
                    return cached
            except Exception:
                return cached
            # Shard is full — fall through to recompute

        shard_id = self._compute_write_shard_id()
        self._local.write_shard_id = shard_id
        return shard_id

    def _compute_write_shard_id(self) -> int:
        """Filesystem glob to find the current best shard for writing (called rarely)."""
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

        if self._legacy_path is not None:
            try:
                if self._legacy_path.stat().st_size < self.max_shard_bytes:
                    return 0
            except Exception:
                pass

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

    def blob_get(self, key: str) -> bytes | None:
        """Read raw bytes from the `graphs` BLOB table — zero text-encoding overhead.

        Shares the same key->shard_id index as the JSON `cache` table (blob keys
        use a distinct prefix, so there's no collision); the shard file just
        holds two tables instead of one.
        """
        try:
            shard_id = self._index_get_shard_id(key)
            if shard_id is None:
                return None
            db_path = self._path_for_shard_id(shard_id)
            conn = self._get_conn_for_path(db_path)
            self._ensure_graphs_table(conn)
            cur = conn.execute("SELECT data FROM graphs WHERE key = ?", (key,))
            row = cur.fetchone()
            return bytes(row[0]) if row else None
        except Exception:
            return None

    def blob_set(self, key: str, data: bytes) -> None:
        shard_id = self._index_get_shard_id(key)
        if shard_id is None:
            shard_id = self._choose_write_shard_id()
            self._get_shard_cache()[key] = shard_id

        db_path = self._path_for_shard_id(shard_id)
        conn = self._get_conn_for_path(db_path)
        self._ensure_graphs_table(conn)

        def _write():
            conn.execute(
                "INSERT OR REPLACE INTO graphs (key, data) VALUES (?, ?)",
                (key, data),
            )
            conn.commit()

        _retry_on_locked(_write)
        self._index_set_shard_id(key, shard_id)

    def blob_batch_get(self, keys: list[str]) -> dict[str, bytes]:
        """Fetch multiple blob keys in one round-trip per shard (mirrors batch_get)."""
        if not keys:
            return {}
        result: dict[str, bytes] = {}
        shard_to_keys: dict[int, list[str]] = {}
        try:
            conn = self._get_index_conn()
            placeholders = ",".join("?" * len(keys))
            cur = conn.execute(
                f"SELECT key, shard_id FROM cache_index WHERE key IN ({placeholders})",
                keys,
            )
            for k, sid in cur.fetchall():
                shard_to_keys.setdefault(int(sid), []).append(k)
        except Exception:
            pass

        for shard_id, shard_keys in shard_to_keys.items():
            try:
                db_path = self._path_for_shard_id(shard_id)
                conn = self._get_conn_for_path(db_path)
                self._ensure_graphs_table(conn)
                placeholders = ",".join("?" * len(shard_keys))
                cur = conn.execute(
                    f"SELECT key, data FROM graphs WHERE key IN ({placeholders})",
                    shard_keys,
                )
                for k, v in cur.fetchall():
                    result[k] = bytes(v)
            except Exception:
                pass
        return result

    def get(self, key: str) -> dict | None:
        try:
            # Fast path: consult index DB.
            shard_id = self._index_get_shard_id(key)
            if shard_id is not None:
                db_path = self._path_for_shard_id(shard_id)
                conn = self._get_conn_for_path(db_path)
                self._ensure_cache_table(conn)
                cur = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
                row = cur.fetchone()
                if row:
                    return json.loads(row[0])
                return None

            # Compatibility path: if legacy exists and key is not indexed yet, look there.
            if self._legacy_path is not None:
                conn = self._get_conn_for_path(self._legacy_path)
                self._ensure_cache_table(conn)
                cur = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
                row = cur.fetchone()
                if row:
                    # Backfill index so future lookups are fast.
                    self._index_set_shard_id(key, 0)
                    return json.loads(row[0])
            return None
        except Exception:
            return None

    def keys_exist_batch(self, keys: list[str]) -> set[str]:
        """Return the subset of ``keys`` that exist in the cache (no value parsing).

        Uses a single index query — O(1) round-trips regardless of ``len(keys)``.
        Falls back to the legacy shard for keys not yet in the index.
        """
        if not keys:
            return set()
        found: set[str] = set()
        try:
            conn = self._get_index_conn()
            placeholders = ",".join("?" * len(keys))
            cur = conn.execute(
                f"SELECT key FROM cache_index WHERE key IN ({placeholders})",
                keys,
            )
            found.update(k for (k,) in cur.fetchall())
        except Exception:
            pass
        # Legacy fallback for keys not yet promoted to the index
        missing = [k for k in keys if k not in found]
        if missing and self._legacy_path is not None:
            try:
                conn = self._get_conn_for_path(self._legacy_path)
                self._ensure_cache_table(conn)
                placeholders = ",".join("?" * len(missing))
                cur = conn.execute(
                    f"SELECT key FROM cache WHERE key IN ({placeholders})",
                    missing,
                )
                found.update(k for (k,) in cur.fetchall())
            except Exception:
                pass
        return found

    def batch_get(self, keys: list[str]) -> dict[str, dict]:
        """Fetch multiple keys in a single round-trip per shard.

        Reduces N×2 sequential queries to 1 index query + 1 data query per shard.
        Returns {key: parsed_value} for all found keys; missing keys are absent.
        """
        if not keys:
            return {}

        result: dict[str, dict] = {}

        # 1. Batch index lookup: get all shard IDs in one query
        shard_to_keys: dict[int, list[str]] = {}
        indexed_keys: set[str] = set()
        try:
            conn = self._get_index_conn()
            placeholders = ",".join("?" * len(keys))
            cur = conn.execute(
                f"SELECT key, shard_id FROM cache_index WHERE key IN ({placeholders})",
                keys,
            )
            for k, sid in cur.fetchall():
                shard_to_keys.setdefault(int(sid), []).append(k)
                indexed_keys.add(k)
        except Exception:
            pass

        # 2. One data query per shard
        for shard_id, shard_keys in shard_to_keys.items():
            try:
                db_path = self._path_for_shard_id(shard_id)
                conn = self._get_conn_for_path(db_path)
                self._ensure_cache_table(conn)
                placeholders = ",".join("?" * len(shard_keys))
                cur = conn.execute(
                    f"SELECT key, value FROM cache WHERE key IN ({placeholders})",
                    shard_keys,
                )
                for k, v in cur.fetchall():
                    try:
                        result[k] = json.loads(v)
                    except Exception:
                        pass
            except Exception:
                pass

        # 3. Legacy fallback for keys not yet in the index
        missing = [k for k in keys if k not in indexed_keys]
        if missing and self._legacy_path is not None:
            try:
                conn = self._get_conn_for_path(self._legacy_path)
                self._ensure_cache_table(conn)
                placeholders = ",".join("?" * len(missing))
                cur = conn.execute(
                    f"SELECT key, value FROM cache WHERE key IN ({placeholders})",
                    missing,
                )
                for k, v in cur.fetchall():
                    try:
                        result[k] = json.loads(v)
                        self._index_set_shard_id(k, 0)  # backfill index
                    except Exception:
                        pass
            except Exception:
                pass

        return result

    def batch_set(self, items: dict[str, dict]) -> None:
        """Write multiple keys atomically — one executemany per shard, one index commit.

        Reduces N×4 SQLite transactions to 2–3 regardless of N.
        All items land in the same shard unless the current shard fills mid-batch.
        """
        if not items:
            return

        shard_cache = self._get_shard_cache()

        # Batch index lookup for keys not yet in memory cache
        unknown = [k for k in items if k not in shard_cache]
        if unknown:
            try:
                conn = self._get_index_conn()
                placeholders = ",".join("?" * len(unknown))
                cur = conn.execute(
                    f"SELECT key, shard_id FROM cache_index WHERE key IN ({placeholders})",
                    unknown,
                )
                for k, sid in cur.fetchall():
                    shard_cache[k] = int(sid)
            except Exception:
                pass

        # Assign shard for each key; new keys go to the current write shard
        by_shard: dict[int, list[tuple[str, str]]] = {}
        new_index: list[tuple[str, int]] = []
        for key, value in items.items():
            shard_id = shard_cache.get(key)
            if shard_id is None:
                shard_id = self._choose_write_shard_id()
                shard_cache[key] = shard_id
                new_index.append((key, shard_id))
            by_shard.setdefault(shard_id, []).append((key, json.dumps(value)))

        # One executemany + commit per shard
        for shard_id, pairs in by_shard.items():
            db_path = self._path_for_shard_id(shard_id)
            conn = self._get_conn_for_path(db_path)
            self._ensure_cache_table(conn)

            def _write(conn=conn, pairs=pairs):
                conn.executemany("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", pairs)
                conn.commit()

            _retry_on_locked(_write)

        # One index commit for all new key→shard mappings
        if new_index:
            try:
                conn = self._get_index_conn()
                conn.executemany(
                    "INSERT OR REPLACE INTO cache_index (key, shard_id) VALUES (?, ?)",
                    new_index,
                )
                conn.commit()
            except Exception:
                pass

    def set(self, key: str, value: dict):
        shard_id = self._index_get_shard_id(key)
        if shard_id is None:
            shard_id = self._choose_write_shard_id()
            self._get_shard_cache()[key] = shard_id

        db_path = self._path_for_shard_id(shard_id)
        conn = self._get_conn_for_path(db_path)
        self._ensure_cache_table(conn)

        def _write():
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            conn.commit()

        _retry_on_locked(_write)
        self._index_set_shard_id(key, shard_id)

    def delete(self, key: str):
        self._get_shard_cache().pop(key, None)
        try:
            shard_id = self._index_get_shard_id(key)
            if shard_id is not None:
                db_path = self._path_for_shard_id(shard_id)
                conn = self._get_conn_for_path(db_path)
                self._ensure_cache_table(conn)
                _retry_on_locked(lambda: (conn.execute("DELETE FROM cache WHERE key = ?", (key,)), conn.commit()))
                self._index_delete_key(key)
                return

            # If unknown, try deleting from legacy DB (and ignore errors).
            if self._legacy_path is not None:
                conn = self._get_conn_for_path(self._legacy_path)
                self._ensure_cache_table(conn)
                _retry_on_locked(lambda: (conn.execute("DELETE FROM cache WHERE key = ?", (key,)), conn.commit()))
        except Exception:
            pass

    def __contains__(self, key: str) -> bool:
        try:
            shard_id = self._index_get_shard_id(key)
            if shard_id is not None:
                return True
            if self._legacy_path is not None:
                conn = self._get_conn_for_path(self._legacy_path)
                self._ensure_cache_table(conn)
                cur = conn.execute("SELECT 1 FROM cache WHERE key = ?", (key,))
                return cur.fetchone() is not None
            return False
        except Exception:
            return False

    def __iter__(self):
        # Primary source of truth is the index DB.
        yielded: set[str] = set()
        try:
            conn = self._get_index_conn()
            cur = conn.execute("SELECT key FROM cache_index")
            for (k,) in cur:
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
                for (k,) in cur:
                    if isinstance(k, str) and k not in yielded:
                        yield k
            except Exception:
                pass
