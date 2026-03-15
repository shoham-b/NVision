from __future__ import annotations

import json
import sqlite3
import threading
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
