"""
SQLite state management.

Stores: channels, last_read_at timestamp, raw message cache, summary cache.
"""

from __future__ import annotations

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "dashboard.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS channels (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT UNIQUE NOT NULL,
                title      TEXT,
                added_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS raw_messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                channel      TEXT NOT NULL,
                message_id   INTEGER NOT NULL,
                sent_at      TEXT NOT NULL,
                text         TEXT NOT NULL,
                UNIQUE(channel, message_id)
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at   TEXT NOT NULL,
                from_ts      TEXT NOT NULL,
                to_ts        TEXT NOT NULL,
                data         TEXT NOT NULL
            );
        """)


# --- State ---

def get_state(key: str) -> str | None:
    with _connect() as conn:
        row = conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_state(key: str, value: str) -> None:
    with _connect() as conn:
        conn.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, value))


def get_last_read_at() -> datetime | None:
    with _connect() as conn:
        row = conn.execute("SELECT value FROM state WHERE key = 'last_read_at'").fetchone()
    if row is None:
        return None
    return datetime.fromisoformat(row["value"])


def set_last_read_at(ts: datetime) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES ('last_read_at', ?)",
            (ts.isoformat(),),
        )


# --- Channels ---

def add_channel(username: str, title: str | None = None) -> None:
    username = username.lstrip("@")
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO channels (username, title, added_at) VALUES (?, ?, ?)",
            (username, title, datetime.now(timezone.utc).isoformat()),
        )


def remove_channel(username: str) -> bool:
    username = username.lstrip("@")
    with _connect() as conn:
        cur = conn.execute("DELETE FROM channels WHERE username = ?", (username,))
        return cur.rowcount > 0


def list_channels() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT username, title, added_at FROM channels ORDER BY added_at").fetchall()
    return [dict(r) for r in rows]


def update_channel_title(username: str, title: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE channels SET title = ? WHERE username = ?", (title, username))


# --- Raw messages ---

def store_messages(channel: str, messages: list[dict]) -> int:
    """Insert messages, ignoring duplicates. Returns count of new rows inserted."""
    if not messages:
        return 0
    with _connect() as conn:
        rows = [(channel, m["id"], m["sent_at"], m["text"]) for m in messages]
        conn.executemany(
            """INSERT OR IGNORE INTO raw_messages (channel, message_id, sent_at, text)
               VALUES (?, ?, ?, ?)""",
            rows,
        )
        return conn.total_changes


def get_messages_since(ts: datetime | None) -> list[dict]:
    """Return all cached messages after ts, ordered by sent_at."""
    with _connect() as conn:
        if ts is None:
            rows = conn.execute(
                "SELECT channel, message_id, sent_at, text FROM raw_messages ORDER BY sent_at"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT channel, message_id, sent_at, text FROM raw_messages WHERE sent_at > ? ORDER BY sent_at",
                (ts.isoformat(),),
            ).fetchall()
    return [dict(r) for r in rows]


# --- Summaries ---

def store_summary(from_ts: datetime | None, to_ts: datetime, data: dict) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO summaries (created_at, from_ts, to_ts, data) VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                from_ts.isoformat() if from_ts else "",
                to_ts.isoformat(),
                json.dumps(data),
            ),
        )
        return cur.lastrowid


def get_latest_summary() -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM summaries ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["data"] = json.loads(result["data"])
    return result
