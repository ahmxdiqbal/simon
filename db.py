"""
State storage backed by Turso (libSQL) over HTTP.

Stores: channels, fetch/read cursors, raw message cache, weekly reports,
and the rolling "since last read" report.

Connection target comes from env:
  TURSO_DATABASE_URL  - libsql:// URL (remote) or a local file path (dev/tests)
  TURSO_AUTH_TOKEN    - auth token for remote; empty for local files
Falls back to a local dashboard.db file when TURSO_DATABASE_URL is unset.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import libsql

DEFAULT_LOCAL = str(Path(__file__).parent / "dashboard.db")


@contextmanager
def _conn() -> Iterator["libsql.Connection"]:
    conn = libsql.connect(
        database=os.environ.get("TURSO_DATABASE_URL", DEFAULT_LOCAL),
        auth_token=os.environ.get("TURSO_AUTH_TOKEN", ""),
    )
    try:
        yield conn
    finally:
        conn.close()


def _rows(cur) -> list[dict]:
    """Map a cursor's result set to a list of dicts using its column names."""
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _one(cur) -> dict | None:
    rows = _rows(cur)
    return rows[0] if rows else None


def init_db() -> None:
    with _conn() as conn:
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

            CREATE TABLE IF NOT EXISTS weekly_reports (
                week_start  TEXT PRIMARY KEY,
                data        TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                read_at     TEXT
            );

            CREATE TABLE IF NOT EXISTS unread_report (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                data        TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
        """)
        conn.commit()


# --- State ---

def get_state(key: str) -> str | None:
    with _conn() as conn:
        row = _one(conn.execute("SELECT value FROM state WHERE key = ?", (key,)))
    return row["value"] if row else None


def set_state(key: str, value: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()


def _get_ts(key: str) -> datetime | None:
    value = get_state(key)
    return datetime.fromisoformat(value) if value else None


def get_last_read_at() -> datetime | None:
    return _get_ts("last_read_at")


def set_last_read_at(ts: datetime) -> None:
    set_state("last_read_at", ts.isoformat())


def get_last_refresh_at() -> datetime | None:
    return _get_ts("last_refresh_at")


def set_last_refresh_at(ts: datetime) -> None:
    set_state("last_refresh_at", ts.isoformat())


# --- Channels ---

def add_channel(username: str, title: str | None = None) -> None:
    username = username.lstrip("@")
    with _conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO channels (username, title, added_at) VALUES (?, ?, ?)",
            (username, title, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def remove_channel(username: str) -> bool:
    username = username.lstrip("@")
    with _conn() as conn:
        cur = conn.execute("DELETE FROM channels WHERE username = ?", (username,))
        conn.commit()
        return cur.rowcount > 0


def list_channels() -> list[dict]:
    with _conn() as conn:
        return _rows(conn.execute(
            "SELECT username, title, added_at FROM channels ORDER BY added_at"
        ))


def update_channel_title(username: str, title: str) -> None:
    with _conn() as conn:
        conn.execute("UPDATE channels SET title = ? WHERE username = ?", (title, username))
        conn.commit()


# --- Raw messages ---

def store_messages(channel: str, messages: list[dict]) -> int:
    """Insert messages, ignoring duplicates. Returns rows newly inserted."""
    if not messages:
        return 0
    rows = [(channel, m["id"], m["sent_at"], m["text"]) for m in messages]
    with _conn() as conn:
        cur = conn.executemany(
            "INSERT OR IGNORE INTO raw_messages (channel, message_id, sent_at, text) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        return cur.rowcount


def get_messages_since(ts: datetime | None) -> list[dict]:
    """Cached messages after ts (or all if ts is None), ordered by sent_at."""
    with _conn() as conn:
        if ts is None:
            cur = conn.execute(
                "SELECT channel, message_id, sent_at, text FROM raw_messages ORDER BY sent_at"
            )
        else:
            cur = conn.execute(
                "SELECT channel, message_id, sent_at, text FROM raw_messages "
                "WHERE sent_at > ? ORDER BY sent_at",
                (ts.isoformat(),),
            )
        return _rows(cur)


# --- Weekly reports ---

def _decode_report(row: dict | None) -> dict | None:
    if row is None:
        return None
    row["data"] = json.loads(row["data"])
    return row


def get_weekly_report(week_start: str) -> dict | None:
    with _conn() as conn:
        row = _one(conn.execute(
            "SELECT week_start, data, updated_at, read_at FROM weekly_reports "
            "WHERE week_start = ?",
            (week_start,),
        ))
    return _decode_report(row)


def upsert_weekly_report(week_start: str, data: dict) -> None:
    """Write a week's report. Any update re-flags the week as unread."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO weekly_reports (week_start, data, updated_at, read_at) "
            "VALUES (?, ?, ?, NULL) "
            "ON CONFLICT(week_start) DO UPDATE SET "
            "data = excluded.data, updated_at = excluded.updated_at, read_at = NULL",
            (week_start, json.dumps(data), now),
        )
        conn.commit()


def list_weekly_reports() -> list[dict]:
    """All weekly reports, newest first, with decoded data."""
    with _conn() as conn:
        rows = _rows(conn.execute(
            "SELECT week_start, data, updated_at, read_at FROM weekly_reports "
            "ORDER BY week_start DESC"
        ))
    return [_decode_report(r) for r in rows]


def mark_weekly_report_read(week_start: str) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE weekly_reports SET read_at = ? WHERE week_start = ?",
            (now, week_start),
        )
        conn.commit()
        return cur.rowcount > 0


# --- Rolling "since last read" report ---

def get_unread_report() -> dict | None:
    with _conn() as conn:
        row = _one(conn.execute(
            "SELECT data, updated_at FROM unread_report WHERE id = 1"
        ))
    return _decode_report(row)


def set_unread_report(data: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO unread_report (id, data, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
            (json.dumps(data), now),
        )
        conn.commit()


def clear_unread_report() -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM unread_report WHERE id = 1")
        conn.commit()
