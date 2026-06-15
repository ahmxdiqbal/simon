"""DB layer tests against a temporary local libSQL file."""

from __future__ import annotations

import importlib
from datetime import datetime, timezone

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("TURSO_DATABASE_URL", str(tmp_path / "test.db"))
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "")
    import db as db_module
    importlib.reload(db_module)
    db_module.init_db()
    return db_module


def test_channels_crud(db):
    db.add_channel("@clashreport", "Clash Report")
    db.add_channel("clashreport")  # duplicate ignored
    rows = db.list_channels()
    assert len(rows) == 1
    assert rows[0]["username"] == "clashreport"
    assert rows[0]["title"] == "Clash Report"

    db.update_channel_title("clashreport", "Clash")
    assert db.list_channels()[0]["title"] == "Clash"

    assert db.remove_channel("@clashreport") is True
    assert db.remove_channel("clashreport") is False
    assert db.list_channels() == []


def test_state_and_cursors(db):
    assert db.get_last_refresh_at() is None
    ts = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    db.set_last_refresh_at(ts)
    db.set_last_read_at(ts)
    assert db.get_last_refresh_at() == ts
    assert db.get_last_read_at() == ts

    db.set_state("k", "v1")
    db.set_state("k", "v2")  # upsert overwrites
    assert db.get_state("k") == "v2"


def test_store_messages_dedup(db):
    msgs = [
        {"id": 10, "sent_at": "2026-06-15T01:00:00+00:00", "text": "a"},
        {"id": 11, "sent_at": "2026-06-15T02:00:00+00:00", "text": "b"},
    ]
    assert db.store_messages("clashreport", msgs) == 2
    # Re-storing the same ids inserts nothing new.
    assert db.store_messages("clashreport", msgs) == 0

    since = datetime(2026, 6, 15, 1, 30, tzinfo=timezone.utc)
    later = db.get_messages_since(since)
    assert [m["message_id"] for m in later] == [11]
    assert len(db.get_messages_since(None)) == 2


def test_weekly_report_upsert_and_reflag(db):
    data1 = {"events": [{"text": "e1", "countries": ["X"], "sources": []}]}
    db.upsert_weekly_report("2026-06-14", data1)
    rpt = db.get_weekly_report("2026-06-14")
    assert rpt["data"] == data1
    assert rpt["read_at"] is None

    # Marking read sets read_at.
    assert db.mark_weekly_report_read("2026-06-14") is True
    assert db.get_weekly_report("2026-06-14")["read_at"] is not None

    # A later upsert (new content) re-flags it unread.
    data2 = {"events": [{"text": "e1+e2", "countries": ["X", "Y"], "sources": []}]}
    db.upsert_weekly_report("2026-06-14", data2)
    rpt = db.get_weekly_report("2026-06-14")
    assert rpt["data"] == data2
    assert rpt["read_at"] is None


def test_weekly_reports_listed_newest_first(db):
    db.upsert_weekly_report("2026-06-07", {"events": []})
    db.upsert_weekly_report("2026-06-21", {"events": []})
    db.upsert_weekly_report("2026-06-14", {"events": []})
    keys = [r["week_start"] for r in db.list_weekly_reports()]
    assert keys == ["2026-06-21", "2026-06-14", "2026-06-07"]


def test_unread_report_lifecycle(db):
    assert db.get_unread_report() is None
    db.set_unread_report({"events": [{"text": "x"}]})
    assert db.get_unread_report()["data"]["events"][0]["text"] == "x"
    # Single-row: a second write overwrites rather than appends.
    db.set_unread_report({"events": [{"text": "y"}]})
    assert db.get_unread_report()["data"]["events"][0]["text"] == "y"
    db.clear_unread_report()
    assert db.get_unread_report() is None
