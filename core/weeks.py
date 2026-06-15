"""
Week bucketing helpers. Weeks run Sunday to Sunday in UTC.

A week is identified by its start date (the Sunday), as an ISO date string
like "2026-06-14". Reports are keyed by this value.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


def week_start(dt: datetime) -> str:
    """Return the ISO date of the UTC Sunday that starts dt's week."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    d = dt.astimezone(timezone.utc).date()
    # weekday(): Monday=0 .. Sunday=6. Days back to Sunday:
    days_since_sunday = (d.weekday() + 1) % 7
    return (d - timedelta(days=days_since_sunday)).isoformat()


def week_label(week_start_iso: str) -> str:
    """Human label for a week key, e.g. 'Week of 06/14/2026'."""
    return f"Week of {date.fromisoformat(week_start_iso).strftime('%m/%d/%Y')}"


def group_by_week(messages: list[dict]) -> dict[str, list[dict]]:
    """Group messages by their week key, preserving order within each week."""
    groups: dict[str, list[dict]] = {}
    for m in messages:
        key = week_start(datetime.fromisoformat(m["sent_at"]))
        groups.setdefault(key, []).append(m)
    return groups
