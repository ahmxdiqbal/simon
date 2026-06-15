"""Week bucketing tests."""

from __future__ import annotations

from datetime import datetime, timezone

from weeks import group_by_week, week_label, week_start


def test_sunday_maps_to_itself():
    # 2026-06-14 is a Sunday.
    assert week_start(datetime(2026, 6, 14, 0, 0, tzinfo=timezone.utc)) == "2026-06-14"
    assert week_start(datetime(2026, 6, 14, 23, 59, tzinfo=timezone.utc)) == "2026-06-14"


def test_midweek_maps_back_to_sunday():
    # Monday through Saturday all belong to the prior Sunday.
    assert week_start(datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)) == "2026-06-14"  # Mon
    assert week_start(datetime(2026, 6, 20, 23, 0, tzinfo=timezone.utc)) == "2026-06-14"  # Sat


def test_next_sunday_starts_new_week():
    assert week_start(datetime(2026, 6, 21, 0, 0, tzinfo=timezone.utc)) == "2026-06-21"


def test_naive_datetime_treated_as_utc():
    assert week_start(datetime(2026, 6, 15, 12, 0)) == "2026-06-14"


def test_non_utc_converted_before_bucketing():
    from datetime import timedelta
    # 2026-06-21 01:00 at UTC+2 is 2026-06-20 23:00 UTC -> still the 06-14 week.
    tz = timezone(timedelta(hours=2))
    assert week_start(datetime(2026, 6, 21, 1, 0, tzinfo=tz)) == "2026-06-14"


def test_label_format():
    assert week_label("2026-06-14") == "Week of 06/14/2026"


def test_group_by_week_splits_across_boundary():
    messages = [
        {"id": 1, "sent_at": "2026-06-20T23:00:00+00:00"},  # Sat -> 06-14
        {"id": 2, "sent_at": "2026-06-21T00:30:00+00:00"},  # Sun -> 06-21
        {"id": 3, "sent_at": "2026-06-21T06:00:00+00:00"},  # Sun -> 06-21
    ]
    groups = group_by_week(messages)
    assert set(groups) == {"2026-06-14", "2026-06-21"}
    assert [m["id"] for m in groups["2026-06-14"]] == [1]
    assert [m["id"] for m in groups["2026-06-21"]] == [2, 3]
