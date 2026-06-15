"""
Refresh worker. Run on a schedule (GitHub Actions) or on demand.

Fetches messages newer than the last refresh, caches them, and merges them
into both the per-week reports and the rolling "since last read" report.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python worker/refresh.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The worker always uses the API backend; local MLX never runs here.
os.environ.setdefault("SUMMARIZER_BACKEND", "deepseek")

import db
import summarizer
import telegram_fetcher
from weeks import group_by_week


def _cache_raw(messages: list[dict]) -> None:
    by_channel: dict[str, list[dict]] = {}
    for m in messages:
        by_channel.setdefault(m["channel"], []).append(
            {"id": m["id"], "sent_at": m["sent_at"], "text": m["text"]}
        )
    for channel, msgs in by_channel.items():
        db.store_messages(channel, msgs)


def _update_titles(messages: list[dict]) -> None:
    seen: dict[str, str] = {}
    for m in messages:
        if m["channel"] not in seen and m.get("channel_title"):
            seen[m["channel"]] = m["channel_title"]
    for username, title in seen.items():
        db.update_channel_title(username, title)


def run() -> None:
    db.init_db()
    now = datetime.now(timezone.utc)

    last_refresh = db.get_last_refresh_at()
    if last_refresh is None:
        # First run: seed cursors to now so we don't pull the full channel history.
        db.set_last_refresh_at(now)
        db.set_last_read_at(now)
        print("First run: cursors seeded to now. Future messages only from here.")
        return

    channels = db.list_channels()
    if not channels:
        print("No channels configured; nothing to fetch.")
        return

    usernames = [c["username"] for c in channels]
    new_messages = telegram_fetcher.run_fetch(usernames, last_refresh)
    if not new_messages:
        # Advance the cursor anyway so callers can tell the run completed.
        db.set_last_refresh_at(now)
        print("No new messages since last refresh.")
        return

    _cache_raw(new_messages)
    _update_titles(new_messages)

    # Per-week reports: merge each week's slice into its report.
    for week_start, msgs in group_by_week(new_messages).items():
        prior = db.get_weekly_report(week_start)
        if prior:
            report = summarizer.summarize_incremental(prior["data"], msgs, now)
        else:
            start = datetime.fromisoformat(week_start).replace(tzinfo=timezone.utc)
            report = summarizer.summarize(msgs, start, now)
        db.upsert_weekly_report(week_start, report)

    # Rolling "since last read" report: merge the whole batch.
    prior_unread = db.get_unread_report()
    if prior_unread:
        unread = summarizer.summarize_incremental(prior_unread["data"], new_messages, now)
    else:
        unread = summarizer.summarize(new_messages, db.get_last_read_at(), now)
    db.set_unread_report(unread)

    db.set_last_refresh_at(now)
    print(f"Refresh complete: {len(new_messages)} new messages processed.")


if __name__ == "__main__":
    run()
