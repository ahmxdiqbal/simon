"""
One-off: fetch the last 6h of messages and generate the DeepSeek baseline.

Re-run any time. It skips files that already exist, so it's safe to run
repeatedly. To regenerate, delete the target files and run again:

    rm tests/fixtures/messages_6h.json tests/fixtures/deepseek_baseline.json
    python tests/collect_fixture.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MESSAGES_FILE = FIXTURE_DIR / "messages_6h.json"
BASELINE_FILE = FIXTURE_DIR / "deepseek_baseline.json"

FETCH_HOURS = 6


def _log(msg: str) -> None:
    print(f"  {msg}")


def collect_messages() -> dict:
    if MESSAGES_FILE.exists():
        print(f"[skip] {MESSAGES_FILE.name} already exists")
        with MESSAGES_FILE.open() as f:
            return json.load(f)

    from db import list_channels
    from telegram_fetcher import run_fetch

    channels = list_channels()
    if not channels:
        print("ERROR: no channels in DB. Add channels via the dashboard first.")
        sys.exit(1)

    usernames = [c["username"] for c in channels]
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=FETCH_HOURS)

    print(f"[fetch] last {FETCH_HOURS}h from {len(usernames)} channels...")
    messages = run_fetch(usernames, since, _log)

    fixture = {
        "fetched_at": now.isoformat(),
        "from_ts": since.isoformat(),
        "to_ts": now.isoformat(),
        "message_count": len(messages),
        "messages": messages,
    }

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with MESSAGES_FILE.open("w") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)
    print(f"[save] {MESSAGES_FILE.name} ({len(messages)} messages)")
    return fixture


def generate_baseline(fixture: dict) -> None:
    if BASELINE_FILE.exists():
        print(f"[skip] {BASELINE_FILE.name} already exists")
        return

    from summarizer_deepseek import summarize_deepseek

    from_ts = datetime.fromisoformat(fixture["from_ts"])
    to_ts = datetime.fromisoformat(fixture["to_ts"])

    print(f"[deepseek] summarizing {fixture['message_count']} messages...")
    t0 = time.monotonic()
    result = summarize_deepseek(fixture["messages"], from_ts, to_ts, _log)
    duration = time.monotonic() - t0

    # Store duration alongside the result so bench_local.py can report it.
    result["duration_seconds"] = duration

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with BASELINE_FILE.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    cost = result["cost"]["total_cost_usd"]
    print(
        f"[save] {BASELINE_FILE.name} "
        f"({len(result['events'])} events, {duration:.1f}s, ${cost:.4f})"
    )


if __name__ == "__main__":
    fixture = collect_messages()
    generate_baseline(fixture)
    print("\nDone. Next: python tests/bench_local.py")
