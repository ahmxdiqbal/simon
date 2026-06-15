"""
Telegram channel message fetcher using Telethon (User API).

Authentication: uses a StringSession from TELEGRAM_SESSION, or a local session
file for first-time auth on a developer machine (run this module directly).

One-shot: each call opens a client, fetches all channels in parallel, and
disconnects. Suits the per-run worker process; no persistent connection.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.sessions import StringSession
from telethon.tl.types import Message

load_dotenv()

SESSION_PATH = Path(__file__).parent / "geopolitics.session"
SESSION_STRING = os.environ.get("TELEGRAM_SESSION")
API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]

MAX_PER_CHANNEL = 2000


def _make_client() -> TelegramClient:
    # Prefer an env-provided StringSession (serverless / CI); fall back to the
    # local session file for first-time auth on a developer machine.
    if SESSION_STRING:
        return TelegramClient(StringSession(SESSION_STRING), API_ID, API_HASH)
    return TelegramClient(str(SESSION_PATH), API_ID, API_HASH)


async def authenticate() -> None:
    """Interactive first-time login. Run once manually if needed."""
    async with _make_client() as client:
        await client.get_me()
    print("Authentication successful. Session saved.")


async def fetch_messages_since(
    usernames: list[str],
    since: datetime | None,
) -> list[dict]:
    """Fetch messages from all channels posted after `since`, sorted chronologically."""
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    total_channels = len(usernames)
    print(f"Fetching {total_channels} channels...")

    async with _make_client() as client:
        if not await client.is_user_authorized():
            raise RuntimeError(
                "Telegram session not authorized. Generate one with: python scripts/gen_session.py"
            )

        async def fetch_one(username: str) -> list[dict]:
            try:
                return await _fetch_channel(client, username, since)
            except FloodWaitError as e:
                print(f"@{username}: rate limited, waiting {e.seconds}s...")
                await asyncio.sleep(e.seconds)
                try:
                    return await _fetch_channel(client, username, since)
                except Exception as retry_err:
                    print(f"  Skipping @{username} after retry: {retry_err}")
                    return []
            except Exception as e:
                print(f"  Skipping @{username}: {e}")
                return []

        results = await asyncio.gather(*[fetch_one(u) for u in usernames])

    all_messages: list[dict] = []
    for channel_msgs in results:
        all_messages.extend(channel_msgs)
    all_messages.sort(key=lambda m: m["sent_at"])

    print(f"Fetched {len(all_messages)} messages from {total_channels} channels")
    return all_messages


async def _fetch_channel(
    client: TelegramClient,
    username: str,
    since: datetime | None,
) -> list[dict]:
    messages: list[dict] = []
    entity = await client.get_entity(f"https://t.me/{username}")
    title = getattr(entity, "title", username)

    async for msg in client.iter_messages(entity, limit=MAX_PER_CHANNEL):
        if not isinstance(msg, Message):
            continue

        text = msg.text or msg.message or ""
        if not text.strip():
            continue

        msg_date = msg.date
        if msg_date.tzinfo is None:
            msg_date = msg_date.replace(tzinfo=timezone.utc)

        if since is not None and msg_date <= since:
            break

        messages.append(
            {
                "id": msg.id,
                "channel": username,
                "channel_title": title,
                "sent_at": msg_date.isoformat(),
                "text": text.strip(),
            }
        )

    messages.reverse()
    return messages


def run_fetch(usernames: list[str], since: datetime | None) -> list[dict]:
    """Synchronous wrapper for use outside async contexts."""
    return asyncio.run(fetch_messages_since(usernames, since))


def run_authenticate() -> None:
    """Synchronous wrapper for first-time auth."""
    asyncio.run(authenticate())


if __name__ == "__main__":
    run_authenticate()
