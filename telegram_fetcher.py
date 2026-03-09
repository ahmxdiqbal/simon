"""
Telegram channel message fetcher using Telethon (User API).

Authentication: on first run, prompts for phone + OTP and saves a session file.
Subsequent runs use the saved session silently.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import Message, Channel

load_dotenv()

SESSION_PATH = Path(__file__).parent / "geopolitics.session"
API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]

# Max messages to fetch per channel per run (safety cap)
MAX_PER_CHANNEL = 2000


def _make_client() -> TelegramClient:
    return TelegramClient(str(SESSION_PATH), API_ID, API_HASH)


async def authenticate() -> None:
    """Interactive first-time login. Run once manually if needed."""
    async with _make_client() as client:
        await client.get_me()
    print("Authentication successful. Session saved.")


async def fetch_channels_info(usernames: list[str]) -> dict[str, str]:
    """Resolve channel usernames to display titles."""
    titles: dict[str, str] = {}
    async with _make_client() as client:
        for username in usernames:
            try:
                entity = await client.get_entity(f"https://t.me/{username}")
                titles[username] = getattr(entity, "title", username)
            except Exception as e:
                print(f"  Warning: could not resolve @{username}: {e}")
                titles[username] = username
    return titles


async def fetch_messages_since(
    usernames: list[str],
    since: datetime | None,
    on_progress: callable | None = None,
) -> list[dict]:
    """
    Fetch messages from all channels posted after `since`.
    Returns list of dicts: {id, channel, channel_title, sent_at, text}
    """
    # Normalize to UTC-aware for comparison
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    all_messages: list[dict] = []

    async with _make_client() as client:
        for username in usernames:
            if on_progress:
                on_progress(f"Fetching @{username}...")
            try:
                channel_messages = await _fetch_channel(client, username, since)
                all_messages.extend(channel_messages)
            except FloodWaitError as e:
                print(f"  Rate limited. Waiting {e.seconds}s before continuing...")
                await asyncio.sleep(e.seconds)
                try:
                    channel_messages = await _fetch_channel(client, username, since)
                    all_messages.extend(channel_messages)
                except Exception as retry_err:
                    print(f"  Skipping @{username} after retry: {retry_err}")
            except Exception as e:
                print(f"  Skipping @{username}: {e}")

    # Sort chronologically
    all_messages.sort(key=lambda m: m["sent_at"])
    return all_messages


async def _fetch_channel(
    client: TelegramClient,
    username: str,
    since: datetime | None,
) -> list[dict]:
    messages: list[dict] = []
    # Use t.me URL so Telethon resolves it as a channel, not a user account
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
            break  # iter_messages goes newest-first; stop when we pass the cutoff

        messages.append(
            {
                "id": msg.id,
                "channel": username,
                "channel_title": title,
                "sent_at": msg_date.isoformat(),
                "text": text.strip(),
            }
        )

    # Return in chronological order
    messages.reverse()
    return messages


def run_fetch(usernames: list[str], since: datetime | None, on_progress=None) -> list[dict]:
    """Synchronous wrapper for use outside async contexts."""
    return asyncio.run(fetch_messages_since(usernames, since, on_progress))


def run_authenticate() -> None:
    """Synchronous wrapper for first-time auth."""
    asyncio.run(authenticate())


if __name__ == "__main__":
    # Run this directly to authenticate: python telegram_fetcher.py
    run_authenticate()
