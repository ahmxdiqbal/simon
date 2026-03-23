"""
Telegram channel message fetcher using Telethon (User API).

Authentication: on first run, prompts for phone + OTP and saves a session file.
Subsequent runs use the saved session silently.

The client connects once and stays connected for the lifetime of the process.
Entity resolution is cached so repeated fetches don't re-resolve channels.
Channel fetches run in parallel via asyncio.gather.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import Message

load_dotenv()

SESSION_PATH = Path(__file__).parent / "geopolitics.session"
API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]

MAX_PER_CHANNEL = 2000

# --- Persistent client and entity cache ---

_client: TelegramClient | None = None
_client_lock: asyncio.Lock | None = None
_entity_cache: dict[str, object] = {}


def _make_client() -> TelegramClient:
    return TelegramClient(str(SESSION_PATH), API_ID, API_HASH)


async def _get_client() -> TelegramClient:
    """Return a persistent, connected TelegramClient."""
    global _client, _client_lock
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    async with _client_lock:
        if _client is None or not _client.is_connected():
            _client = _make_client()
            await _client.connect()
            if not await _client.is_user_authorized():
                raise RuntimeError("Telegram session not authorized. Run: python telegram_fetcher.py")
    return _client


async def _resolve_entity(client: TelegramClient, username: str):
    """Resolve a channel entity, using cache to avoid repeated API calls."""
    if username in _entity_cache:
        return _entity_cache[username]
    entity = await client.get_entity(f"https://t.me/{username}")
    _entity_cache[username] = entity
    return entity


async def authenticate() -> None:
    """Interactive first-time login. Run once manually if needed."""
    async with _make_client() as client:
        await client.get_me()
    print("Authentication successful. Session saved.")


async def fetch_messages_since(
    usernames: list[str],
    since: datetime | None,
    on_progress: callable | None = None,
) -> list[dict]:
    """
    Fetch messages from all channels posted after `since`.
    Channels are fetched in parallel. Returns list of dicts sorted chronologically.
    """
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    client = await _get_client()

    async def fetch_one(username: str) -> list[dict]:
        if on_progress:
            on_progress(f"Fetching @{username}...")
        try:
            return await _fetch_channel(client, username, since)
        except FloodWaitError as e:
            print(f"  Rate limited on @{username}. Waiting {e.seconds}s...")
            await asyncio.sleep(e.seconds)
            try:
                return await _fetch_channel(client, username, since)
            except Exception as retry_err:
                print(f"  Skipping @{username} after retry: {retry_err}")
                return []
        except Exception as e:
            print(f"  Skipping @{username}: {e}")
            return []

    # Fetch all channels in parallel
    results = await asyncio.gather(*[fetch_one(u) for u in usernames])

    all_messages: list[dict] = []
    for channel_msgs in results:
        all_messages.extend(channel_msgs)

    all_messages.sort(key=lambda m: m["sent_at"])
    return all_messages


async def _fetch_channel(
    client: TelegramClient,
    username: str,
    since: datetime | None,
) -> list[dict]:
    messages: list[dict] = []
    entity = await _resolve_entity(client, username)
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


def run_fetch(usernames: list[str], since: datetime | None, on_progress=None) -> list[dict]:
    """Synchronous wrapper for use outside async contexts."""
    return asyncio.run(fetch_messages_since(usernames, since, on_progress))


def run_authenticate() -> None:
    """Synchronous wrapper for first-time auth."""
    asyncio.run(authenticate())


if __name__ == "__main__":
    run_authenticate()
