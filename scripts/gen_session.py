"""
One-time: generate a Telethon StringSession for use as the TELEGRAM_SESSION
secret (GitHub Actions / Vercel). Run locally and follow the login prompts.

    python scripts/gen_session.py

Treat the printed string like a password.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.sessions import StringSession

load_dotenv()

api_id = int(os.environ["TELEGRAM_API_ID"])
api_hash = os.environ["TELEGRAM_API_HASH"]

with TelegramClient(StringSession(), api_id, api_hash) as client:
    print("\nLogged in. Add this to your secrets as TELEGRAM_SESSION:\n")
    print(client.session.save())
