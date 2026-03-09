"""
FastAPI server for SiMon.

Endpoints:
  GET  /                  - serve the dashboard HTML
  GET  /api/status        - last_read_at, channel count, cached summary metadata
  POST /api/refresh       - fetch new messages + summarize, return summary
  POST /api/mark-read     - set last_read_at to now
  GET  /api/summary       - return latest cached summary
  GET  /api/channels      - list channels
  POST /api/channels      - add channel {"username": "@foo"}
  DELETE /api/channels/{username} - remove channel
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import summarizer
import telegram_fetcher

app = FastAPI(title="SiMon")

# Serve static files (the frontend)
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")


# --- Startup ---

@app.on_event("startup")
def on_startup():
    db.init_db()
    # Seed last_read_at to now on first run so Refresh doesn't pull full channel history.
    if db.get_last_read_at() is None:
        db.set_last_read_at(datetime.now(timezone.utc))
        print("First run: last_read_at seeded to now. Refresh will only fetch future messages.")


# --- Models ---

class ChannelAdd(BaseModel):
    username: str


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
def index():
    html = (static_path / "index.html").read_text()
    return HTMLResponse(content=html)


@app.get("/api/status")
def get_status():
    last_read = db.get_last_read_at()
    channels = db.list_channels()
    latest = db.get_latest_summary()
    return {
        "last_read_at": last_read.isoformat() if last_read else None,
        "channel_count": len(channels),
        "summary_available": latest is not None,
        "summary_created_at": latest["created_at"] if latest else None,
    }


@app.post("/api/refresh")
async def refresh():
    """Fetch new messages from Telegram and summarize them."""
    channels = db.list_channels()
    if not channels:
        raise HTTPException(status_code=400, detail="No channels configured. Add channels first.")

    usernames = [c["username"] for c in channels]
    last_read = db.get_last_read_at()
    now = datetime.now(timezone.utc)

    # Fetch from Telegram
    try:
        messages = await asyncio.to_thread(
            telegram_fetcher.run_fetch, usernames, last_read
        )
    except Exception as e:
        print(f"[ERROR] Telegram fetch failed: {e}")
        raise HTTPException(status_code=502, detail=f"Telegram fetch failed: {e}")

    # Cache any newly fetched messages
    for msg in messages:
        db.store_messages(
            msg["channel"],
            [{"id": msg["id"], "sent_at": msg["sent_at"], "text": msg["text"]}],
        )

    # If Telegram returned nothing (e.g. rate-limited after a prior failed attempt),
    # fall back to SQLite cache so a previous successful fetch isn't lost.
    if not messages:
        messages = db.get_messages_since(last_read)

    if not messages:
        return {"status": "no_new_messages", "message_count": 0, "nations": {}}

    # Summarize
    try:
        summary = await asyncio.to_thread(
            summarizer.summarize, messages, last_read, now
        )
    except Exception as e:
        print(f"[ERROR] Summarization failed: {e}")
        raise HTTPException(status_code=502, detail=f"Summarization failed: {e}")

    # Update channel titles in DB
    seen_titles: dict[str, str] = {}
    for msg in messages:
        if msg["channel"] not in seen_titles and msg.get("channel_title"):
            seen_titles[msg["channel"]] = msg["channel_title"]
    for username, title in seen_titles.items():
        db.update_channel_title(username, title)

    db.store_summary(last_read, now, summary)
    db.set_state("last_refresh_at", now.isoformat())

    return {"status": "ok", **summary}


@app.get("/api/summary")
def get_summary():
    latest = db.get_latest_summary()
    if latest is None:
        return {"status": "no_summary", "nations": {}}
    return {"status": "ok", **latest["data"], "created_at": latest["created_at"]}


@app.post("/api/mark-read")
def mark_read():
    # Use last_refresh_at so the next fetch starts from when we last refreshed,
    # not from when the button was clicked. This prevents the gap between
    # refresh and mark-as-read from being silently dropped.
    last_refresh = db.get_state("last_refresh_at")
    ts = datetime.fromisoformat(last_refresh) if last_refresh else datetime.now(timezone.utc)
    db.set_last_read_at(ts)
    return {"status": "ok", "marked_at": ts.isoformat()}


@app.get("/api/channels")
def get_channels():
    return db.list_channels()


@app.post("/api/channels", status_code=201)
def add_channel(body: ChannelAdd):
    username = body.username.lstrip("@").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty.")
    db.add_channel(username)
    return {"status": "ok", "username": username}


@app.delete("/api/channels/{username}")
def remove_channel(username: str):
    removed = db.remove_channel(username)
    if not removed:
        raise HTTPException(status_code=404, detail="Channel not found.")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
