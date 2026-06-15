"""
FastAPI server for SiMon (read/serve layer).

Fetching and summarization run in the GitHub Actions worker, not here. This
app serves the dashboard, exposes the reports, manages channels, handles the
two read actions, and triggers an on-demand refresh by dispatching the worker.

Endpoints:
  GET    /                              - dashboard HTML
  GET    /api/status                    - cursors + unread/weekly counts
  GET    /api/reports                   - the catch-up report + weekly reports
  POST   /api/refresh                   - trigger the worker via GitHub dispatch
  POST   /api/mark-read                 - clear the catch-up report
  POST   /api/reports/{week_start}/mark-read - mark one weekly report read
  GET    /api/channels                  - list channels
  POST   /api/channels                  - add channel {"username": "@foo"}
  DELETE /api/channels/{username}       - remove channel
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db

app = FastAPI(title="SiMon")

# Static assets live in public/ and are served by Vercel's CDN in production.
# These routes serve them for local same-origin development only.
public_path = Path(__file__).parent / "public"
app.mount("/static", StaticFiles(directory=public_path / "static"), name="static")

# Ensure the schema exists. Idempotent; runs at import so it also covers
# serverless runtimes that skip ASGI lifespan events.
db.init_db()


# --- Models ---

class ChannelAdd(BaseModel):
    username: str


# --- Pages ---

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=(public_path / "index.html").read_text())


@app.get("/favicon.ico")
def favicon():
    return FileResponse(public_path / "static" / "simon-bolivar.png", media_type="image/png")


# --- Status & reports ---

@app.get("/api/status")
def get_status():
    last_refresh = db.get_last_refresh_at()
    unread = db.get_unread_report()
    weeks = db.list_weekly_reports()
    return {
        "last_refresh_at": last_refresh.isoformat() if last_refresh else None,
        "channel_count": len(db.list_channels()),
        "unread_available": unread is not None,
        "unread_updated_at": unread["updated_at"] if unread else None,
        "weekly_report_count": len(weeks),
        "unread_week_count": sum(1 for w in weeks if w["read_at"] is None),
    }


@app.get("/api/reports")
def get_reports():
    unread = db.get_unread_report()
    weeks = db.list_weekly_reports()
    return {
        "unread": {
            "data": unread["data"],
            "updated_at": unread["updated_at"],
        } if unread else None,
        "weeks": [
            {
                "week_start": w["week_start"],
                "data": w["data"],
                "updated_at": w["updated_at"],
                "read_at": w["read_at"],
            }
            for w in weeks
        ],
    }


# --- Refresh trigger ---

@app.post("/api/refresh", status_code=202)
def trigger_refresh():
    """Dispatch the GitHub Actions worker. Requires GITHUB_TOKEN + GITHUB_REPO."""
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPO")
    if not token or not repo:
        raise HTTPException(
            status_code=503,
            detail="On-demand refresh is not configured. Reports update on schedule.",
        )
    resp = httpx.post(
        f"https://api.github.com/repos/{repo}/dispatches",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        json={"event_type": "refresh"},
        timeout=10,
    )
    if resp.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"GitHub dispatch failed: {resp.status_code}")
    return {"status": "triggered"}


# --- Read actions ---

@app.post("/api/mark-read")
def mark_read():
    """Clear the catch-up report. Weekly reports are unaffected."""
    db.clear_unread_report()
    last_refresh = db.get_last_refresh_at()
    db.set_last_read_at(last_refresh or datetime.now(timezone.utc))
    return {"status": "ok"}


@app.post("/api/reports/{week_start}/mark-read")
def mark_week_read(week_start: str):
    if not db.mark_weekly_report_read(week_start):
        raise HTTPException(status_code=404, detail="Weekly report not found.")
    return {"status": "ok"}


# --- Channels ---

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
    if not db.remove_channel(username):
        raise HTTPException(status_code=404, detail="Channel not found.")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
