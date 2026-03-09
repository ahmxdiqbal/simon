# SiMon

Personal geopolitics dashboard. Reads Telegram channels, extracts events via LLM, and displays them on a local web UI organized by country.

## How it works

1. Connects to Telegram as a regular user account (not a bot) via the [Telethon](https://github.com/LonamiWebs/Telethon) User API
2. Fetches messages from subscribed channels since you last clicked "Mark as Read"
3. Sends messages to DeepSeek-V3 to extract and deduplicate events with source citations
4. Displays results at `localhost:8000` with per-country filtering

Events are tagged with involved countries. The "All" tab shows everything; country tabs filter client-side. Each event has inline citations traced back to either a news outlet (if the channel cited one) or the Telegram channel itself.

## Setup

### Prerequisites

- Python 3.9+
- A Telegram account (not a bot)
- Telegram API credentials from [my.telegram.org/apps](https://my.telegram.org/apps)
- DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com)

### Install

```
git clone <repo-url> && cd SiMon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Create `.env`:

```
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
DEEPSEEK_API_KEY=your_key
```

### First run

Authenticate with Telegram (interactive, saves a session file):

```
python telegram_fetcher.py
```

Start the server:

```
python app.py
```

Open `http://localhost:8000`. Add channels via the Channels button (e.g. `@clashreport`), then hit Refresh.

## File structure

| File | Purpose |
|---|---|
| `app.py` | FastAPI server, all API routes |
| `db.py` | SQLite wrapper (channels, messages, summaries, state) |
| `telegram_fetcher.py` | Telethon message fetcher |
| `summarizer.py` | DeepSeek summarization + citation processing |
| `static/index.html` | Single-page dashboard UI |

## Cost

DeepSeek-V3 pricing: $0.27/MTok input, $1.10/MTok output. A typical refresh of a few hundred messages costs under $0.01. Cost per run is displayed in the header.
