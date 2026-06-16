# SiMon

Personal geopolitics dashboard. Reads Telegram channels, summarizes events with an LLM, and shows them on a web dashboard organized by country, as a self-updating "since last read" catch-up plus a per-week archive.

## Architecture

Three pieces share a database so nothing has to run on your machine:

- **Vercel** serves the dashboard and the read/write API ([app.py](app.py), entry [api/index.py](api/index.py)).
- **GitHub Actions** runs the worker ([worker/refresh.py](worker/refresh.py)) on a schedule and on demand: it fetches Telegram messages and summarizes them.
- **Turso** (libSQL) is the shared database both sides use.

## How it works

1. The worker connects to Telegram as a regular user account (not a bot) via the [Telethon](https://github.com/LonamiWebs/Telethon) User API and fetches messages since the last run.
2. New messages are summarized with DeepSeek-V3, merged incrementally into the report (only the delta plus the existing summary is sent, keeping cost down) with inline source citations.
3. Each run updates two reports: the **current week's** report (one per UTC-Sunday week, kept as an archive) and the rolling **"since last read"** catch-up.
4. The dashboard shows the catch-up on top and the weekly archive below, with per-country tabs. Citations link back to the news outlet or the Telegram channel.

## Repo layout

| Path | Purpose |
|---|---|
| `app.py`, `api/`, `public/` | Web layer: API + dashboard, served by Vercel |
| `core/` | Shared modules (`db`, `weeks`) used by both runtimes |
| `worker/` | Telegram fetch + summarization, run by GitHub Actions |
| `scripts/gen_session.py` | One-time: generate the Telegram session string |
| `tests/` | Unit tests (`pytest`) |

## Deploying

See [DEPLOY.md](DEPLOY.md) for the full Turso + secrets + Vercel setup.

## Local development

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-worker.txt
```

Create `.env`:

```
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
DEEPSEEK_API_KEY=your_key
```

With no `TURSO_DATABASE_URL` set, the app uses a local `dashboard.db` file. Authenticate Telegram once (`python -m worker.telegram_fetcher` saves a session file, or `python scripts/gen_session.py` prints a portable session string), then:

```
python -m worker.refresh   # fetch + summarize into the database
python app.py              # serve the dashboard at localhost:8000
```

Run the tests with `pytest`.

## Cost

DeepSeek-V3 pricing: $0.28/MTok input, $0.42/MTok output. Incremental merging keeps each refresh small; a typical run costs well under $0.01. Per-report cost is shown in the dashboard header.
