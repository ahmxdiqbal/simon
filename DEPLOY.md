# Deploying SiMon (Vercel + GitHub Actions + Turso)

Architecture:

- **Vercel** serves the dashboard and the read/write API (FastAPI in `api/index.py`).
- **GitHub Actions** runs the worker (`worker/refresh.py`) on a schedule and on demand: it fetches Telegram messages and summarizes them.
- **Turso** (libSQL) is the shared database both sides read and write.

Local development is unchanged: with no `TURSO_DATABASE_URL` the app uses a local `dashboard.db`, and with no `TELEGRAM_SESSION` the fetcher uses the `geopolitics.session` file.

## 1. Turso database

```bash
# Install the CLI: https://docs.turso.tech/cli/installation
turso auth signup

# Create the DB. Importing the existing file preserves channels + history.
turso db create simon --from-file dashboard.db   # or: turso db create simon

turso db show simon --url          # -> TURSO_DATABASE_URL  (libsql://...)
turso db tokens create simon       # -> TURSO_AUTH_TOKEN
```

The worker creates any missing tables (`weekly_reports`, `unread_report`) on its first run, so importing an older `dashboard.db` is safe.

## 2. Telegram session string

The worker can't use a session file. Generate a portable session locally (needs `.env` with `TELEGRAM_API_ID` / `TELEGRAM_API_HASH`):

```bash
python scripts/gen_session.py
```

Copy the printed string. Treat it like a password.

## 3. GitHub Actions secrets

In the repo: Settings to Secrets and variables to Actions to New repository secret. Add:

| Secret | Value |
|---|---|
| `TELEGRAM_API_ID` | from my.telegram.org |
| `TELEGRAM_API_HASH` | from my.telegram.org |
| `TELEGRAM_SESSION` | the string from step 2 |
| `DEEPSEEK_API_KEY` | from platform.deepseek.com |
| `TURSO_DATABASE_URL` | from step 1 |
| `TURSO_AUTH_TOKEN` | from step 1 |

The workflow (`.github/workflows/refresh.yml`) runs every 3 hours, and can be run manually from the Actions tab.

## 4. Vercel

Import the repo at vercel.com. Under Settings to Environment Variables add:

| Variable | Value |
|---|---|
| `TURSO_DATABASE_URL` | from step 1 |
| `TURSO_AUTH_TOKEN` | from step 1 |

Optional, to enable the dashboard's **Refresh** button (which triggers the Action on demand):

| Variable | Value |
|---|---|
| `GITHUB_TOKEN` | a fine-grained PAT with **Actions: read and write** on this repo |
| `GITHUB_REPO` | `owner/repo` |

Without these two, the Refresh button reports that refresh runs on schedule; everything else works.

## 5. First run

1. Open the Vercel URL and add your channels (Channels button), or import them via the DB.
2. In the Actions tab, run the **refresh** workflow once. The first run only seeds the fetch cursor to "now" so it doesn't pull full history.
3. Run it again (or wait for the schedule). It now fetches new messages and builds the catch-up report and the current week's report.

## Notes

- `dashboard.db`, `geopolitics.session`, and `.env` are gitignored. Never commit them; their contents live in Turso and the secrets stores instead.
- GitHub Actions scheduled crons are best-effort and can run a few minutes late.
- The dashboard Refresh button triggers the Action and polls for completion (~1 minute including runner startup), so it is not instant.
