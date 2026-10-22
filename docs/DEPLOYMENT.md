# Deployment & execution guide

## A. One-time setup (about 10 minutes + model download)
1. **Python 3.12** - https://www.python.org/downloads/ - tick **Add python.exe to PATH**. (Python 3.13 packages also exist, but 3.12 is the tested version.)
2. **Unzip** the project to a folder without spaces if possible, e.g. `C:\MediaIntel`.
3. **Create `.env`**: copy `.env.example` -> `.env`, then fill in
   * `YOUTUBE_API_KEY` - Google Cloud Console -> create a project -> *APIs & Services* -> enable **YouTube Data API v3** -> *Credentials* -> API key.
   * `SENDER_EMAIL`, `SENDER_PASSWORD`, `RECIPIENT_EMAIL` - for Gmail create an **App password** (Google Account -> Security -> 2-Step Verification -> App passwords) and use it as `SENDER_PASSWORD`.
   * Leave the database section empty to use SQLite (nothing to install). For MySQL see the comments in `.env.example`.
4. **Channels**: open `target_channels.json` and list the competitors (`channel_id` starts with `UC...`; find it in a channel's *About -> Share -> Copy channel ID*). Set `"is_tracked": false` to stop tracking one without deleting its data. At least 5 are recommended.

## B. Run
Double-click **`run.bat`** (or in Command Prompt: `cd C:\MediaIntel` then `run.bat`).
* First run: creates a private Python environment, installs the packages and downloads the AI models (~2 GB). Be patient; later runs skip this.
* Then you see the pre-checks and 6 progress bars. When they finish, the PDF and e-mail are done and the dashboard opens in your browser (stop it with Ctrl+C).
* Recommended first run: `run.bat --no-email`, check `outputs/`, then run normally.

| Command | Effect |
|---|---|
| `run.bat` | full pipeline + dashboard |
| `run.bat --quick` | skip collection and AI; only anomalies + RPA reports/e-mail |
| `run.bat --no-email` | the bot saves the e-mail to `data/rpa_logs/outbox` instead of sending |
| `run.bat --no-dashboard` | do not open the browser |
| `python tests/selftest.py` | verifies the whole pipeline on a throwaway database (no key needed) |

## C. Where everything is
| What | Where |
|---|---|
| PDF report | `outputs/MediaIntel_Dashboard_Documentation.pdf` (older copies in `outputs/archive/`) |
| Strategy report / alert summary | `outputs/Content_Strategy_Report.txt`, `outputs/AlertSummary.txt` |
| Dashboard | opens automatically, or `.venv\Scripts\streamlit run dashboard\dashboard_app.py` |
| RPA bot log (visual) | `data/rpa_logs/rpa_log.html`; every e-mail copy in `data/rpa_logs/outbox/` |
| Full log of every run | `data/logs/pipeline.log` (stage times are here) |
| Database | `data/mediaintel.db` |

## D. Tuning speed and depth
Set `SPEED_PROFILE` in `.env` (`fast`, `balanced` default, `accurate`) - see `.env.example`.
Whisper covers the next few videos of every channel per run and keeps going on later runs; RoBERTa analyses the 25 most-liked comments of each video (`MAX_COMMENTS_PER_VIDEO`, 0 = all comments) on the first run and only new ones afterwards. At most 10 alerts (`MAX_ALERTS_PER_EMAIL`) go into one e-mail; the rest wait for the next one.
After the first run, read the stage times in `data/logs/pipeline.log`; if Whisper dominates, lower `WHISPER_BUDGET_MIN` or use `fast`.

## E. Daily automatic run (optional, Windows Task Scheduler)
```
schtasks /Create /SC DAILY /ST 09:00 /TN "MediaIntel" /TR "C:\MediaIntel\run.bat --no-dashboard"
```
(Standard Windows command; not tested in the build environment.)

## F. Moving from an older version / the `Testing` build
* Keep your `.env` secrets but start from the new `.env.example` (setting names changed; common old names such as `SMTP_USER`, `ALERT_RECIPIENT`,
  `WHISPER_MODEL_SIZE`, `VARIANCE_THRESHOLD_PCT` are still understood).
* If a database from another build is found, it is kept as `data/..._legacy_backup_<time>.db` and a clean one is created automatically.
* **Rotate** any API key or e-mail password that was ever sent in a zip file.

## G. Troubleshooting
| Message | Fix |
|---|---|
| `No .env file found` | Step A3. |
| `YOUTUBE_API_KEY is missing` / `no channel could be read` | Check the key, that the YouTube Data API v3 is enabled, and the channel IDs. |
| `No tracked channels` | Step A4. |
| Whisper stage shows "captions" or "unavailable" | YouTube refused the audio stream. `run.bat` updates `yt-dlp` on every start; the pipeline continues with captions and retries Whisper next run. |
| E-mail not sent | Gmail needs an App password; read `data/rpa_logs/rpa_log.html`. |
| Something looks odd in sentiment/aspects | `.venv\Scripts\python core\nlp_pipeline.py --preview 20` shows real predictions with evidence. |
| A stage shows ✖ Failed | Read the last lines under the progress bars or `data/logs/pipeline.log`. |
