# MediaIntel - AI-Driven Automated Competitive Intelligence Platform using NLP and RPA

| | |
|---|---|
| **Institution** | CMR College of Engineering & Technology (UGC Autonomous), Hyderabad |
| **Department** | CSE (Data Science) - Major Project (A467803), Batch 21, IV/I, 2026-2027 |
| **Guide** | Ms. R. Dhanalakshmi, Assistant Professor |
| **Team** | Kacham Sathvika (23H51A67F1) - Kurapati Harinath (23H51A6733) - Baling Kausthub (24H55A6702) |

MediaIntel collects competitor YouTube data, analyses the audio with **OpenAI Whisper** (streamed into memory - nothing is stored),
runs **RoBERTa aspect-based sentiment analysis on the most-liked comments of every video (top 25 per video by default; set `MAX_COMMENTS_PER_VIDEO=0` for all)**, measures how closely comments follow what each video says,
detects engagement anomalies, and lets a **Robot Framework RPA bot** generate the reports and e-mail them with the alerts - all from one click.

> **Honesty rule:** nothing is ever simulated. If the YouTube key is missing or the API fails, the pipeline stops and tells you.
> Every number in the dashboard, PDF and e-mail is computed from collected data. See [`docs/ABSTRACT_COVERAGE.md`](docs/ABSTRACT_COVERAGE.md).

## Run it - 5 steps (Windows)
1. Install **Python 3.12** (tick *Add python.exe to PATH*).
2. Copy `.env.example` to `.env`; paste your **YouTube API key** and e-mail settings (Gmail needs an *App password*).
3. Edit `target_channels.json` - the competitors to track (the brief asks for at least 5; five are pre-filled).
4. Double-click **`run.bat`**. First run installs packages and downloads the AI models (~2 GB, several minutes).
5. Read the progress bars; when they finish the dashboard opens in your browser.

Full guide with troubleshooting: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).  Linux/macOS: `./run.sh`.
Options: `--quick` (re-analyse existing data only) - `--no-email` (save the e-mail instead of sending) - `--no-dashboard`.
Verify the install without any API key: `python tests/selftest.py` (runs everything on a throwaway database).

## The 6 stages
| # | Stage | What it does |
|---|-------|--------------|
| 1 | YouTube Ingestion | latest videos, statistics and comments per channel (one batched API call per channel, comments in parallel) + a statistics snapshot per run |
| 2 | Whisper Audio Analysis | the first minutes of each video's audio are **streamed into memory** through ffmpeg and transcribed by Whisper - **no audio file is ever written to disk**. Fair round-robin across channels, time budget, cached; YouTube captions are only a stop-gap |
| 3 | RoBERTa Sentiment | each analysed comment (the top 25 most-liked per video by default) is split into clauses; **XLM-RoBERTa** scores each clause; each clause is matched to one or two aspects (Content Quality, Engagement, Relevance, Feature Request, Pricing/Cost, Technical Issue) and stored with its evidence |
| 4 | Relevance & Topics | multilingual embeddings compare transcript passages with comments -> 0-100 relevance index per video; topics viewers mention/request are ranked by demand, requests, content gap and sentiment |
| 5 | Anomaly Detection | hourly rate of views, likes and comments vs the channel baseline (default 3x) plus a between-runs spike check; severity MEDIUM / HIGH / CRITICAL; no repeat alerts within 24 h |
| 6 | RPA Reports & Alerts | Robot Framework bot: generates the strategy report and the PDF, verifies and archives them, composes the e-mail, attaches PDF + alert summary, sends it, marks alerts as sent |

## What you get
* **Dashboard (8 tabs):** Overview - Sentiment Intelligence (aspect scorecard + evidence explorer) - Video Performance - **Whisper Audio Analysis** - Relevance & Next Topics - Channel Comparison - Alerts (live velocity + severity) - System Health
* **PDF report** (`outputs/`): dashboard charts, aspect scorecard, **Whisper audio analysis**, relevance, next-video topics, anomaly alerts and **content recommendations from Whisper audio + viewer comments**
* **E-mail** with the PDF and alert summary attached, sent by the RPA bot

## Project structure
```
MediaIntel/
├── run.bat / run.sh              one-click launcher (creates venv, installs, runs)
├── main_orchestrator.py          the 6 stages with live progress bars, then the dashboard
├── target_channels.json          the competitors to track (single source of truth)
├── requirements.txt   .env.example
├── core/                         config, database, ingestion, Whisper, RoBERTa, aspects, relevance/topics,
│                                 anomaly detection, insights, strategy report, PDF report
├── rpa/                          mediaintel_rpa.robot + MediaIntelRPA.py (Robot Framework bot)
├── dashboard/                    dashboard_app.py (Streamlit) + assets/
├── outputs/                      PDF report, AlertSummary.txt, strategy report, archive/
├── data/                         mediaintel.db, logs/pipeline.log, rpa_logs/ (bot logs + e-mail outbox)
├── docs/                         ABSTRACT_COVERAGE.md, DEPLOYMENT.md
└── tests/                        selftest.py + stand-in models used only by the self-test
```

## Speed
`SPEED_PROFILE=fast | balanced | accurate` in `.env`. Whisper takes the next few videos of *every* channel per run (round-robin), never exceeds its
time budget and caches results; the backlog clears over successive runs. RoBERTa analyses **all** comments on the first run (identical comments are scored once)
and only new comments afterwards. Run times depend on your CPU - after your first run, `data/logs/pipeline.log` shows the time of every stage.

## Database
SQLite by default (nothing to install). For MySQL set `DB_TYPE=mysql` and the `DB_*` values in `.env`. Older databases are upgraded in place; a database from a
different build is kept as `data/..._legacy_backup_...db` and a clean one is created. Nothing is ever dropped.

## Known limitations
* YouTube only (other platforms are future work).
* Whisper analyses the first minutes of each video (profile-dependent), not the full length.
* Topic mining reads Latin-script text (English / romanised Hinglish); Devanagari-only comments still get sentiment but not topics.
* The relevance index is a similarity score for the default embedding model - use it as a relative ranking.
* Aspect thresholds are starting values; `python core/nlp_pipeline.py --preview 20` shows real predictions so you can tune `ASPECT_MIN_SIM`.

## Which channels are tracked

`Input Channels.txt` is only a reference list of candidate channels. The channels actually analysed are the ones in `target_channels.json` - keep it to **5-6 channels** per run so ingestion, Whisper and the dashboard stay fast.
