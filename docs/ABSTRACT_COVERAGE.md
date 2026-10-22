# Abstract -> implementation coverage

The abstract is the one in `Abstract.pdf` (Batch 21, A467803). **Status:** *Implemented* = working code for the whole claim.
*Scope decision* = intentionally limited for this version. "Verified" says how it was tested.

| # | Abstract claim | Status | Where | Verified by |
|---|----------------|--------|-------|-------------|
| 1 | Collect competitor **videos and user comments via APIs** | Implemented | `core/youtube_ingest.py` | Pipeline self-test with a stand-in API client. Live YouTube calls were not exercised in the build environment. |
| 2 | ...from **various platforms** / automated extraction methods | **Scope decision: YouTube only** | - | Other platforms are planned for a later version. |
| 3 | **Continuous / real-time** collection | Implemented as one-click, on demand | `run.bat` | One click runs everything; schedule `run.bat --no-dashboard` for periodic runs (see `docs/DEPLOYMENT.md`). |
| 4 | Audio **transcribed with OpenAI Whisper** | Implemented | `core/transcribe_videos.py` | Audio is streamed into memory with ffmpeg and never stored. The in-memory streaming (HTTP, custom headers, webm/m4a, clamping, errors) was tested with real ffmpeg; scheduling, fairness, budget, caching and fallbacks were tested with a stand-in model. **Real Whisper output and the yt-dlp stream lookup were not run in the build environment - run once on your laptop.** |
| 5 | **Transformer (BERT/RoBERTa) NLP on the comments** | Implemented - top-liked comments per video (25 by default, configurable up to all) | `core/nlp_models.py`, `core/nlp_pipeline.py` | Pipeline logic tested with a stand-in model (600 of 600 comments analysed, incremental reruns, no duplicates). Real XLM-RoBERTa inference was not run in the build environment; `python core/nlp_pipeline.py --preview 20` shows real predictions. |
| 6 | **Aspect-based sentiment** (content quality, engagement, relevance, ...) | Implemented - complete | `core/aspects.py`, `core/nlp_pipeline.py`, dashboard + PDF | Comments are split into clauses; each clause gets its own sentiment and one or two aspects; every aspect sentiment stores its evidence clause. Output: aspect distribution, channel x aspect scorecard, evidence explorer. Multi-aspect rules unit-tested; aspect thresholds are starting values to tune with `--preview`. |
| 7 | **Structured relational database** | Implemented | `core/db_setup.py` | SQLite (default) and MySQL, in-place upgrades, safe handling of a foreign database file. |
| 8 | **Streamlit dashboard** for comparative analysis | Implemented | `dashboard/dashboard_app.py` | 8 tabs incl. Whisper Audio Analysis and Channel Comparison; loads without errors under channel / search filters, with alerts present and with no matching data. |
| 9 | **Anomaly detection** on unusual engagement spikes | Implemented - complete | `core/anomaly_alert.py` | Views, likes **and** comments; lifetime and between-runs rates vs the channel baseline; MEDIUM / HIGH / CRITICAL severity; cooldown; live velocity view in the dashboard. Tested on synthetic spikes. |
| 10 | **RPA workflows generate reports and send alerts** | Implemented - complete | `rpa/mediaintel_rpa.robot`, `rpa/MediaIntelRPA.py` | The Robot Framework bot itself generates the strategy report and the PDF, then verifies, archives, e-mails (PDF + alert summary attached) and closes the alert ledger. Tested end to end including a real SMTP delivery to a local test server; Gmail itself not tested. |
| 11 | **End-to-end** solution, reduced manual effort | Implemented | `main_orchestrator.py` | 23-check self-test (`python tests/selftest.py`): full pipeline twice, incremental rerun, every comment analysed, evidence stored, no audio on disk, PDF sections, dashboard. |

## Requested outputs
* **Dashboard section about the audio analysis** - the *Whisper Audio Analysis* tab (coverage, minutes analysed, languages, speaking pace, topics creators talk about, viewer demand vs coverage, transcript viewer).
* **Report section of recommendations from Whisper + comments** - PDF section 4 (Whisper audio analysis), section 6 (topics for the next video) and section 8 (content recommendations).

## Additional features beyond the abstract
Content relevance index, next-video topic ranking, aspect evidence explorer, speed profiles, fair multi-channel Whisper scheduling.

## What is deliberately NOT done
* No data is ever generated or simulated - a missing/invalid API key stops the run with a clear message.
* Audio is never stored. (Only if YouTube refuses direct streaming, a temp file is created and deleted at once; disable with `WHISPER_ALLOW_TEMP_DOWNLOAD=0`.)
* Whisper analyses the first minutes of each video (profile-dependent) so run time stays bounded.
