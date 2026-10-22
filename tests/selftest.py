"""Pipeline self-test.

    python tests/selftest.py

Runs the COMPLETE 6-stage pipeline twice on a throwaway SQLite database inside a temp folder, using lightweight
stand-ins for the YouTube API, Whisper and the AI models (tests/fakes.py). It proves that every stage is wired
together correctly, that reruns are incremental and that the PDF / alert / e-mail outputs are produced.
It does NOT download or evaluate the real RoBERTa / Whisper models, and it never touches your own data.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> None:
    CHECKS.append((ok, label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def run_pipeline(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "main_orchestrator.py"), "--no-dashboard", "--no-email"],
                          cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")


def count(db: Path, sql: str) -> int:
    with sqlite3.connect(db) as c:
        return int(c.execute(sql).fetchone()[0])


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mediaintel_selftest_") as tmp_s:
        tmp = Path(tmp_s)
        names = ["Alpha Coding", "Beta Academy", "Gamma Tech", "Delta Dev", "Epsilon CS"]
        (tmp / "channels.json").write_text(json.dumps(
            [{"channel_id": f"UCSELFTEST{i}", "name": n, "is_tracked": True} for i, n in enumerate(names)]))
        db = tmp / "data" / "mediaintel.db"
        env = dict(os.environ, MEDIAINTEL_WORKDIR=str(tmp), TARGET_CHANNELS_FILE=str(tmp / "channels.json"),
                   MEDIAINTEL_DB_URL=f"sqlite:///{db}", MEDIAINTEL_TEST_FAKES="1", RPA_DRY_RUN="1",
                   PYTHONIOENCODING="utf-8", COLUMNS="140")

        print("Run 1 - fresh database, all 6 stages ...")
        r1 = run_pipeline(env)
        check(r1.returncode == 0, "orchestrator finished with exit code 0")
        check("All 6 stages completed" in r1.stdout, "all 6 stages reported success")
        out = tmp / "outputs"
        for f in ("MediaIntel_Dashboard_Documentation.pdf", "AlertSummary.txt", "Content_Strategy_Report.txt"):
            check((out / f).exists() and (out / f).stat().st_size > 200, f"output created: {f}")
        check(any((tmp / "data" / "rpa_logs" / "outbox").glob("*.eml")), "RPA bot composed the e-mail (saved in outbox)")
        check(not [f for f in tmp.rglob("*") if f.suffix.lower() in (".mp3", ".m4a", ".wav", ".webm", ".opus", ".ogg")],
              "no audio file was written to disk anywhere")
        pdf = (out / "MediaIntel_Dashboard_Documentation.pdf").read_bytes()
        pages = pdf.count(b"/Type /Page") - pdf.count(b"/Type /Pages")
        check(pages >= 7, f"PDF report has all sections ({pages} pages)")
        check(count(db, "SELECT COUNT(*) FROM channels WHERE is_tracked = 1") == 5, "5 channels synced from JSON")
        videos = count(db, "SELECT COUNT(*) FROM videos")
        check(videos >= 15, f"videos collected ({videos})")
        check(count(db, "SELECT COUNT(*) FROM comments") >= 100, "comments collected")
        check(count(db, "SELECT COUNT(*) FROM transcripts WHERE source = 'whisper'") >= 5, "Whisper transcripts stored")
        comments_n = count(db, "SELECT COUNT(*) FROM comments")
        check(count(db, "SELECT COUNT(DISTINCT comment_id) FROM sentiment_insights") == comments_n,
              f"EVERY collected comment was analysed ({comments_n} of {comments_n})")
        check(count(db, "SELECT COUNT(*) FROM aspect_sentiments") >= 50, "aspect-level sentiment rows stored")
        check(count(db, "SELECT COUNT(*) FROM aspect_sentiments WHERE evidence IS NOT NULL AND evidence <> ''") ==
              count(db, "SELECT COUNT(*) FROM aspect_sentiments"), "every aspect sentiment keeps its evidence clause")
        check(count(db, "SELECT COUNT(*) FROM transcripts WHERE audio_seconds > 0 AND word_count > 0 AND source = 'whisper'") >= 5,
              "Whisper analysis stored minutes of audio + word counts")
        check(count(db, "SELECT COUNT(*) FROM video_relevance") >= 5, "relevance scores stored")
        check(count(db, "SELECT COUNT(*) FROM topic_suggestions WHERE scope = 'ALL'") >= 3, "next-video topics stored")
        check(count(db, "SELECT COUNT(*) FROM video_stats_history") >= videos, "statistics history snapshots stored")

        print("Run 2 - rerun must be incremental ...")
        r2 = run_pipeline(env)
        check(r2.returncode == 0, "second run finished with exit code 0")
        check(count(db, "SELECT COUNT(*) FROM videos") == videos, "no duplicate videos")
        check(count(db, "SELECT COUNT(*) FROM sentiment_insights") == count(db, "SELECT COUNT(DISTINCT comment_id) FROM sentiment_insights"),
              "no duplicate sentiment rows")

        # give the Alerts tab something to render (the pipeline's synthetic data has no real spike)
        with sqlite3.connect(db) as c:
            vid, ch = c.execute("SELECT video_id, channel_id FROM videos LIMIT 1").fetchone()
            c.execute("INSERT INTO anomaly_alerts (video_id, channel_id, metric, velocity, baseline, ratio, severity, detected_at, emailed) "
                      "VALUES (?, ?, 'views', 1200, 100, 12, 'CRITICAL', datetime('now'), 0)", (vid, ch))

        print("Dashboard - loads without errors ...")
        probe = (
            "from streamlit.testing.v1 import AppTest\n"
            f"at = AppTest.from_file(r'{ROOT / 'dashboard' / 'dashboard_app.py'}', default_timeout=120)\n"
            "at.run()\nprint('EXC', len(at.exception), 'TABS', len(at.tabs))\n"
            "css = [m.value for m in at.markdown if '<style>' in m.value]\n"
            "print('CSS_OK', bool(css) and all('{{' not in c and '}}' not in c for c in css))\n")
        r3 = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env, capture_output=True, text=True)
        check("EXC 0" in r3.stdout and "TABS 8" in r3.stdout, "dashboard renders 8 tabs with an alert present, no exception")
        check("CSS_OK True" in r3.stdout, "dashboard theme CSS is rendered correctly (no stray doubled braces)")

    failed = [label for ok, label in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed.")
    if failed:
        print("FAILED:", *failed, sep="\n  - ")
        print("\n--- last orchestrator output ---\n" + (r2.stdout if 'r2' in dir() else r1.stdout)[-1500:])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
