
"""Pipeline self-test.

    python tests/selftest.py

Runs the COMPLETE 6-stage pipeline twice on a throwaway SQLite database
inside a temporary folder, using lightweight stand-ins for the YouTube API,
Whisper, and AI models (tests/fakes.py).

It verifies:
- All six pipeline stages
- Incremental reruns
- Sentiment coverage
- Reports and email output
- Streamlit dashboard rendering

It does NOT download or evaluate real RoBERTa/Whisper models.
It never touches the project's actual database.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKS: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> None:
    CHECKS.append((ok, label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def run_pipeline(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "main_orchestrator.py"),
            "--no-dashboard",
            "--no-email",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def count(db: Path, sql: str) -> int:
    conn = sqlite3.connect(db, timeout=10)

    try:
        return int(conn.execute(sql).fetchone()[0])
    finally:
        conn.close()


def release_test_database(db: Path) -> None:
    """Checkpoint SQLite and close the connection explicitly."""
    gc.collect()

    if not db.exists():
        return

    conn = sqlite3.connect(db, timeout=5)

    try:
        result = conn.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchone()

        if result is not None and result[0] != 0:
            print(
                "WARNING: SQLite checkpoint is busy; "
                "another connection may still be active."
            )

    finally:
        conn.close()


def cleanup_temp_directory(path: Path) -> None:
    """Retry Windows temporary-directory cleanup on transient locks."""
    for attempt in range(10):
        try:
            if path.exists():
                shutil.rmtree(path)
            return

        except PermissionError:
            if attempt == 9:
                raise

            gc.collect()
            time.sleep(0.5)


def main() -> int:
    CHECKS.clear()

    tmp = Path(tempfile.mkdtemp(prefix="mediaintel_selftest_"))

    try:
        names = [
            "Alpha Coding",
            "Beta Academy",
            "Gamma Tech",
            "Delta Dev",
            "Epsilon CS",
        ]

        channels = [
            {
                "channel_id": f"UCSELFTEST{i}",
                "name": name,
                "is_tracked": True,
            }
            for i, name in enumerate(names)
        ]

        (tmp / "channels.json").write_text(
            json.dumps(channels),
            encoding="utf-8",
        )

        db = tmp / "data" / "mediaintel.db"

        env = dict(
            os.environ,
            MEDIAINTEL_WORKDIR=str(tmp),
            TARGET_CHANNELS_FILE=str(tmp / "channels.json"),
            MEDIAINTEL_DB_URL=f"sqlite:///{db}",
            MEDIAINTEL_TEST_FAKES="1",
            RPA_DRY_RUN="1",
            MAX_COMMENTS_PER_VIDEO="0",
            PYTHONIOENCODING="utf-8",
            COLUMNS="140",
        )

        # --------------------------------------------------
        # RUN 1 - COMPLETE PIPELINE
        # --------------------------------------------------

        print("Run 1 - fresh database, all 6 stages ...")

        r1 = run_pipeline(env)

        check(
            r1.returncode == 0,
            "orchestrator finished with exit code 0",
        )

        check(
            "All 6 stages completed" in r1.stdout,
            "all 6 stages reported success",
        )

        out = tmp / "outputs"

        for filename in (
            "MediaIntel_Dashboard_Documentation.pdf",
            "AlertSummary.txt",
            "Content_Strategy_Report.txt",
        ):
            file_path = out / filename

            check(
                file_path.exists()
                and file_path.stat().st_size > 200,
                f"output created: {filename}",
            )

        outbox = tmp / "data" / "rpa_logs" / "outbox"

        check(
            outbox.exists()
            and any(outbox.glob("*.eml")),
            "RPA bot composed the e-mail (saved in outbox)",
        )

        audio_extensions = {
            ".mp3",
            ".m4a",
            ".wav",
            ".webm",
            ".opus",
            ".ogg",
        }

        audio_files = [
            file
            for file in tmp.rglob("*")
            if file.suffix.lower() in audio_extensions
        ]

        check(
            not audio_files,
            "no audio file was written to disk anywhere",
        )

        pdf_path = out / "MediaIntel_Dashboard_Documentation.pdf"

        if pdf_path.exists():
            pdf = pdf_path.read_bytes()

            pages = (
                pdf.count(b"/Type /Page")
                - pdf.count(b"/Type /Pages")
            )
        else:
            pages = 0

        check(
            pages >= 7,
            f"PDF report has all sections ({pages} pages)",
        )

        # --------------------------------------------------
        # DATABASE VALIDATION
        # --------------------------------------------------

        check(
            count(
                db,
                "SELECT COUNT(*) FROM channels WHERE is_tracked = 1",
            ) == 5,
            "5 channels synced from JSON",
        )

        videos = count(
            db,
            "SELECT COUNT(*) FROM videos",
        )

        check(
            videos >= 15,
            f"videos collected ({videos})",
        )

        check(
            count(db, "SELECT COUNT(*) FROM comments") >= 100,
            "comments collected",
        )

        check(
            count(
                db,
                "SELECT COUNT(*) FROM transcripts "
                "WHERE source = 'whisper'",
            ) >= 5,
            "Whisper transcripts stored",
        )

        comments_n = count(
            db,
            "SELECT COUNT(*) FROM comments",
        )

        analysed_n = count(
            db,
            "SELECT COUNT(DISTINCT comment_id) "
            "FROM sentiment_insights",
        )

        check(
            analysed_n == comments_n,
            f"EVERY collected comment was analysed "
            f"({analysed_n} of {comments_n})",
        )

        check(
            count(
                db,
                "SELECT COUNT(*) FROM aspect_sentiments",
            ) >= 50,
            "aspect-level sentiment rows stored",
        )

        aspect_total = count(
            db,
            "SELECT COUNT(*) FROM aspect_sentiments",
        )

        aspect_evidence = count(
            db,
            "SELECT COUNT(*) FROM aspect_sentiments "
            "WHERE evidence IS NOT NULL AND evidence <> ''",
        )

        check(
            aspect_evidence == aspect_total,
            "every aspect sentiment keeps its evidence clause",
        )

        check(
            count(
                db,
                "SELECT COUNT(*) FROM transcripts "
                "WHERE audio_seconds > 0 "
                "AND word_count > 0 "
                "AND source = 'whisper'",
            ) >= 5,
            "Whisper analysis stored minutes of audio + word counts",
        )

        check(
            count(
                db,
                "SELECT COUNT(*) FROM video_relevance",
            ) >= 5,
            "relevance scores stored",
        )

        check(
            count(
                db,
                "SELECT COUNT(*) FROM topic_suggestions "
                "WHERE scope = 'ALL'",
            ) >= 3,
            "next-video topics stored",
        )

        check(
            count(
                db,
                "SELECT COUNT(*) FROM video_stats_history",
            ) >= videos,
            "statistics history snapshots stored",
        )

        # --------------------------------------------------
        # RUN 2 - INCREMENTAL PIPELINE
        # --------------------------------------------------

        print("Run 2 - rerun must be incremental ...")

        r2 = run_pipeline(env)

        check(
            r2.returncode == 0,
            "second run finished with exit code 0",
        )

        check(
            count(
                db,
                "SELECT COUNT(*) FROM videos",
            ) == videos,
            "no duplicate videos",
        )

        total_sentiments = count(
            db,
            "SELECT COUNT(*) FROM sentiment_insights",
        )

        distinct_sentiments = count(
            db,
            "SELECT COUNT(DISTINCT comment_id) "
            "FROM sentiment_insights",
        )

        check(
            total_sentiments == distinct_sentiments,
            "no duplicate sentiment rows",
        )

        # --------------------------------------------------
        # DASHBOARD VALIDATION
        # --------------------------------------------------

        # Insert a synthetic alert for dashboard testing.

        conn = sqlite3.connect(db, timeout=10)

        try:
            vid, ch = conn.execute(
                "SELECT video_id, channel_id "
                "FROM videos LIMIT 1"
            ).fetchone()

            conn.execute(
                """
                INSERT INTO anomaly_alerts
                (
                    video_id,
                    channel_id,
                    metric,
                    velocity,
                    baseline,
                    ratio,
                    severity,
                    detected_at,
                    emailed
                )
                VALUES
                (
                    ?, ?, 'views', 1200, 100, 12,
                    'CRITICAL', datetime('now'), 0
                )
                """,
                (vid, ch),
            )

            conn.commit()

        finally:
            conn.close()

        print("Dashboard - loads without errors ...")

        probe = (
            "from streamlit.testing.v1 import AppTest\n"
            f"at = AppTest.from_file("
            f"r'{ROOT / 'dashboard' / 'dashboard_app.py'}', "
            "default_timeout=120)\n"
            "at.run()\n"
            "print('EXC', len(at.exception), "
            "'TABS', len(at.tabs))\n"
            "css = [m.value for m in at.markdown "
            "if '<style>' in m.value]\n"
            "print('CSS_OK', bool(css) and "
            "all('{{' not in c and '}}' not in c "
            "for c in css))\n"
        )

        r3 = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        check(
            r3.returncode == 0
            and "EXC 0" in r3.stdout
            and "TABS 8" in r3.stdout,
            "dashboard renders 8 tabs with an alert present, no exception",
        )

        check(
            "CSS_OK True" in r3.stdout,
            "dashboard theme CSS is rendered correctly "
            "(no stray doubled braces)",
        )

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        failed = [
            label
            for ok, label in CHECKS
            if not ok
        ]

        passed = len(CHECKS) - len(failed)

        print(
            f"\n{passed}/{len(CHECKS)} checks passed."
        )

        if failed:
            print(
                "FAILED:",
                *failed,
                sep="\n  - ",
            )

            print(
                "\n--- last orchestrator output ---\n"
                + r2.stdout[-1500:]
            )

        return 1 if failed else 0

    finally:
        # Close any connections owned by this process.
        gc.collect()

        if db.exists():
            release_test_database(db)

        cleanup_temp_directory(tmp)


if __name__ == "__main__":
    sys.exit(main())
