
from pathlib import Path
import sqlite3
import shutil

SOURCE = Path("demo_data/mediaintel_demo.db")
TARGET = Path("demo_data/mediaintel_public.db")

if not SOURCE.exists():
    raise SystemExit("Source demo database not found.")

if TARGET.exists():
    raise SystemExit(
        "Public database already exists. "
        "Rename or move it before running again."
    )

TARGET.parent.mkdir(parents=True, exist_ok=True)

# Use SQLite's backup API for a consistent copy.
source = sqlite3.connect(
    f"file:{SOURCE.resolve().as_posix()}?mode=ro",
    uri=True
)
target = sqlite3.connect(str(TARGET))

try:
    source.backup(target)
finally:
    source.close()
    target.close()

conn = sqlite3.connect(str(TARGET))

try:
    conn.execute("BEGIN")

    # Anonymize comment authors while preserving comment IDs.
    conn.execute("""
        UPDATE comments
        SET author_name = 'Anonymous Viewer'
    """)

    # Remove raw text that may contain personal information
    # or substantial third-party content.
    conn.execute("""
        UPDATE comments
        SET comment_text = '[Text removed for public demo]'
    """)

    conn.execute("""
        UPDATE transcripts
        SET transcript_text = '[Transcript removed for public demo]',
            last_error = NULL
    """)

    # Remove additional free-text content that may contain
    # identifying information or copied source material.
    conn.execute("""
        UPDATE videos
        SET description = '[Description removed for public demo]'
    """)

    conn.execute("""
        UPDATE topic_suggestions
        SET evidence = '[Evidence removed for public demo]'
    """)

    conn.execute("""
        UPDATE aspect_sentiments
        SET evidence = '[Evidence removed for public demo]'
    """)

    result = conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    if result != "ok":
        raise RuntimeError(f"Integrity check failed: {result}")

    conn.commit()

    print("Public demo database created successfully!")
    print("Location:", TARGET.resolve())
    print("Integrity check:", result)

except Exception:
    conn.rollback()
    conn.close()
    TARGET.unlink(missing_ok=True)
    raise

finally:
    try:
        conn.close()
    except Exception:
        pass
