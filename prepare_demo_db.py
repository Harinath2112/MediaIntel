
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "data" / "mediaintel.db"
DESTINATION = ROOT / "demo_data" / "mediaintel_demo.db"

if not SOURCE.exists():
    raise FileNotFoundError(
        f"Source database not found: {SOURCE}"
    )

DESTINATION.parent.mkdir(
    parents=True,
    exist_ok=True
)

source = sqlite3.connect(str(SOURCE))
target = sqlite3.connect(str(DESTINATION))
try:
    source.backup(target)
finally:
    source.close()
    target.close()

db = sqlite3.connect(str(DESTINATION))
try:
    cur = db.cursor()
    try:
        cur.execute("PRAGMA integrity_check")
        row = cur.fetchone()
        result = row[0] if row else "failed"

        if result != "ok":
            raise RuntimeError(
                f"Database integrity check failed: {result}"
            )

        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = cur.fetchall()
    finally:
        cur.close()
finally:
    db.close()

print("Demo database created successfully!")
print(f"Location: {DESTINATION}")
print(f"Tables found: {len(tables)}")
print("Integrity check: OK")
print("Review data for private information before publishing.")
