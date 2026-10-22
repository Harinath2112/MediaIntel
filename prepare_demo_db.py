
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

with sqlite3.connect(str(SOURCE)) as source:
    with sqlite3.connect(str(DESTINATION)) as target:
        source.backup(target)

with sqlite3.connect(str(DESTINATION)) as db:
    result = db.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    if result != "ok":
        raise RuntimeError(
            f"Database integrity check failed: {result}"
        )

    tables = db.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()

print("Demo database created successfully!")
print(f"Location: {DESTINATION}")
print(f"Tables found: {len(tables)}")
print("Integrity check: OK")
print("Review data for private information before publishing.")
