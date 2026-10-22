
import sqlite3
import re
from pathlib import Path

db_path = Path("demo_data/mediaintel_public.db")

if not db_path.is_file():
    raise SystemExit("Demo database not found!")

patterns = {
    "Google API key": r"AIza[0-9A-Za-z_-]{35}",
    "Private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "Credential assignment": (
        r"(?i)\b(?:password|api[_-]?key|client[_-]?secret)"
        r"\s*[:=]\s*[^\s,;]{6,}"
    ),
}

connection = sqlite3.connect(
    f"file:{db_path.as_posix()}?mode=ro",
    uri=True
)

total_matches = 0

try:
    tables = connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()

    for (table,) in tables:
        quoted_table = '"' + table.replace('"', '""') + '"'

        cursor = connection.execute(
            f"SELECT * FROM {quoted_table}"
        )

        columns = [column[0] for column in cursor.description]
        matches = {}

        for row in cursor:
            for column, value in zip(columns, row):
                if not isinstance(value, str):
                    continue

                for label, pattern in patterns.items():
                    if re.search(pattern, value):
                        key = (column, label)
                        matches[key] = matches.get(key, 0) + 1
                        total_matches += 1

        if matches:
            print(f"\nTABLE: {table}")
            for (column, label), count in matches.items():
                print(f"  {column}: {count} {label} matches")

    print("\nSecurity scan complete.")
    print("Total potential matches:", total_matches)

finally:
    connection.close()
