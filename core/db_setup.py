"""Database setup - works on SQLite (default), MySQL, and PostgreSQL.

* creates every table that is missing, upgrades older databases in place (adds new columns)
* NEVER drops or deletes anything
* syncs the channel list from target_channels.json (add / rename / untrack - no data is ever removed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402
from sqlalchemy import (  # noqa: E402
    BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, MetaData, String, Table, Text,
    UniqueConstraint, func, inspect, text,
)
from sqlalchemy.dialects import mysql  # noqa: E402

metadata = MetaData()
LONGTEXT = Text().with_variant(mysql.LONGTEXT(), "mysql")
BIGINT_PK = BigInteger().with_variant(Integer(), "sqlite")        # SQLite only auto-increments INTEGER keys
KW = {"mysql_charset": "utf8mb4"}
NOW = func.current_timestamp()


def _fk(target):
    return ForeignKey(target, ondelete="CASCADE")


Table("channels", metadata,
      Column("channel_id", String(50), primary_key=True),
      Column("channel_name", String(100), nullable=False),
      Column("is_tracked", Boolean, server_default="1"),
      Column("created_at", DateTime, server_default=NOW), **KW)

Table("videos", metadata,
      Column("video_id", String(50), primary_key=True),
      Column("channel_id", String(50), _fk("channels.channel_id")),
      Column("title", String(255), nullable=False),
      Column("description", Text),
      Column("duration_sec", Integer),
      Column("view_count", BigInteger, server_default="0"),
      Column("like_count", BigInteger, server_default="0"),
      Column("comment_count", BigInteger, server_default="0"),
      Column("published_at", DateTime),
      Column("fetched_at", DateTime, server_default=NOW), **KW)

Table("comments", metadata,
      Column("comment_id", String(50), primary_key=True),
      Column("video_id", String(50), _fk("videos.video_id")),
      Column("author_name", String(255)),
      Column("comment_text", Text, nullable=False),
      Column("like_count", Integer, server_default="0"),
      Column("published_at", DateTime),
      Column("is_processed", Boolean, server_default="0"), **KW)

Table("sentiment_insights", metadata,
      Column("insight_id", Integer, primary_key=True, autoincrement=True),
      Column("comment_id", String(50), _fk("comments.comment_id")),
      Column("sentiment_label", String(20)),
      Column("sentiment_score", Float),
      Column("polarity", Float),
      Column("aspect_category", String(50)),
      Column("model_name", String(100)),
      Column("analyzed_at", DateTime, server_default=NOW), **KW)

Table("aspect_sentiments", metadata,
      Column("id", Integer, primary_key=True, autoincrement=True),
      Column("comment_id", String(50), _fk("comments.comment_id"), nullable=False),
      Column("aspect", String(50), nullable=False),
      Column("sentiment_label", String(20)),
      Column("polarity", Float),
      Column("confidence", Float),
      Column("evidence", String(300)),                              # the clause this aspect sentiment is based on
      UniqueConstraint("comment_id", "aspect", name="uq_comment_aspect"), **KW)

Table("transcripts", metadata,
      Column("video_id", String(50), _fk("videos.video_id"), primary_key=True),
      Column("source", String(20), nullable=False),               # whisper | captions | failed
      Column("language", String(12)),
      Column("model_name", String(60)),
      Column("transcript_text", LONGTEXT),
      Column("audio_seconds", Integer),
      Column("word_count", Integer),
      Column("attempts", Integer, server_default="0"),
      Column("last_error", String(255)),
      Column("updated_at", DateTime, server_default=NOW), **KW)

Table("video_audio_assets", metadata,
      Column("video_id", String(50), _fk("videos.video_id"), primary_key=True),
      Column("bucket", String(100), nullable=False),
      Column("object_path", String(500), nullable=False),
      Column("status", String(20), nullable=False, server_default="pending"),
      Column("attempts", Integer, nullable=False, server_default="0"),
      Column("last_error", String(255)),
      Column("claimed_at", DateTime),
      Column("claim_token", String(100)),
      Column("lease_expires_at", DateTime),
      Column("created_at", DateTime, server_default=NOW),
      Column("updated_at", DateTime, server_default=NOW),
      UniqueConstraint("bucket", "object_path", name="uq_video_audio_storage_object"), **KW)

Table("video_stats_history", metadata,
      Column("id", BIGINT_PK, primary_key=True, autoincrement=True),
      Column("video_id", String(50), _fk("videos.video_id"), nullable=False),
      Column("view_count", BigInteger),
      Column("like_count", BigInteger),
      Column("comment_count", BigInteger),
      Column("captured_at", DateTime, server_default=NOW),
      Index("idx_hist_video_time", "video_id", "captured_at"), **KW)

Table("video_relevance", metadata,
      Column("video_id", String(50), _fk("videos.video_id"), primary_key=True),
      Column("relevance_index", Float),
      Column("comment_alignment", Float),
      Column("on_topic_share", Float),
      Column("title_alignment", Float),
      Column("n_comments", Integer),
      Column("text_source", String(20)),
      Column("computed_at", DateTime, server_default=NOW), **KW)

Table("topic_suggestions", metadata,
      Column("id", Integer, primary_key=True, autoincrement=True),
      Column("scope", String(50), nullable=False),
      Column("rank_no", Integer),
      Column("topic", String(120)),
      Column("score", Float),
      Column("mentions", Integer),
      Column("request_count", Integer),
      Column("avg_polarity", Float),
      Column("transcript_coverage", Float),
      Column("evidence", String(500)),
      Column("generated_at", DateTime, server_default=NOW),
      Index("idx_topic_scope", "scope", "rank_no"), **KW)

Table("anomaly_alerts", metadata,
      Column("id", Integer, primary_key=True, autoincrement=True),
      Column("video_id", String(50), _fk("videos.video_id"), nullable=False),
      Column("channel_id", String(50)),
      Column("metric", String(20), nullable=False),               # views | likes | comments
      Column("velocity", Float),
      Column("baseline", Float),
      Column("ratio", Float),
      Column("severity", String(10)),                              # MEDIUM | HIGH | CRITICAL
      Column("detected_at", DateTime, server_default=NOW),
      Column("emailed", Integer, server_default="0"),
      Index("idx_alert_video_metric", "video_id", "metric", "detected_at"), **KW)

Table("channel_run_coverage", metadata,
      Column("channel_id", String(50), _fk("channels.channel_id"), primary_key=True),
      Column("channel_name", String(100), nullable=False),
      Column("target_videos", Integer, server_default="10"),
      Column("selected_videos", Integer, server_default="0"),
      Column("target_comments", Integer, server_default="1000"),
      Column("collected_comments", Integer, server_default="0"),
      Column("transcription_candidates", Integer, server_default="2"),
      Column("completed_transcripts", Integer, server_default="0"),
      Column("video_status", String(50), server_default="Completed"),
      Column("comment_status", String(50), server_default="Completed"),
      Column("sentiment_status", String(50), server_default="Completed"),
      Column("transcription_status", String(50), server_default="Audio Pending"),
      Column("missing_data_reason", String(255)),
      Column("last_run_status", String(50), server_default="success"),
      Column("updated_at", DateTime, server_default=NOW), **KW)

# columns added to databases created by earlier versions of the project: (table, column, DDL)
COLUMN_MIGRATIONS = [
    ("videos", "description", "TEXT"),
    ("videos", "duration_sec", "INTEGER"),
    ("comments", "like_count", "INTEGER DEFAULT 0"),
    ("sentiment_insights", "polarity", "FLOAT"),
    ("sentiment_insights", "model_name", "VARCHAR(100)"),
    ("aspect_sentiments", "evidence", "VARCHAR(300)"),
    ("anomaly_alerts", "severity", "VARCHAR(10)"),
    ("channel_run_coverage", "video_status", "VARCHAR(50) DEFAULT 'Completed'"),
    ("channel_run_coverage", "comment_status", "VARCHAR(50) DEFAULT 'Completed'"),
    ("channel_run_coverage", "sentiment_status", "VARCHAR(50) DEFAULT 'Completed'"),
    ("channel_run_coverage", "transcription_status", "VARCHAR(50) DEFAULT 'Audio Pending'"),
]



def ensure_database() -> None:
    """Create a database only for legacy MySQL configurations.

    SQLite creates its database file automatically.
    PostgreSQL databases must already exist.
    """
    from sqlalchemy.engine import make_url

    url = config.database_url()
    dialect = make_url(url).get_backend_name()

    if dialect in ("sqlite", "postgresql"):
        return

    if dialect != "mysql":
        raise ValueError(f"Unsupported database dialect: {dialect}")

    from sqlalchemy import create_engine

    engine = create_engine(config.database_url(with_database=False))

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS "
                    f"`{config.env('DB_NAME', 'mediaintel_project')}` "
                    f"CHARACTER SET utf8mb4"
                )
            )
    finally:
        engine.dispose()



def _quarantine_incompatible_sqlite() -> str | None:
    """A SQLite file written by a different build of MediaIntel (e.g. the old `Testing` project: its `channels`
    table has no `channel_name`) cannot be upgraded in place. Move it aside as a backup and start clean."""
    url = config.database_url()
    if not url.startswith("sqlite:///"):
        return None
    path = Path(url[len("sqlite:///"):])
    if not path.exists() or path.stat().st_size == 0:
        return None
    import sqlite3
    try:
        with sqlite3.connect(path) as c:
            tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            cols = {r[1] for r in c.execute("PRAGMA table_info(channels)")} if "channels" in tables else set()
    except sqlite3.DatabaseError:
        return None
    if "channels" not in tables or "channel_name" in cols:
        return None
    from datetime import datetime
    backup = path.with_name(f"{path.stem}_legacy_backup_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    for suffix in ("", "-wal", "-shm"):
        src = Path(str(path) + suffix)
        if src.exists():
            src.rename(Path(str(backup) + suffix))
    return backup.name


def ensure_schema(verbose: bool = True) -> list[str]:
    """Creates missing tables, upgrades old ones. Returns a list of what was upgraded."""
    ensure_database()
    moved = _quarantine_incompatible_sqlite()
    if moved:
        print(f"NOTE: the existing database was created by a different version and cannot be upgraded - "
              f"it was kept as data/{moved}; a clean database is being created.")
    engine = config.get_engine()
    metadata.create_all(engine)                        # only creates tables that do not exist yet
    upgraded: list[str] = []
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, column, ddl in COLUMN_MIGRATIONS:
            existing = {c["name"] for c in insp.get_columns(table)}
            if column not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                upgraded.append(f"{table}.{column}")
    if verbose:
        print("Database ready (" + config.db_label() + ")." + (" Upgraded: " + ", ".join(upgraded) if upgraded else ""))
    return upgraded


def sync_channels() -> list[dict]:
    """Applies target_channels.json to the `channels` table. Adds / renames / (un)tracks, never deletes."""
    if config.env("MEDIAINTEL_CHANNEL_SOURCE", "").lower() == "database":
        return []
    path = config.CHANNELS_FILE
    if not path.exists():
        return []
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ERROR: {path.name} is not valid JSON: {exc}")
    rows = []
    for it in items:
        cid = (it.get("channel_id") or "").strip()
        if not cid:
            continue
        rows.append({"channel_id": cid, "channel_name": (it.get("name") or it.get("channel_name") or cid)[:100],
                     "is_tracked": bool(it.get("is_tracked", True))})
    if rows:
        from sqlalchemy import bindparam
        with config.get_engine().begin() as conn:
            conn.execute(config.upsert_sql("channels", ["channel_id", "channel_name", "is_tracked"], ["channel_id"],
                                           ["channel_name", "is_tracked"]), rows)
            # the JSON file is the source of truth for WHAT IS TRACKED: channels missing from it are untracked
            # (hidden from every stage and report) - their stored data is kept, never deleted
            
            gone = conn.execute(
                text(
                    "UPDATE channels SET is_tracked = FALSE "
                    "WHERE is_tracked = TRUE AND channel_id NOT IN :ids"
                ).bindparams(bindparam("ids", expanding=True)),
                {"ids": [r["channel_id"] for r in rows]}
            )

            sync_channels.untracked = gone.rowcount or 0
    return rows


sync_channels.untracked = 0


if __name__ == "__main__":
    ensure_schema()
    synced = sync_channels()
    print(f"Channels synced from {config.CHANNELS_FILE.name}: {len(synced)}")
