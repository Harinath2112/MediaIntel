"""Shared configuration, paths and database access for every MediaIntel stage.

* Reads <project root>/.env once, so every script works no matter which folder it is started from.
* Database: SQLite by default (zero configuration, runs on any laptop). MySQL is supported too -
  set DATABASE_URL or DB_TYPE=mysql with DB_HOST / DB_USER / DB_PASSWORD / DB_NAME.
* Speed profiles (SPEED_PROFILE=fast | balanced | accurate) pick sensible defaults for the heavy AI stages.
"""
from __future__ import annotations

import atexit
import os
import sqlite3
import sys
import urllib.parse
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Register SQLite datetime adapters to eliminate Python 3.12+ DeprecationWarnings
try:
    sqlite3.register_adapter(datetime, lambda dt: dt.isoformat(" "))
    sqlite3.register_adapter(date, lambda d: d.isoformat())
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# make "import core..." work from any script location
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# paths   (MEDIAINTEL_WORKDIR lets the self-test write somewhere disposable)
# ---------------------------------------------------------------------------
WORK = Path(os.getenv("MEDIAINTEL_WORKDIR") or ROOT)
OUTPUTS_DIR = WORK / "outputs"
DATA_DIR = WORK / "data"
LOG_DIR = DATA_DIR / "logs"
RPA_LOG_DIR = DATA_DIR / "rpa_logs"
CHANNELS_FILE = Path(os.getenv("TARGET_CHANNELS_FILE") or (ROOT / "target_channels.json"))
for _d in (OUTPUTS_DIR, DATA_DIR, LOG_DIR, RPA_LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# environment helpers
# ---------------------------------------------------------------------------
def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().strip('"').strip("'")


def env_int(name: str, default: int) -> int:
    try:
        return int(env(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool = False) -> bool:
    value = env(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y", "on")


def utcnow() -> datetime:
    """Naive UTC 'now' - YouTube timestamps are UTC, so every comparison must be UTC too."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# speed profiles: one switch trades speed for fidelity. Any single value can still be overridden in .env
# ---------------------------------------------------------------------------
# MAX_COMMENTS_PER_VIDEO = 0 means "analyse ALL collected comments" (profile default); N keeps the N most-liked per video.
PROFILES = {
    "fast": dict(WHISPER_MODEL="tiny", WHISPER_MAX_AUDIO_MIN=2, WHISPER_VIDEOS_PER_CHANNEL=3, WHISPER_BUDGET_MIN=6,
                 MAX_COMMENTS_PER_VIDEO=0, SENTIMENT_MAX_TOKENS=64, VIDEOS_PER_CHANNEL=10),
    "balanced": dict(WHISPER_MODEL="base", WHISPER_MAX_AUDIO_MIN=4, WHISPER_VIDEOS_PER_CHANNEL=4, WHISPER_BUDGET_MIN=12,
                     MAX_COMMENTS_PER_VIDEO=0, SENTIMENT_MAX_TOKENS=96, VIDEOS_PER_CHANNEL=20),
    "accurate": dict(WHISPER_MODEL="small", WHISPER_MAX_AUDIO_MIN=10, WHISPER_VIDEOS_PER_CHANNEL=10, WHISPER_BUDGET_MIN=40,
                     MAX_COMMENTS_PER_VIDEO=0, SENTIMENT_MAX_TOKENS=128, VIDEOS_PER_CHANNEL=20),
}

# setting names used by older versions / other builds of this project -> current names
ALIASES = {"WHISPER_MODEL_SIZE": "WHISPER_MODEL", "SMTP_HOST": "SMTP_SERVER", "SMTP_USER": "SENDER_EMAIL",
           "SMTP_PASSWORD": "SENDER_PASSWORD", "ALERT_RECIPIENT": "RECIPIENT_EMAIL"}
for _old, _new in ALIASES.items():
    if os.getenv(_old) and not os.getenv(_new):
        os.environ[_new] = os.getenv(_old)
# VARIANCE_THRESHOLD_PCT=200 (+200 %) is the same rule as ANOMALY_THRESHOLD=3 (3x)
if os.getenv("VARIANCE_THRESHOLD_PCT") and not os.getenv("ANOMALY_THRESHOLD"):
    try:
        os.environ["ANOMALY_THRESHOLD"] = str(1 + float(os.getenv("VARIANCE_THRESHOLD_PCT")) / 100)
    except ValueError:
        pass
PROFILE = (env("SPEED_PROFILE", "balanced") or "balanced").lower()
if PROFILE not in PROFILES:
    PROFILE = "balanced"


def setting(name: str):
    """.env value if present, otherwise the active profile's default."""
    value = env(name)
    return PROFILES[PROFILE][name] if value is None else value


def setting_int(name: str) -> int:
    try:
        return int(float(setting(name)))
    except (TypeError, ValueError):
        return int(PROFILES["balanced"][name])


def setting_float(name: str) -> float:
    try:
        return float(setting(name))
    except (TypeError, ValueError):
        return float(PROFILES["balanced"][name])


# ---------------------------------------------------------------------------
# e-mail settings (accepts both naming styles used across versions of this project)
# ---------------------------------------------------------------------------
def mail_settings() -> dict:
    return {
        "host": env("SMTP_SERVER") or env("SMTP_HOST") or "smtp.gmail.com",
        "port": env_int("SMTP_PORT", 587),
        "sender": env("SENDER_EMAIL") or env("SMTP_USER"),
        "password": env("SENDER_PASSWORD") or env("SMTP_PASSWORD") or "",
        "recipients": [r.strip() for r in (env("RECIPIENT_EMAIL") or env("ALERT_RECIPIENT") or "").split(",") if r.strip()],
    }


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------
def database_url(with_database: bool = True) -> str:
    """DATABASE_URL > DB_TYPE > legacy MySQL variables > SQLite default."""
    explicit = env("MEDIAINTEL_DB_URL") or env("DATABASE_URL")

    if not explicit:
        try:
            import streamlit as st
            explicit = (
                st.secrets.get("MEDIAINTEL_DB_URL")
                or st.secrets.get("DATABASE_URL")
                or (st.secrets.get("postgres", {}) or {}).get("url")
                or (st.secrets.get("connections", {}) or {}).get("supabase", {}).get("url")
            )
        except Exception:
            pass
    if explicit and with_database:
        if explicit.startswith("sqlite:///") and not explicit.startswith("sqlite:////") and ":" not in explicit[10:]:
            return "sqlite:///" + str((WORK / explicit[10:]).resolve())      # relative path -> inside the project

        # Normalize postgres / postgresql to postgresql+psycopg if psycopg is installed
        if explicit.startswith("postgres://"):
            explicit = "postgresql://" + explicit[11:]
        if explicit.startswith("postgresql://") and not explicit.startswith("postgresql+"):
            try:
                import psycopg  # noqa: F401
                explicit = "postgresql+psycopg://" + explicit[13:]
            except Exception:
                try:
                    import psycopg2  # noqa: F401
                    explicit = "postgresql+psycopg2://" + explicit[13:]
                except Exception:
                    pass

        return explicit
    db_type = (env("DB_TYPE") or ("mysql" if any(os.getenv(k) for k in ("DB_HOST", "DB_USER", "DB_NAME")) else "sqlite")).lower()
    if db_type == "mysql":
        pwd = urllib.parse.quote_plus(os.getenv("DB_PASSWORD", ""))
        base = f"mysql+pymysql://{env('DB_USER', 'root')}:{pwd}@{env('DB_HOST', 'localhost')}:{env('DB_PORT', '3306')}"
        return f"{base}/{env('DB_NAME', 'mediaintel_project')}?charset=utf8mb4" if with_database else f"{base}/?charset=utf8mb4"
    return "sqlite:///" + str(DATA_DIR / "mediaintel.db")


def db_label() -> str:
    """Return a readable label for the configured database."""
    from sqlalchemy.engine import make_url

    url = make_url(database_url())
    dialect = url.get_backend_name()

    if dialect == "sqlite":
        return f"SQLite  {url.database or ':memory:'}"

    if dialect == "postgresql":
        return f"PostgreSQL  {url.database or 'unknown'}"

    if dialect == "mysql":
        return f"MySQL  {url.database or env('DB_NAME', 'mediaintel_project')}"

    return f"{dialect}  {url.database or 'unknown'}"

_engine = None


def get_engine():
    """One shared SQLAlchemy engine per process."""
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine, event
        url = database_url()
        if url.startswith("sqlite"):
            _engine = create_engine(url, connect_args={"timeout": 60})

            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):      # foreign keys + concurrent readers/writers
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=60000")
                cur.close()
        elif "postgresql" in url:
            _engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_recycle=1800,
                connect_args={"sslmode": "require", "connect_timeout": 15},
            )
        else:
            _engine = create_engine(url, pool_pre_ping=True, pool_recycle=1800)
    return _engine


def dispose_engine() -> None:
    """Explicitly dispose the shared SQLAlchemy engine, closing all pooled connections cleanly."""
    global _engine
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            pass
        _engine = None


# Ensure any open connection pool is cleanly closed before process termination
atexit.register(dispose_engine)


def is_sqlite() -> bool:
    return get_engine().dialect.name == "sqlite"


def read_sql(query, params: dict | None = None):
    """Run a SELECT and return a DataFrame. Timestamp columns (*_at) always come back as datetimes."""
    import pandas as pd
    from sqlalchemy import text
    with get_engine().connect() as conn:
        stmt = text(query) if isinstance(query, str) else query
        df = pd.read_sql(stmt, conn, params=params or {})
    for col in df.columns:
        if str(col).endswith("_at") and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")      # SQLite hands back text; MySQL already datetimes
    return df


def execute(query, params: dict | list | None = None) -> int:
    """Run a write statement (or executemany when params is a list). Returns affected rows."""
    from sqlalchemy import text
    with get_engine().begin() as conn:
        stmt = text(query) if isinstance(query, str) else query
        result = conn.execute(stmt, params or {})
        return result.rowcount



def upsert_sql(table: str, columns: list[str], keys: list[str], update: list[str] | None = None):
    """Generate database-specific UPSERT SQL for SQLite, PostgreSQL, and MySQL."""
    from sqlalchemy import text

    update = [c for c in columns if c not in keys] if update is None else update
    cols = ", ".join(columns)
    vals = ", ".join(f":{c}" for c in columns)
    dialect = get_engine().dialect.name

    if dialect in ("sqlite", "postgresql"):
        conflict_keys = ", ".join(keys)

        if update:
            assignments = ", ".join(
                f"{c} = excluded.{c}" for c in update
            )
            tail = (
                f"ON CONFLICT ({conflict_keys}) "
                f"DO UPDATE SET {assignments}"
            )
        else:
            tail = f"ON CONFLICT ({conflict_keys}) DO NOTHING"

        return text(
            f"INSERT INTO {table} ({cols}) VALUES ({vals}) {tail}"
        )

    if dialect == "mysql":
        if update:
            assignments = ", ".join(
                f"{c} = VALUES({c})" for c in update
            )
            return text(
                f"INSERT INTO {table} ({cols}) VALUES ({vals}) "
                f"ON DUPLICATE KEY UPDATE {assignments}"
            )

        return text(
            f"INSERT IGNORE INTO {table} ({cols}) VALUES ({vals})"
        )

    raise ValueError(f"Unsupported database dialect: {dialect}")


def diagnose_database_connection(url_str: str | None = None) -> dict:
    """Non-secret-revealing database connectivity preflight and actionable diagnostics."""
    from sqlalchemy.engine import make_url
    from sqlalchemy import create_engine, text

    raw_url = url_str or database_url()
    try:
        url_obj = make_url(raw_url)
    except Exception as exc:
        return {
            "healthy": False,
            "error_type": "URL_PARSE_ERROR",
            "message": f"Failed to parse database URL: {exc}",
            "recommendation": "Check for unencoded characters (@, #, :, /, %) in the database password. Special characters must be URL percent-encoded.",
        }

    dialect = url_obj.get_backend_name()
    host = url_obj.host or "local"
    port = url_obj.port
    username = url_obj.username or ""
    database_name = url_obj.database or ""

    diag = {
        "healthy": False,
        "backend": dialect,
        "host": host,
        "port": port,
        "database": database_name,
        "user_masked": username[:3] + "***" if len(username) > 3 else ("***" if username else "none"),
        "error_type": None,
        "message": None,
        "recommendation": None,
    }

    # Supabase Connection Pooler tenant check
    if host and ("pooler.supabase.com" in host or port == 6543):
        if username and ("." not in username and "[" not in username):
            diag["recommendation"] = (
                "Actionable Check: Connecting to Supabase Connection Pooler (port 6543 / pooler.supabase.com) "
                "requires username format 'postgres.<project-ref>'. Connecting with plain 'postgres' causes "
                "'FATAL: password authentication failed for user postgres'. Verify MEDIAINTEL_DB_URL."
            )

    test_engine = None
    try:
        if dialect == "sqlite":
            test_engine = create_engine(raw_url, connect_args={"timeout": 10})
        elif "postgresql" in dialect:
            test_engine = create_engine(raw_url, connect_args={"sslmode": "require", "connect_timeout": 10})
        else:
            test_engine = create_engine(raw_url)

        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        diag["healthy"] = True
        diag["message"] = f"Successfully connected to {dialect} database ({database_name})."
        return diag
    except Exception as exc:
        err_msg = str(exc)
        if "password authentication failed" in err_msg.lower():
            diag["error_type"] = "AUTHENTICATION_FAILED"
            diag["message"] = "Password authentication failed."
            if not diag["recommendation"]:
                diag["recommendation"] = (
                    "Verify MEDIAINTEL_DB_URL secret. If using Supabase Connection Pooler, "
                    "ensure username is 'postgres.<project-ref>' and password special characters are URL-encoded."
                )
        elif "could not translate host name" in err_msg.lower() or "name or service not known" in err_msg.lower():
            diag["error_type"] = "DNS_RESOLUTION_FAILED"
            diag["message"] = f"Host name '{host}' could not be resolved."
            if not diag["recommendation"]:
                diag["recommendation"] = "Verify the database host in MEDIAINTEL_DB_URL."
        elif "connection timed out" in err_msg.lower() or "timeout" in err_msg.lower():
            diag["error_type"] = "CONNECTION_TIMEOUT"
            diag["message"] = "Database connection timed out."
            if not diag["recommendation"]:
                diag["recommendation"] = "Verify network connectivity, firewall, or ensure Supabase project is not paused."
        elif "ssl" in err_msg.lower():
            diag["error_type"] = "SSL_ERROR"
            diag["message"] = "SSL negotiation failed."
            if not diag["recommendation"]:
                diag["recommendation"] = "Ensure 'sslmode=require' is configured in PostgreSQL connection."
        else:
            diag["error_type"] = "OPERATIONAL_ERROR"
            diag["message"] = f"Database connection failed: {err_msg[:120]}"
            if not diag["recommendation"]:
                diag["recommendation"] = "Verify MEDIAINTEL_DB_URL connection parameters."
        return diag
    finally:
        if test_engine is not None:
            try:
                test_engine.dispose()
            except Exception:
                pass


def verify_database_preflight() -> bool:
    """Run database connectivity preflight and print safe diagnostic output."""
    res = diagnose_database_connection()
    print("\n--- MediaIntel Database Preflight Diagnosis ---")
    print(f"Backend       : {res.get('backend')}")
    print(f"Host          : {res.get('host')}")
    print(f"Port          : {res.get('port')}")
    print(f"Database      : {res.get('database')}")
    print(f"User          : {res.get('user_masked')}")
    print(f"Status        : {'CONNECTED' if res['healthy'] else 'FAILED'}")
    if res["healthy"]:
        print(f"Notice        : {res['message']}\n")
        return True
    else:
        print(f"Error Type    : {res.get('error_type')}")
        print(f"Details       : {res.get('message')}")
        if res.get('recommendation'):
            print(f"Action Advice : {res.get('recommendation')}")
        print("\n")
        return False


if __name__ == "__main__":
    ok = verify_database_preflight()
    sys.exit(0 if ok else 1)

