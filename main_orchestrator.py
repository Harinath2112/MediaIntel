"""
MediaIntel - one-click orchestrator.

    python main_orchestrator.py                 full pipeline (6 stages), then opens the dashboard
    python main_orchestrator.py --quick         re-analyse existing data only (anomalies + RPA reports/e-mail)
    python main_orchestrator.py --no-dashboard  do not open Streamlit at the end
    python main_orchestrator.py --no-email      RPA bot saves the e-mail instead of sending it

Stages
    1 YouTube Ingestion       competitor videos, statistics, comments          (YouTube Data API)
    2 Whisper Audio Analysis  audio streamed into memory -> text               (OpenAI Whisper, nothing stored)
    3 RoBERTa Sentiment       aspect-based sentiment of top-liked comments  (XLM-RoBERTa)
    4 Relevance & Topics      transcript <-> comment relevance, next-video topics
    5 Anomaly Detection       spikes in views / likes / comments, with severity
    6 RPA Reports & Alerts    Robot Framework bot: generates strategy + PDF reports, e-mails them with the alerts

Every script's detailed output goes to data/logs/pipeline.log; the console only shows progress.
Speed: SPEED_PROFILE=fast | balanced | accurate in .env (default balanced).
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from core import config  # noqa: E402  (loads .env, defines every path)

LOG_DIR = config.LOG_DIR
LOG_FILE = LOG_DIR / "pipeline.log"
RPA_LOG_DIR = config.RPA_LOG_DIR
OUTPUTS_DIR = config.OUTPUTS_DIR


def _need(module: str, pip_name: str | None = None):
    try:
        __import__(module)
    except ImportError:
        print(f"ERROR: Python package '{pip_name or module}' is missing.\n"
              f"       Run  run.bat  (or: pip install -r requirements-local.txt)")
        sys.exit(1)


_need("rich")
from rich.console import Console  # noqa: E402
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn  # noqa: E402
from rich.table import Column  # noqa: E402


# ---------------------------------------------------------------------------
# Live progress parsers (return 0-100 or None) and result summarisers (return (text, ok))
# ---------------------------------------------------------------------------
def _p_nlp(line, ctx):
    m = re.search(r"analysed (\d+)/(\d+)", line)
    return min(99, 100 * int(m[1]) / max(int(m[2]), 1)) if m else None


def _p_whisper(line, ctx):
    ctx.setdefault("t0", time.time())
    m = re.search(r"(\d+) video\(s\) need transcription \(budget (\d+) min", line)
    if m:
        ctx["n"], ctx["budget"] = int(m[1]), max(int(m[2]), 1)
    if re.match(r"\s+(Whisper|skipped)\s", line):
        ctx["done"] = ctx.get("done", 0) + 1
    if "n" in ctx:
        by_count = ctx.get("done", 0) / max(ctx["n"], 1)
        by_time = (time.time() - ctx["t0"]) / (ctx["budget"] * 60)
        return min(97, 100 * max(by_count, min(by_time, 0.97)))
    return None


def _r_ingest(out):
    m = re.search(r"(\d+) videos, (\d+) comments synced( \((\d+) channel\(s\) failed\))?", out)
    if not m:
        return "Data synced", True
    if m[4]:
        return f"{m[1]} videos, {m[2]} comments ({m[4]} channel(s) failed)", False
    return f"{m[1]} videos, {m[2]} comments", True


def _r_whisper(out):
    if "nothing to do" in out:
        return "All videos transcribed", True
    m = re.search(r"(\d+) via Whisper, (\d+) via captions, (\d+) without", out)
    if m:
        txt = f"{m[1]} Whisper, {m[2]} captions"
        return (txt + f", {m[3]} unavailable", False) if int(m[3]) else (txt, True)
    return "Done", True


def _r_nlp(out):
    if "up to date" in out:
        return "Already up to date", True
    m = re.search(r"stored (\d+) comments", out)
    return (f"{m[1]} comments analysed", True) if m else ("Comments analysed", True)


def _r_topics(out):
    m = re.search(r"1\. (.+?)  \(score", out)
    return (f"Top topic: {m[1]}", True) if m else ("Relevance computed", True)


def _r_anomaly(out):
    n = out.count("ANOMALY DETECTED")
    if n:
        return f"{n} new anomaly signal(s)", True
    if "already reported" in out:
        return "Known anomalies only", True
    return "No anomalies", True


def _r_rpa(out):
    import json
    f = RPA_LOG_DIR / "dispatch_result.json"
    try:
        res = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return "Bot finished", True
    n = res.get("alerts", 0)
    return {
        "SENT": (f"E-mail sent ({n} alert{'s' if n != 1 else ''})", True),
        "DRY_RUN": ("E-mail saved to outbox (no send)", True),
        "SKIPPED": ("No anomalies - e-mail skipped", True),
    }.get(res.get("status"), ("Bot finished", True))


PY = sys.executable
ROBOT = [PY, "-m", "robot", "--rpa", "--outputdir", str(RPA_LOG_DIR), "--pythonpath", "rpa", "--output", "rpa_output.xml",
         "--log", "rpa_log.html", "--report", "rpa_report.html", "--consolewidth", "100", "rpa/mediaintel_rpa.robot"]

# (name, command, typical seconds - paces the bar when no live parser, live parser, result summariser, quick-mode?)
STAGES = [
    ("YouTube Ingestion",     [PY, "core/youtube_ingest.py"],          60, None,      _r_ingest,  False),
    ("Whisper Audio Analysis", [PY, "core/transcribe_videos.py"],     600, _p_whisper, _r_whisper, False),
    ("RoBERTa Sentiment",     [PY, "core/nlp_pipeline.py"],           240, _p_nlp,     _r_nlp,     False),
    ("Relevance & Topics",    [PY, "core/relevance_topics.py"],        60, None,      _r_topics,  False),
    ("Anomaly Detection",     [PY, "core/anomaly_alert.py"],            6, None,      _r_anomaly, True),
    ("RPA Reports & Alerts",  ROBOT,                                    30, None,      _r_rpa,     True),
]


# ---------------------------------------------------------------------------
# Preflight: friendly, fast checks so a one-click run never dies with a stack trace
# ---------------------------------------------------------------------------
def preflight(console, quick: bool) -> bool:
    ok = True
    offline_test = config.env_bool("MEDIAINTEL_TEST_FAKES")

    def good(msg): console.print(f"  [green]✔[/] {msg}")
    def warn(msg): console.print(f"  [yellow]⚠[/] {msg}")
    def bad(msg):
        nonlocal ok
        ok = False
        console.print(f"  [red]✖[/] {msg}")

    if sys.version_info < (3, 10):
        bad(f"Python {sys.version.split()[0]} is too old - install Python 3.12.")
    else:
        good(f"Python {sys.version.split()[0]}   ·   speed profile: {config.PROFILE}")

    if (ROOT / ".env").exists():
        good(".env loaded")
    elif not offline_test:
        bad("No .env file found. Copy .env.example to .env and fill in your keys.")
        return False

    if not quick and not offline_test and not config.env("YOUTUBE_API_KEY"):
        bad("YOUTUBE_API_KEY is missing in .env  (the pipeline never invents data - a real key is required)")

    try:
        from core import db_setup
        upgraded = db_setup.ensure_schema(verbose=False)
        synced = db_setup.sync_channels()
        n = int(config.read_sql("SELECT COUNT(*) AS n FROM channels WHERE is_tracked = 1").iloc[0, 0])
        good(f"Database ready ({config.db_label()})" + (f" - upgraded: {', '.join(upgraded)}" if upgraded else ""))
        if getattr(db_setup.sync_channels, "untracked", 0):
            warn(f"{db_setup.sync_channels.untracked} channel(s) in the database are not in target_channels.json - "
                 f"now untracked (their data is kept)")
        if n == 0:
            bad("No tracked channels - add them to target_channels.json")
        elif n < 5:
            warn(f"{n} tracked channel(s) - the project brief asks for at least 5 (edit target_channels.json)")
        else:
            good(f"{n} tracked competitor channels" + (f" (synced from {config.CHANNELS_FILE.name})" if synced else ""))
    except Exception as exc:  # noqa: BLE001
        bad(f"Cannot use the database ({config.db_label()}): {str(exc).splitlines()[0][:110]}\n"
            f"      -> for MySQL check the service is running and DB_HOST / DB_USER / DB_PASSWORD / DB_NAME in .env")

    mail = config.mail_settings()
    if mail["sender"] and mail["recipients"]:
        good(f"E-mail alerts -> {', '.join(mail['recipients'])}")
    else:
        warn("E-mail not configured (SENDER_EMAIL / RECIPIENT_EMAIL) - the RPA bot cannot send until you set them")
    return ok


# ---------------------------------------------------------------------------
# Running one stage with a live progress callback
# ---------------------------------------------------------------------------
def run_stage(cmd, typical, parser, on_update, env):
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    lines: list[str] = []
    state = {"latest": "", "pct": None}
    ctx: dict = {}

    def reader():
        for line in proc.stdout:
            lines.append(line)
            if line.strip():
                state["latest"] = line.strip()
            if parser:
                pct = parser(line, ctx)
                if pct is not None:
                    state["pct"] = pct

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    start = time.time()
    try:
        while proc.poll() is None:
            eased = 92 * (1 - math.exp(-(time.time() - start) / max(typical / 2.5, 1)))
            on_update(state["pct"] if state["pct"] is not None else eased, state["latest"])
            time.sleep(0.12)
    except KeyboardInterrupt:
        proc.terminate()
        raise
    t.join(timeout=5)
    return proc.returncode, "".join(lines), time.time() - start


def _hint(text: str, width: int = 34) -> str:
    text = re.sub(r"[^\x20-\x7E]", "", text).strip()
    return (text[: width - 1] + "…") if len(text) > width else text


def highlights(console):
    """Short result summary printed when everything is done."""
    try:
        from core import config
        topics = config.read_sql("SELECT topic, score FROM topic_suggestions WHERE scope = 'ALL' ORDER BY rank_no LIMIT 3")
        pend = int(config.read_sql("SELECT COUNT(*) AS n FROM anomaly_alerts a JOIN channels c ON c.channel_id = a.channel_id WHERE a.emailed = 0 AND c.is_tracked = 1").iloc[0, 0])
        rel = config.read_sql("SELECT ROUND(AVG(r.relevance_index),0) AS r FROM video_relevance r JOIN videos v ON v.video_id = r.video_id JOIN channels c ON c.channel_id = v.channel_id WHERE c.is_tracked = 1").iloc[0, 0]
        if len(topics):
            console.print("[bold]Next-video ideas:[/] " + "  ·  ".join(f"{i}. {t.topic}" for i, t in enumerate(topics.itertuples(), 1)))
        if rel == rel and rel is not None:
            console.print(f"[bold]Avg content relevance:[/] {int(rel)}/100")
        if pend:
            console.print(f"[yellow]{pend} anomaly alert(s) still waiting to be e-mailed (see data/rpa_logs).[/]")
    except Exception:
        pass


def main() -> int:
    args = set(sys.argv[1:])
    quick = "--quick" in args
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    console = Console()
    console.print()
    console.rule(f"[bold cyan]MediaIntel Pipeline[/]  [dim]{datetime.now():%d %b %Y, %H:%M}[/]")
    console.print()

    if not preflight(console, quick):
        console.print("\n[red]Fix the items marked ✖ above and run again.[/]")
        return 2
    console.print()

    stages = [s for s in STAGES if s[5]] if quick else STAGES
    total = len(stages)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
               HF_HUB_DISABLE_SYMLINKS_WARNING="1", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1")
    rpa_env = dict(env, RPA_DRY_RUN="1") if "--no-email" in args else env

    progress = Progress(
        TextColumn("[bold]Stage {task.fields[idx]}/" + str(total) + "[/]"),
        TextColumn("{task.description}", table_column=Column(width=22, no_wrap=True)),
        BarColumn(bar_width=18, complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TextColumn("{task.fields[status]}", table_column=Column(width=44, no_wrap=True, overflow="ellipsis")),
        TimeElapsedColumn(),
        console=console,
    )

    failures: list[tuple[str, str]] = []
    started = time.time()
    with open(LOG_FILE, "a", encoding="utf-8") as log, progress:
        log.write(f"\n{'=' * 70}\nRUN {datetime.now():%Y-%m-%d %H:%M:%S}  mode={'quick' if quick else 'full'}\n{'=' * 70}\n")
        tasks = [progress.add_task(name, total=100, idx=i + 1, status="[dim]waiting[/]")
                 for i, (name, *_r) in enumerate(stages)]
        try:
            for task, (name, cmd, typical, parser, summarise, _q) in zip(tasks, stages):
                progress.update(task, status="[yellow]starting…[/]")

                def tick(pct, latest, t=task):
                    progress.update(t, completed=pct, status=f"[yellow]{_hint(latest) or 'running…'}[/]")

                code, out, secs = run_stage(cmd, typical, parser, tick, rpa_env if name.startswith("RPA") else env)
                log.write(f"\n--- {name} exit={code} {secs:.1f}s ---\n{out}\n")
                log.flush()
                if code == 0:
                    msg, ok = summarise(out)
                    progress.update(task, completed=100,
                                    status=f"[{'green' if ok else 'yellow'}]{'✔' if ok else '⚠'} {msg}[/]")
                else:
                    failures.append((name, out))
                    progress.update(task, status="[bold red]✖ Failed - see below[/]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/]")
            return 130

    console.print()
    if failures:
        for name, out in failures:
            tail = "\n".join(out.strip().splitlines()[-10:])
            console.print(f"[bold red]✖ {name} failed.[/] Last lines:")
            console.print(tail + "\n", style="dim", markup=False, highlight=False)
        console.print(f"[red]Finished with {len(failures)} failed stage(s).[/] Full details: [cyan]data/logs/pipeline.log[/]")
    else:
        console.print(f"[bold green]All {total} stages completed in {time.time() - started:.0f}s.[/]")
    highlights(console)
    console.print("[dim]PDF report  : outputs/MediaIntel_Dashboard_Documentation.pdf  (generated by the RPA bot)[/]")
    console.print("[dim]Strategy    : outputs/Content_Strategy_Report.txt   ·   Alerts: outputs/AlertSummary.txt[/]")
    console.print("[dim]RPA run log : data/rpa_logs/rpa_log.html   ·   Full log: data/logs/pipeline.log[/]\n")

    if "--no-dashboard" not in args:
        console.print("[bold cyan]📊 Launching Streamlit Web Dashboard...[/]  (Ctrl+C to stop)")
        try:
            subprocess.run([PY, "-m", "streamlit", "run", str(ROOT / "dashboard" / "dashboard_app.py"),
                            "--browser.gatherUsageStats", "false"], cwd=ROOT)
        except KeyboardInterrupt:
            pass
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Could not launch Streamlit automatically: {exc}[/]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
