"""Stage 5 - Engagement anomaly detection (views, likes AND comments).

For every recently-updated video it compares the current hourly rate of each metric with the
channel's own baseline:

  * lifetime rate : metric / hours since publish (UTC)            vs channel mean   -> "viral signature"
  * recent rate   : growth since an earlier snapshot (>= 1 h ago) vs channel baseline -> "spiking right now"

A metric is anomalous when its rate is above  ANOMALY_THRESHOLD x baseline  (default 3x = +200 %) AND above a
small absolute floor (so tiny numbers cannot trigger noise). Severity follows the ratio:
MEDIUM (>= threshold), HIGH (>= 5x), CRITICAL (>= 10x). New anomalies are written to `anomaly_alerts`
(a video+metric already reported in the last ANOMALY_COOLDOWN_HOURS is not reported again), and a
human-readable AlertSummary.txt is produced. The RPA bot (rpa/mediaintel_rpa.robot) e-mails the result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402

THRESHOLD = config.env_float("ANOMALY_THRESHOLD", 3.0)
COOLDOWN_H = config.env_float("ANOMALY_COOLDOWN_HOURS", 24)
LOOKBACK_H = config.env_float("ANOMALY_LOOKBACK_HOURS", 24)
MIN_BASELINE_VIDEOS = 3
FLOORS = {"views": 50.0, "likes": 2.0, "comments": 0.5}      # min events per hour before a spike counts
COLUMN = {"views": "view_count", "likes": "like_count", "comments": "comment_count"}
UNIT = {"views": "views/hr", "likes": "likes/hr", "comments": "comments/hr"}
SUMMARY_FILE = config.OUTPUTS_DIR / "AlertSummary.txt"
MAX_ALERTS = max(config.env_int("MAX_ALERTS_PER_EMAIL", 10), 1)   # most alerts listed per summary / e-mail
LOOKBACK_H = config.env_int("ALERT_LOOKBACK_HOURS", 72)           # recent, already-sent alerts that fill a repeat / quick run


def severity_for(ratio: float) -> str:
    return "CRITICAL" if ratio >= 10 else "HIGH" if ratio >= 5 else "MEDIUM"


def _recent_rates(videos: pd.DataFrame) -> pd.DataFrame:
    """Hourly growth since the latest snapshot that is at least 1 hour older than the newest one."""
    hist = config.read_sql("SELECT video_id, view_count, like_count, comment_count, captured_at "
                           "FROM video_stats_history ORDER BY captured_at")
    hist = hist[hist["video_id"].isin(videos["video_id"])]
    if hist.empty:
        return pd.DataFrame(columns=["video_id", "recent_views", "recent_likes", "recent_comments"])
    rows = []
    from datetime import timedelta
    for vid, g in hist.groupby("video_id"):
        if len(g) < 2:
            continue
        g_captured = pd.to_datetime(g["captured_at"])
        newest_time = g_captured.iloc[-1]
        older_mask = g_captured <= (newest_time - timedelta(hours=1))
        older = g[older_mask]
        if older.empty:
            continue
        ref = older.iloc[-1]
        newest = g.iloc[-1]
        ref_time = g_captured[older.index[-1]]
        hrs = (newest_time - ref_time).total_seconds() / 3600.0
        if hrs <= 0:
            continue
        rows.append({"video_id": vid,
                     "recent_views": (newest["view_count"] - ref["view_count"]) / hrs,
                     "recent_likes": (newest["like_count"] - ref["like_count"]) / hrs,
                     "recent_comments": (newest["comment_count"] - ref["comment_count"]) / hrs})
    return pd.DataFrame(rows, columns=["video_id", "recent_views", "recent_likes", "recent_comments"])


def detect() -> list[dict]:
    vids = config.read_sql("""
        SELECT v.video_id, v.title, v.channel_id, c.channel_name, v.view_count, v.like_count,
               v.comment_count, v.published_at, v.fetched_at
        FROM videos v JOIN channels c ON c.channel_id = v.channel_id
        WHERE v.published_at IS NOT NULL AND c.is_tracked = TRUE""")
    if vids.empty:
        return []
    now = pd.Timestamp(config.utcnow())
    pub_series = pd.to_datetime(vids["published_at"])
    vids["hours"] = ((now - pub_series).dt.total_seconds() / 3600.0).clip(lower=1.0)
    for m, col in COLUMN.items():
        vids[f"rate_{m}"] = vids[col] / vids["hours"]

    recent = _recent_rates(vids)
    vids = vids.merge(recent, on="video_id", how="left")

    # baselines per channel (needs a few videos to mean anything)
    grp = vids.groupby("channel_id")
    n_vids = grp["video_id"].transform("count")
    found = []
    for m in COLUMN:
        base_life = grp[f"rate_{m}"].transform("mean")
        for _, v in vids.iterrows():
            i = v.name
            if n_vids[i] < MIN_BASELINE_VIDEOS:
                continue
            candidates = [(v[f"rate_{m}"], base_life[i], "lifetime")]
            if pd.notna(v.get(f"recent_{m}")):
                candidates.append((v[f"recent_{m}"], base_life[i], "recent"))
            for rate, base, kind in candidates:
                if base > 0 and rate > THRESHOLD * base and rate >= FLOORS[m]:
                    found.append({"video_id": v.video_id, "channel_id": v.channel_id, "channel_name": v.channel_name,
                                  "title": v.title, "metric": m, "velocity": float(rate), "baseline": float(base),
                                  "ratio": float(rate / base), "kind": kind, "severity": severity_for(rate / base)})
                    break   # one alert per video+metric
    return found


def _already_reported() -> set[tuple[str, str]]:
    if COOLDOWN_H <= 0:
        return set()
    from datetime import timedelta
    cutoff = config.utcnow() - timedelta(hours=COOLDOWN_H)
    df = config.read_sql("SELECT video_id, metric FROM anomaly_alerts WHERE detected_at >= :cut",
                         {"cut": cutoff})
    return set(zip(df["video_id"], df["metric"]))


def pending_alerts() -> pd.DataFrame:
    """Alerts the RPA bot has not e-mailed yet (includes anything from earlier runs that failed to send)."""
    return config.read_sql("""
        SELECT a.id, a.metric, a.velocity, a.baseline, a.ratio, a.severity, a.detected_at,
               v.title, c.channel_name
        FROM anomaly_alerts a JOIN videos v ON v.video_id = a.video_id
        JOIN channels c ON c.channel_id = a.channel_id
        WHERE a.emailed = 0 AND c.is_tracked = TRUE ORDER BY a.ratio DESC""")


def alerts_for_report(limit: int = MAX_ALERTS) -> tuple[pd.DataFrame, int]:
    """The alerts listed in EVERY e-mail and in AlertSummary.txt, so a repeat or --quick run looks just like a full run:
    unsent alerts first (strongest first), then topped up with the strongest recent alerts that were already sent.
    Returns (alerts with an `is_new` flag, number of unsent alerts in total)."""
    from datetime import timedelta
    base = """
        SELECT a.id, a.metric, a.velocity, a.baseline, a.ratio, a.severity, a.detected_at, a.emailed,
               v.title, c.channel_name
        FROM anomaly_alerts a JOIN videos v ON v.video_id = a.video_id
        JOIN channels c ON c.channel_id = a.channel_id
        WHERE c.is_tracked = TRUE AND """
    new = config.read_sql(base + "a.emailed = 0 ORDER BY a.ratio DESC, a.id")
    total_new = len(new)
    frames = [new.head(limit)]
    room = limit - len(frames[0])
    if room > 0 and LOOKBACK_H > 0:
        cutoff = config.utcnow() - timedelta(hours=LOOKBACK_H)
        old = config.read_sql(base + "a.emailed = 1 AND a.detected_at >= :cut ORDER BY a.ratio DESC, a.id",
                              {"cut": cutoff})
        frames.append(old.head(room))
    frames = [f for f in frames if not f.empty]
    df = pd.concat(frames, ignore_index=True) if frames else new
    df["is_new"] = df["emailed"].astype(int) == 0
    return df, total_new


def write_summary(alerts: pd.DataFrame, total_new: int = 0) -> None:
    lines = ["MEDIAINTEL - COMPETITOR ENGAGEMENT ANOMALY REPORT",
             "=" * 62,
             f"Generated (UTC): {config.utcnow():%Y-%m-%d %H:%M}",
             f"Rule: metric rate > {THRESHOLD:g}x the channel baseline (+{(THRESHOLD - 1) * 100:.0f}%).", ""]
    if not alerts.empty:
        n_new = int(alerts["is_new"].sum())
        lines.append(f"{len(alerts)} anomaly signal(s) shown - {n_new} new, {len(alerts) - n_new} reported earlier:\n")
        for n, a in enumerate(alerts.itertuples(index=False), 1):
            lines += [f"{n}. [{a.severity or severity_for(a.ratio)}] [{a.metric.upper()}] {a.channel_name}   ({'NEW' if a.is_new else 'REPORTED EARLIER'})",
                      f"   Video : {a.title}",
                      f"   Rate  : {a.velocity:.1f} {UNIT[a.metric]}  (channel baseline {a.baseline:.1f}, {a.ratio:.1f}x)",
                      f"   Seen  : {a.detected_at:%Y-%m-%d %H:%M} UTC", ""]
        if total_new > n_new:
            lines.append(f"... and {total_new - n_new} more new alert(s) waiting for the next e-mail.\n")
    else:
        lines.append("No anomalies recorded recently - every tracked video is within its normal baseline range.")
    lines.append("\nOpen the live Streamlit dashboard (Alerts tab) to evaluate market reaction trends.")
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def run_anomaly_detection() -> None:
    print("Initializing engagement anomaly engine (views / likes / comments)...")
    found = detect()
    seen = _already_reported()
    new = [a for a in found if (a["video_id"], a["metric"]) not in seen]

    if new:
        config.execute("""INSERT INTO anomaly_alerts (video_id, channel_id, metric, velocity, baseline, ratio, severity, detected_at, emailed)
                          VALUES (:video_id, :channel_id, :metric, :velocity, :baseline, :ratio, :severity, :now, 0)""",
                       [dict(a, now=config.utcnow()) for a in new])
    pending = pending_alerts()

    for a in sorted(new, key=lambda r: -r["ratio"]):
        print(f"\nANOMALY DETECTED [{a['severity']}] [{a['metric']}]\n   Video: '{a['title']}'\n   Channel: {a['channel_name']}\n"
              f"   Current rate: {a['velocity']:.1f} {UNIT[a['metric']]} (baseline {a['baseline']:.1f}, {a['ratio']:.1f}x)")
    report, total_new = alerts_for_report()
    write_summary(report, total_new)
    if new:
        print(f"\nAnomaly check complete. {len(new)} new anomaly signal(s) recorded; {len(pending)} queued for e-mail.")
    elif found:
        print(f"\nAnomaly check complete. {len(found)} known anomaly signal(s) already reported - nothing new.")
    else:
        print("\nVelocity scan complete. All tracked videos are within normal baseline ranges.")


if __name__ == "__main__":
    run_anomaly_detection()
