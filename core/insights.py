"""Shared data loading + data-driven recommendations for the PDF report and the strategy report."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402


def load_all() -> dict[str, pd.DataFrame]:
    """Everything the reports need, straight from MySQL."""
    q = config.read_sql
    return {
        "channels": q("SELECT * FROM channels WHERE is_tracked = 1"),
        "videos": q("SELECT v.* FROM videos v JOIN channels c ON c.channel_id = v.channel_id WHERE c.is_tracked = 1"),
        "insights": q("""SELECT s.* FROM sentiment_insights s JOIN comments m ON m.comment_id = s.comment_id
                           JOIN videos v ON v.video_id = m.video_id JOIN channels c ON c.channel_id = v.channel_id
                           WHERE c.is_tracked = 1"""),
        "aspects": q("""SELECT a.* FROM aspect_sentiments a JOIN comments m ON m.comment_id = a.comment_id
                        JOIN videos v ON v.video_id = m.video_id JOIN channels c ON c.channel_id = v.channel_id
                        WHERE c.is_tracked = 1"""),
        "relevance": q("""
            SELECT r.*, v.title, v.channel_id, c.channel_name
            FROM video_relevance r JOIN videos v ON v.video_id = r.video_id
            JOIN channels c ON c.channel_id = v.channel_id WHERE c.is_tracked = 1"""),
        "topics": q("""
            SELECT t.*, COALESCE(c.channel_name, 'All channels') AS channel_name
            FROM topic_suggestions t LEFT JOIN channels c ON c.channel_id = t.scope
            ORDER BY t.scope, t.rank_no"""),
        "transcripts": q("SELECT video_id, source, language, model_name, word_count, audio_seconds, transcript_text FROM transcripts"),
        "comment_videos": q("SELECT comment_id, video_id FROM comments"),
        "alerts": q("""
            SELECT a.detected_at, a.metric, a.velocity, a.baseline, a.ratio, a.severity, a.emailed, v.title, c.channel_name
            FROM anomaly_alerts a JOIN videos v ON v.video_id = a.video_id
            JOIN channels c ON c.channel_id = a.channel_id
            WHERE a.detected_at >= :cut AND c.is_tracked = 1 ORDER BY a.detected_at DESC""",
                   {"cut": config.utcnow() - pd.Timedelta(days=7)}),
    }


def build_recommendations(d: dict[str, pd.DataFrame]) -> list[tuple[str, str]]:
    """Returns [(headline, explanation), ...] computed from the live data."""
    recs: list[tuple[str, str]] = []
    topics = d["topics"][d["topics"]["scope"] == "ALL"] if not d["topics"].empty else d["topics"]

    if not topics.empty:
        t = topics.iloc[0]
        recs.append((f"Make \"{t.topic}\" your next video.",
                     f"It ranks first across all competitors (score {t.score:.0f}/100). {t.evidence}"))
        asked = topics[topics["request_count"] > 0].sort_values("request_count", ascending=False)
        if not asked.empty and asked.iloc[0].topic != t.topic:
            a = asked.iloc[0]
            recs.append((f"Answer the most requested topic: \"{a.topic}\".",
                         f"{int(a.request_count)} viewers explicitly asked for it and only "
                         f"{a.transcript_coverage * 100:.0f}% of the tracked videos touch it."))
        gaps = topics.sort_values("transcript_coverage").iloc[0]
        if gaps.topic not in (t.topic,) and gaps.transcript_coverage == 0:
            recs.append((f"Fill the content gap on \"{gaps.topic}\".",
                         f"No competitor transcript covers it, yet {int(gaps.mentions)} comments mention it."))

    tr = d["transcripts"]
    if not tr.empty and (tr["source"] == "whisper").any():
        spoken = spoken_topics(tr[tr["source"].isin(["whisper", "captions"])][["video_id", "transcript_text"]], top_k=3)
        if not spoken.empty:
            said = ", ".join(spoken["topic"].head(3))
            tail = ""
            if not topics.empty:
                tail = f" Viewers are asking for \"{topics.iloc[0].topic}\" instead - cover it to stand apart."
            recs.append(("Differentiate from what competitors already say.",
                         f"Whisper analysis of their audio shows they keep returning to {said}.{tail}"))

    asp = d["aspects"]
    if not asp.empty:
        neg = asp[(asp["sentiment_label"] == "Negative") & (asp["aspect"] != "General Feedback")]
        if not neg.empty:
            worst = neg["aspect"].value_counts().idxmax()
            n = int((neg["aspect"] == worst).sum())
            recs.append((f"Exploit competitor friction around \"{worst}\".",
                         f"{n} comments voice negative sentiment about it - position your video as the better "
                         f"alternative on this point."))

    rel = d["relevance"].dropna(subset=["relevance_index"])
    if not rel.empty:
        best = rel.sort_values("relevance_index", ascending=False).iloc[0]
        recs.append(("Replicate the structure of the best-aligned video.",
                     f"\"{best.title[:70]}\" ({best.channel_name}) has the strongest match between what is said "
                     f"and what viewers discuss (relevance index {best.relevance_index:.0f}/100)."))

    al = d["alerts"]
    if not al.empty:
        top = al.sort_values("ratio", ascending=False).iloc[0]
        recs.append(("Study the video that is spiking right now.",
                     f"\"{top.title[:70]}\" ({top.channel_name}) runs at {top.ratio:.1f}x its channel's normal "
                     f"{top.metric} rate - analyse its hook and title."))

    if not recs:
        recs.append(("Collect more data.", "Run the full pipeline so topics, relevance and anomalies can be computed."))
    return recs


# ---------------------------------------------------------------------------
# helpers shared by the PDF report and the dashboard
# ---------------------------------------------------------------------------
def aspect_scorecard(aspects: pd.DataFrame, comments_videos: pd.DataFrame, videos: pd.DataFrame, channels: pd.DataFrame) -> pd.DataFrame:
    """Channel x aspect table of NET SENTIMENT = % positive - % negative (range -100 ... +100)."""
    if aspects.empty:
        return pd.DataFrame()
    j = (aspects.merge(comments_videos[["comment_id", "video_id"]], on="comment_id")
                .merge(videos[["video_id", "channel_id"]], on="video_id")
                .merge(channels[["channel_id", "channel_name"]], on="channel_id"))
    g = j.groupby(["channel_name", "aspect"])["sentiment_label"]
    net = (g.apply(lambda x: 100 * ((x == "Positive").mean() - (x == "Negative").mean()))).unstack("aspect")
    return net.round(0)


def whisper_summary(transcripts: pd.DataFrame, videos: pd.DataFrame) -> dict:
    """Facts about the Whisper audio analysis: coverage, minutes analysed, languages, speaking pace."""
    if transcripts.empty:
        return {"videos": 0, "minutes": 0.0, "languages": {}, "pace": None, "words": 0, "share": 0.0}
    w = transcripts[transcripts["source"] == "whisper"]
    secs = float(w["audio_seconds"].fillna(0).sum()) if "audio_seconds" in w else 0.0
    words = int(w["word_count"].fillna(0).sum())
    langs = w["language"].fillna("und").value_counts().to_dict()
    return {"videos": int(len(w)), "minutes": secs / 60, "languages": langs,
            "pace": (words / (secs / 60)) if secs > 0 else None, "words": words,
            "share": 100 * len(w) / max(len(videos), 1)}


def spoken_topics(videos_text: pd.DataFrame, top_k: int = 12) -> pd.DataFrame:
    """Topics the creators actually SAY (from Whisper transcripts): phrase -> number of videos mentioning it.

    videos_text needs columns video_id, transcript_text. Uses the same phrase extractor as the viewer-topic miner."""
    from collections import Counter
    from core import relevance_topics as rt
    counts: Counter = Counter()
    for body in videos_text["transcript_text"].dropna():
        for g in rt.ngrams(str(body), set()):
            counts[g] += 1
    rows = [(rt.pretty(g), n) for g, n in counts.items() if n >= 2 and len(g) > 2]
    # drop a short phrase if a longer one containing it is about as frequent
    rows.sort(key=lambda r: (-r[1], -len(r[0])))
    kept = []
    for t, n in rows:
        if not any(t.lower() in k.lower() and kn >= n * 0.8 for k, kn in kept):
            kept.append((t, n))
    return pd.DataFrame(kept[:top_k], columns=["topic", "videos"])
