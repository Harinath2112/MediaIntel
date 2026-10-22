"""Stage 6 - Executive content-strategy report (text), driven by the live data."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402
from core import insights  # noqa: E402

REPORT_PATH = config.OUTPUTS_DIR / "Content_Strategy_Report.txt"


def generate_strategic_insights() -> None:
    print("Analyzing the MediaIntel database for strategic content recommendations...")
    d = insights.load_all()
    videos, channels = d["videos"], d["channels"]
    top = (videos.merge(channels[["channel_id", "channel_name"]], on="channel_id")
           .nlargest(5, "view_count")) if not videos.empty else videos
    recs = insights.build_recommendations(d)

    lines = ["MEDIAINTEL EXECUTIVE CONTENT STRATEGY REPORT", "=" * 60, "",
             "TOP PERFORMING COMPETITOR VIDEOS"]
    for r in top.itertuples(index=False):
        lines.append(f"- [{r.channel_name}] {r.title[:80]}")
        lines.append(f"    Views {int(r.view_count):,} | Likes {int(r.like_count):,} | Comments {int(r.comment_count):,}")

    topics = d["topics"][d["topics"]["scope"] == "ALL"]
    if not topics.empty:
        lines += ["", "TOPICS FOR YOUR NEXT VIDEO (from Whisper transcripts + comments)"]
        for t in topics.head(5).itertuples(index=False):
            lines.append(f"{t.rank_no}. {t.topic}  (score {t.score:.0f})")
            lines.append(f"    {t.evidence}")

    lines += ["", "AI-DRIVEN RECOMMENDATIONS"]
    for n, (head, body) in enumerate(recs, 1):
        lines.append(f"{n}. {head}\n   {body}")

    text_out = "\n".join(lines)
    print(text_out)
    REPORT_PATH.write_text(text_out, encoding="utf-8")
    print(f"\nStrategy report saved to 'outputs/{REPORT_PATH.name}'.")


if __name__ == "__main__":
    generate_strategic_insights()
