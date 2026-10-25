"""Stage 6 - Executive content-strategy report (text), driven by the live data."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
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
        tx = d.get("transcripts", pd.DataFrame())
        whisper_cnt = int((tx["source"] == "whisper").sum()) if not tx.empty else 0
        captions_cnt = int((tx["source"] == "captions").sum()) if not tx.empty else 0
        if whisper_cnt > 0:
            src_label = "from Whisper transcripts + comments"
        elif captions_cnt > 0:
            src_label = "from YouTube captions + comments"
        else:
            src_label = "from video titles + comments"
        lines += ["", f"TOPICS FOR YOUR NEXT VIDEO ({src_label})"]
        for t in topics.head(5).itertuples(index=False):
            lines.append(f"{t.rank_no}. {t.topic}  (score {t.score:.0f})")
            lines.append(f"    {t.evidence}")

    lines += ["", "AI-DRIVEN RECOMMENDATIONS"]
    for n, (head, body) in enumerate(recs, 1):
        lines.append(f"{n}. {head}\n   {body}")

    text_out = "\n".join(lines)
    try:
        print(text_out)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text_out.encode(enc, errors="replace").decode(enc, errors="replace"))
    REPORT_PATH.write_text(text_out, encoding="utf-8")
    print(f"\nStrategy report saved to 'outputs/{REPORT_PATH.name}'.")


if __name__ == "__main__":
    generate_strategic_insights()
