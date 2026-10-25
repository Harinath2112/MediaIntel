import io
import re
import sys
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402
from core import insights as ins  # noqa: E402

PDF_PATH = str(config.OUTPUTS_DIR / "MediaIntel_Dashboard_Documentation.pdf")

# Same palette family the Streamlit dashboard uses
PASTEL = ["#66c5cc", "#f6cf71", "#f89c74", "#dcb0f2", "#87c55f", "#9eb9f3", "#fe88b1", "#c9db74"]
SENTIMENT_COLORS = {"Positive": "#48bb78", "Negative": "#e53e3e", "Neutral": "#a0aec0"}

plt.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#cbd5e0",
    "axes.labelcolor": "#4a5568",
    "xtick.color": "#4a5568",
    "ytick.color": "#4a5568",
})


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def fetch_dashboard_data():
    """Loads every table the report uses (same source as the Streamlit dashboard)."""
    return ins.load_all()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def clean_text(value, limit=None):
    """Safely normalizes text for PDF rendering, replacing rupee and unsupported glyphs."""
    s = str(value)
    # Map Indian Rupee symbol and other currency/math symbols to clean ASCII equivalents
    s = s.replace("₹", "Rs. ").replace("\u20B9", "Rs. ")
    s = s.replace("€", "EUR ").replace("£", "GBP ")
    # Map typographical quotes and dashes to standard ASCII equivalents
    s = s.replace("–", "-").replace("—", "-").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # Filter to printable Latin range supported by ReportLab standard fonts
    s = re.sub(r"[^\x20-\x7E\u00A0-\u00FF]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if limit and len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return s


def compact(x, _pos=None):
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{x:.0f}"


def shade(cmap_name, values):
    """Colour bars by value (like plotly's continuous colour scale), avoiding washed-out lows."""
    cmap = matplotlib.colormaps[cmap_name]
    v = np.asarray(values, dtype=float)
    span = v.max() - v.min()
    norm = (v - v.min()) / span if span else np.ones_like(v)
    return [cmap(0.35 + 0.6 * n) for n in norm]


def fig_to_image(fig, width_pt):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    pw, ph = matplotlib.image.imread(io.BytesIO(buf.getvalue()), format="png").shape[:2][::-1]
    buf.seek(0)
    return Image(buf, width=width_pt, height=width_pt * ph / pw)


# ---------------------------------------------------------------------------
# CHARTS (mirror dashboard items 5-12)
# ---------------------------------------------------------------------------
HALF = (3.9, 2.8)
FULL = (7.4, 2.6)


def chart_aspect(insights):
    counts = insights["aspect_category"].value_counts()
    fig, ax = plt.subplots(figsize=HALF)
    bars = ax.bar(counts.index, counts.values, color=PASTEL[: len(counts)])
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], fontsize=7, padding=2)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=7)
    return fig


def chart_sentiment(insights):
    counts = insights["sentiment_label"].value_counts()
    fig, ax = plt.subplots(figsize=HALF)
    ax.pie(
        counts.values, labels=None, startangle=90, counterclock=False,
        colors=[SENTIMENT_COLORS.get(k, "#a0aec0") for k in counts.index],
        autopct="%1.1f%%", pctdistance=0.78, textprops={"fontsize": 7, "color": "white"},
        wedgeprops={"width": 0.45, "edgecolor": "white"},
    )
    ax.legend([f"{k} ({v:,})" for k, v in counts.items()], loc="center left",
              bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)
    ax.set_aspect("equal")
    return fig


def chart_confidence(insights):
    conf = insights.groupby("sentiment_label")["sentiment_score"].mean() * 100
    fig, ax = plt.subplots(figsize=HALF)
    bars = ax.bar(conf.index, conf.values, color=[SENTIMENT_COLORS.get(k, "#a0aec0") for k in conf.index])
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in conf.values], fontsize=7, padding=2)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, max(conf.max() * 1.18, 1))
    return fig


def chart_crosstab(aspects):
    """Aspect x sentiment heat-map. `aspects` has columns (aspect, sentiment_label) from aspect_sentiments."""
    ct = pd.crosstab(aspects["aspect"], aspects["sentiment_label"])
    ct = ct.reindex(columns=[c for c in ("Negative", "Neutral", "Positive") if c in ct.columns])
    fig, ax = plt.subplots(figsize=HALF)
    ax.imshow(ct.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(ct.columns)), ct.columns, fontsize=7)
    ax.set_yticks(range(len(ct.index)), ct.index, fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    peak = ct.values.max() or 1
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            val = ct.values[i, j]
            ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=7,
                    color="white" if val > peak * 0.55 else "#1a202c")
    return fig


def chart_top_videos(videos, column, cmap_name, xlabel):
    top = videos.nlargest(5, column).iloc[::-1]  # largest bar at the top
    labels = [clean_text(t, 60) for t in top["title"]]
    fig, ax = plt.subplots(figsize=FULL)
    ypos = np.arange(len(top))
    bars = ax.barh(ypos, top[column], color=shade(cmap_name, top[column]))
    ax.set_yticks(ypos, labels)
    ax.bar_label(bars, labels=[f"{int(v):,}" for v in top[column]], fontsize=7, padding=3)
    ax.xaxis.set_major_formatter(FuncFormatter(compact))
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.margins(x=0.12)
    return fig


def chart_trend(videos):
    fig, ax = plt.subplots(figsize=HALF)
    v_clean = videos.copy()
    if "published_at" in v_clean.columns and v_clean["published_at"].notna().any():
        v_clean["pub_dt"] = pd.to_datetime(v_clean["published_at"])
        v_sorted = v_clean.dropna(subset=["pub_dt"]).sort_values("pub_dt")
        x_dates = v_sorted["pub_dt"].dt.strftime("%b %d")
        y = v_sorted["view_count"].values
        ax.plot(range(len(y)), y, color="#805ad5", marker="o", markersize=3, linewidth=1.4)
        ax.fill_between(range(len(y)), y, color="#805ad5", alpha=0.08)
        tick_step = max(1, len(y) // 5)
        tick_indices = list(range(0, len(y), tick_step))
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([x_dates.iloc[i] for i in tick_indices], rotation=25, ha="right", fontsize=6.5)
        ax.set_xlabel("Publication Date (Chronological)")
    else:
        y = v_clean["view_count"].reset_index(drop=True)
        ax.plot(y.index, y.values, color="#805ad5", marker="o", markersize=3, linewidth=1.4)
        ax.fill_between(y.index, y.values, color="#805ad5", alpha=0.08)
        ax.set_xlabel("Video Sequence")
    ax.set_ylabel("Views")
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    return fig


def chart_comment_leaderboard(videos):
    top = videos.nlargest(5, "comment_count")
    labels = [clean_text(t, 18) for t in top["title"]]
    fig, ax = plt.subplots(figsize=HALF)
    xpos = np.arange(len(top))
    bars = ax.bar(xpos, top["comment_count"], color=shade("Purples", top["comment_count"]))
    ax.set_xticks(xpos, labels)
    ax.bar_label(bars, labels=[f"{int(v):,}" for v in top["comment_count"]], fontsize=7, padding=2)
    ax.set_ylabel("Comments")
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=6.5)
    return fig


def chart_relevance_by_channel(rel):
    g = rel.groupby("channel_name")["relevance_index"].mean().sort_values(ascending=False)
    labels = [clean_text(n, 16) for n in g.index]
    xpos = np.arange(len(g))
    fig, ax = plt.subplots(figsize=HALF)
    bars = ax.bar(xpos, g.values, color=PASTEL[: len(g)])
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in g.values], fontsize=7, padding=2)
    ax.set_xticks(xpos, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Avg relevance index (0-100)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=7)
    return fig


def chart_transcript_sources(videos, transcripts):
    src = transcripts.set_index("video_id")["source"] if not transcripts.empty else pd.Series(dtype=str)
    mapped = videos["video_id"].map(src).fillna("none").replace({"failed": "none"})
    names = {"whisper": "Whisper (audio)", "captions": "YouTube captions", "none": "Title only"}
    cols = {"whisper": "#805ad5", "captions": "#3182ce", "none": "#a0aec0"}
    counts = mapped.value_counts()
    fig, ax = plt.subplots(figsize=HALF)
    ax.pie(counts.values, startangle=90, counterclock=False, colors=[cols.get(k, "#a0aec0") for k in counts.index],
           autopct="%1.0f%%", pctdistance=0.78, textprops={"fontsize": 7, "color": "white"},
           wedgeprops={"width": 0.45, "edgecolor": "white"})
    ax.legend([f"{names.get(k, k)} ({v})" for k, v in counts.items()], loc="center left",
              bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)
    ax.set_aspect("equal")
    return fig


def chart_relevance_extremes(rel):
    ordered = rel.sort_values("relevance_index")
    k = min(5, max(len(ordered) // 2, 1))
    low, high = ordered.head(k), ordered.tail(k)
    df = pd.concat([low, high]) if len(ordered) >= 2 * k else ordered
    cols = ["#e53e3e"] * len(low) + ["#48bb78"] * len(high) if len(ordered) >= 2 * k else ["#48bb78"] * len(df)
    labels = [f"[{clean_text(c, 12)}] {clean_text(t, 46)}" for c, t in zip(df["channel_name"], df["title"])]
    ypos = np.arange(len(df))
    fig, ax = plt.subplots(figsize=FULL)
    bars = ax.barh(ypos, df["relevance_index"], color=cols)
    ax.set_yticks(ypos, labels)
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in df["relevance_index"]], fontsize=7, padding=3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Relevance index (green = best aligned, red = least aligned)")
    ax.tick_params(axis="y", labelsize=7.5)
    return fig


def chart_topics(topics):
    t = topics.head(10).iloc[::-1]
    ypos = np.arange(len(t))
    fig, ax = plt.subplots(figsize=FULL)
    bars = ax.barh(ypos, t["score"], color=shade("Oranges", t["score"]))
    ax.set_yticks(ypos, [clean_text(x, 40) for x in t["topic"]])
    ax.bar_label(bars, labels=[f"{int(m)} mentions | {int(r)} requests" for m, r in zip(t["mentions"], t["request_count"])],
                 fontsize=7, padding=3)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Opportunity score (0-100)")
    ax.tick_params(axis="y", labelsize=8)
    return fig


def chart_scorecard(score_tbl):
    """Channel x aspect heat-map of net sentiment (% positive - % negative)."""
    fig, ax = plt.subplots(figsize=FULL)
    vals = score_tbl.values.astype(float)
    ax.imshow(vals, cmap="RdYlGn", vmin=-100, vmax=100, aspect="auto")
    ax.set_xticks(range(vals.shape[1]), score_tbl.columns, fontsize=7.5)
    ax.set_yticks(range(vals.shape[0]), [clean_text(c, 22) for c in score_tbl.index], fontsize=8)
    ax.tick_params(length=0)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:+.0f}", ha="center", va="center", fontsize=7.5, color="#1a202c")
    return fig


def chart_languages(summary):
    langs = summary["languages"]
    fig, ax = plt.subplots(figsize=HALF)
    ax.pie(list(langs.values()), startangle=90, counterclock=False, colors=PASTEL[: len(langs)],
           autopct="%1.0f%%", pctdistance=0.78, textprops={"fontsize": 7, "color": "#1a202c"},
           wedgeprops={"width": 0.45, "edgecolor": "white"})
    ax.legend([f"{k} ({v})" for k, v in langs.items()], loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)
    ax.set_aspect("equal")
    return fig


def chart_pace_by_channel(pace):
    pace = pace.sort_values(ascending=False)
    xpos = np.arange(len(pace))
    fig, ax = plt.subplots(figsize=HALF)
    bars = ax.bar(xpos, pace.values, color=PASTEL[: len(pace)])
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in pace.values], fontsize=7, padding=2)
    ax.set_xticks(xpos, [clean_text(n, 16) for n in pace.index])
    ax.set_ylabel("Words per minute")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=7)
    return fig


def chart_spoken_topics(spoken):
    t = spoken.head(12).iloc[::-1]
    ypos = np.arange(len(t))
    fig, ax = plt.subplots(figsize=FULL)
    bars = ax.barh(ypos, t["videos"], color=shade("Purples", t["videos"]))
    ax.set_yticks(ypos, [clean_text(x, 40) for x in t["topic"]])
    ax.bar_label(bars, labels=[f"{int(v)} videos" for v in t["videos"]], fontsize=7, padding=3)
    ax.set_xlabel("Videos in which the topic is spoken")
    ax.margins(x=0.12)
    ax.tick_params(axis="y", labelsize=8)
    return fig


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def generate_professional_pdf():
    doc = SimpleDocTemplate(PDF_PATH, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    content_w = letter[0] - 80

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#1A365D"), spaceAfter=8
    )
    subtitle_style = ParagraphStyle(
        'DocSub', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#4A5568"), spaceAfter=14
    )
    heading_style = ParagraphStyle(
        'DocHead', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor("#2B6CB0"), spaceAfter=4, spaceBefore=10
    )
    body_style = ParagraphStyle(
        'DocBody', parent=styles['Normal'], fontSize=9.5, textColor=colors.HexColor("#2D3748"), spaceAfter=6, leading=13
    )
    chart_title_style = ParagraphStyle(
        'ChartTitle', parent=styles['Normal'], fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1E293B"), spaceBefore=4, spaceAfter=2
    )
    chart_note_style = ParagraphStyle(
        'ChartNote', parent=styles['Normal'], fontSize=7.5, leading=9.5,
        textColor=colors.HexColor("#64748B"), spaceBefore=2, spaceAfter=4
    )

    # Fetch live data (same source as the dashboard)
    data = fetch_dashboard_data()
    channels, videos, insights = data["channels"], data["videos"], data["insights"]
    aspects = data["aspects"]
    rel, topics_all, transcripts, alerts = data["relevance"], data["topics"], data["transcripts"], data["alerts"]
    total_channels, total_videos, total_insights = len(channels), len(videos), len(insights)

    # Title & Metadata
    story.append(Paragraph("MediaIntel Platform: Professional Dashboard & Architecture Documentation", title_style))
    story.append(Paragraph(
        f"Active Channels Tracked: {total_channels} | Total Videos Audited: {total_videos} | "
        f"Analyzed Comments: {total_insights} | Generated: {datetime.now():%d %b %Y, %H:%M}",
        subtitle_style))
    story.append(Spacer(1, 4))

    # Section 1: Dashboard Overview
    story.append(Paragraph("1. Dashboard System Architecture & Interface", heading_style))
    intro_text = (
        "The MediaIntel Web Dashboard is engineered as a portable, high-performance visual analytics layer "
        "built entirely in Python using Streamlit. It removes the dependency on desktop software like Power BI, "
        "allowing executive stakeholders to examine real-time competitor metrics on any machine via a web browser."
    )
    story.append(Paragraph(intro_text, body_style))

    # Section 2: Explanation of Dashboard Diagrams
    story.append(Paragraph("2. Detailed Explanation of Dashboard Visualizations", heading_style))
    diagram_explanations = (
        "<b>A. Executive KPI Metric Cards (Top Row):</b><br/>"
        "• <i>Active Competitors, Audited Videos, and Processed Comments:</i> Provides instant operational scale markers.<br/>"
        "• <i>AI Confidence Score:</i> Aggregates the mean probability certainty of the underlying RoBERTa transformer model across all inferences.<br/><br/>"
        "<b>B. Aspect-Based Sentiment (RoBERTa):</b><br/>"
        "• <i>Explanation:</i> Each analysed comment (the most-liked ones of every video) is split into clauses; each clause is matched to an aspect "
        "(<i>Content Quality, Engagement, Relevance, Feature Request, Pricing/Cost, Technical Issue, General Feedback</i>) "
        "and receives its own Negative/Neutral/Positive sentiment, so one comment can praise the explanation and criticise the audio.<br/><br/>"
        "<b>C. Sentiment Breakdown:</b><br/>"
        "• <i>Explanation:</i> Quantifies emotional polarity (Positive, Neutral, Negative) to track overall market sentiment trends.<br/><br/>"
        "<b>D. Top Performing Competitor Videos (Engagement Table):</b><br/>"
        "• <i>Explanation:</i> Ranks competitor uploads by view, like, and comment velocity to uncover high-impact engagement benchmarks.<br/><br/>"
        "<b>E. Whisper Audio Analysis:</b><br/>"
        "• <i>Explanation:</i> The first minutes of each video's audio are streamed into memory and analysed by OpenAI Whisper (nothing is stored). "
        "The report shows coverage, audio minutes analysed, spoken languages, speaking pace and the topics the creators actually talk about.<br/><br/>"
        "<b>F. Content Relevance (Whisper transcript x comments):</b><br/>"
        "• <i>Explanation:</i> Video audio is transcribed with OpenAI Whisper and compared, using multilingual embeddings, with what viewers write. "
        "The relevance index (0-100) shows how closely the discussion follows what the video actually says.<br/><br/>"
        "<b>G. Topics for the Next Video:</b><br/>"
        "• <i>Explanation:</i> Topics viewers mention and request are ranked by demand, explicit requests, content gap in existing transcripts and audience sentiment.<br/><br/>"
        "<b>H. Engagement Anomaly Alerts:</b><br/>"
        "• <i>Explanation:</i> Views, likes and comments are monitored against each channel's baseline; spikes are graded MEDIUM / HIGH / CRITICAL and e-mailed by the RPA bot."
    )
    story.append(Paragraph(diagram_explanations, body_style))

    # Section 3: Live dashboard analytics (charts)
    story.append(PageBreak())
    story.append(Paragraph("3. Live Dashboard Analytics (All Channels)", heading_style))
    story.append(Paragraph(
        "The visuals below are rendered from the same warehouse tables that power the Streamlit dashboard, "
        "so this document always reflects the data at the moment the report was generated.", body_style))

    # KPI cards
    avg_conf = (insights["sentiment_score"].mean() * 100) if not insights.empty else 0
    kpi_label = ParagraphStyle('KpiL', parent=styles['Normal'], fontSize=7.5, alignment=1,
                               textColor=colors.HexColor("#64748B"))
    kpi_value = ParagraphStyle('KpiV', parent=styles['Normal'], fontSize=16, leading=20, alignment=1,
                               fontName="Helvetica-Bold", textColor=colors.HexColor("#0F172A"))
    kpis = [
        ("TRACKED COMPETITORS", f"{total_channels}"),
        ("AUDITED VIDEOS", f"{total_videos:,}"),
        ("AUDITED COMMENTS", f"{total_insights:,}"),
        ("MEAN AI CONFIDENCE", f"{avg_conf:.1f}%"),
    ]
    kpi_table = Table(
        [[Paragraph(label, kpi_label) for label, _ in kpis], [Paragraph(val, kpi_value) for _, val in kpis]],
        colWidths=[content_w / 4] * 4,
    )
    kpi_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 8))

    def cell(title, builder, note, width):
        """One chart block: title, image, one-line explanation. Falls back to a note if data is missing."""
        try:
            img = fig_to_image(builder(), width)
        except Exception as exc:  # never let one chart break the whole report
            img = Paragraph(f"<i>Chart unavailable ({escape(str(exc)[:80])}).</i>", chart_note_style)
        return [Paragraph(escape(title), chart_title_style), img, Paragraph(escape(note), chart_note_style)]

    def pair(left, right):
        t = Table([[left, right]], colWidths=[content_w / 2] * 2)
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        return KeepTogether([t])

    half_w = content_w / 2 - 8
    has_insights = not insights.empty
    has_videos = not videos.empty

    if has_insights:
        story.append(pair(
            cell("5. Aspect-Based Feedback Distribution", lambda: chart_aspect(insights),
                 "Groups comments into Bug/UI, Pricing, Feature Request and General Feedback.", half_w),
            cell("6. Sentiment Polarity Breakdown", lambda: chart_sentiment(insights),
                 "Share of Positive, Neutral and Negative comments across all tracked channels.", half_w)))
        story.append(pair(
            cell("7. Average AI Confidence by Sentiment Class", lambda: chart_confidence(insights),
                 "Mean model confidence per sentiment class; higher means a more decisive reading.", half_w),
            cell("8. Aspect-Level Sentiment Matrix", lambda: chart_crosstab(
                     aspects if not aspects.empty else insights.rename(columns={"aspect_category": "aspect"})),
                 "RoBERTa sentiment towards each aspect (a comment can hold several); darker cells = more comments.", half_w)))
    else:
        story.append(Paragraph("<i>No sentiment records available yet.</i>", body_style))

    if has_videos:
        for title, col, cmap, label, note in (
            ("9. Top Videos Ranked by View Count", "view_count", "Blues", "Views",
             "The five most-viewed uploads across all tracked channels."),
            ("10. Top Videos Ranked by Like Count", "like_count", "Greens", "Likes",
             "The five most-liked uploads, a proxy for audience approval."),
        ):
            story.append(KeepTogether(cell(
                title, lambda c=col, m=cmap, l=label: chart_top_videos(videos, c, m, l), note, content_w * 0.97)))
        story.append(pair(
            cell("11. View Count Chronological Timeline Trend", lambda: chart_trend(videos),
                 "View counts across uploads ordered chronologically by publication date; peaks mark standout uploads.", half_w),
            cell("12. Comment Volume per Upload Leaderboard", lambda: chart_comment_leaderboard(videos),
                 "The five uploads that generated the most discussion.", half_w)))

        # Engagement table referenced in section 2-D
        top = videos.merge(channels[["channel_id", "channel_name"]], on="channel_id", how="left") \
                    .nlargest(5, "view_count")
        cell_style = ParagraphStyle('Cell', parent=styles['Normal'], fontSize=8, leading=10)
        rows = [["Channel", "Video", "Views", "Likes", "Comments"]]
        for _, r in top.iterrows():
            rows.append([
                Paragraph(escape(clean_text(r["channel_name"], 22)), cell_style),
                Paragraph(escape(clean_text(r["title"], 60)), cell_style),
                f"{int(r['view_count']):,}", f"{int(r['like_count']):,}", f"{int(r['comment_count']):,}",
            ])
        eng = Table(rows, colWidths=[90, content_w - 90 - 3 * 62, 62, 62, 62], repeatRows=1)
        eng.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ]))
        story.append(KeepTogether([
            Paragraph("Top Performing Competitor Videos (Engagement Table)", chart_title_style), eng]))
    else:
        story.append(Paragraph("<i>No video records available yet.</i>", body_style))

    # scorecard (aspect-based sentiment per channel)
    score_tbl = ins.aspect_scorecard(aspects, data["comment_videos"], videos, channels)
    if not score_tbl.empty:
        story.append(Spacer(1, 6))
        story.append(KeepTogether(cell(
            "13. Aspect Scorecard: Net Sentiment per Channel", lambda: chart_scorecard(score_tbl),
            "% positive minus % negative of the comments about each aspect (+100 all positive, -100 all negative). One comment can count for several aspects.",
            content_w * 0.97)))

    # ------------------------------------------------------------------ Section 4: Whisper audio analysis
    story.append(PageBreak())
    story.append(Paragraph("4. Audio Analysis & Spoken Content: What the Competitors Say", heading_style))
    whisper_cnt = int((transcripts["source"] == "whisper").sum()) if not transcripts.empty else 0
    captions_cnt = int((transcripts["source"] == "captions").sum()) if not transcripts.empty else 0
    unavail_cnt = int((transcripts["source"] == "audio_unavailable").sum()) if not transcripts.empty else 0

    if whisper_cnt > 0:
        desc_s4 = ("The first minutes of each video's audio are streamed straight into memory and analysed by OpenAI Whisper - no audio file is ever stored, only the text. "
                   "Videos Whisper has not reached yet use YouTube captions as a stop-gap and are upgraded on later runs.")
    elif captions_cnt > 0:
        desc_s4 = (f"Audio transcription via Whisper is currently pending or restricted by platform access. YouTube captions ({captions_cnt} video(s)) "
                   "have been lawfully acquired as a validated fallback for content intelligence.")
    else:
        desc_s4 = ("Audio transcription via Whisper and captions are currently pending or restricted by platform access controls. "
                   "Metadata and comment analysis continue uninterrupted.")
    story.append(Paragraph(desc_s4, body_style))

    wsum = ins.whisper_summary(transcripts, videos)
    if wsum["videos"] > 0:
        kpi_w = [("VIDEOS ANALYSED BY WHISPER", f"{wsum['videos']} ({wsum['share']:.0f}%)"),
                 ("AUDIO ANALYSED", f"{wsum['minutes']:.0f} min"),
                 ("LANGUAGES DETECTED", f"{len(wsum['languages'])}"),
                 ("SPEAKING PACE", f"{wsum['pace']:.0f} wpm" if wsum["pace"] else "-")]
        tw = Table([[Paragraph(l, kpi_label) for l, _ in kpi_w], [Paragraph(v, kpi_value) for _, v in kpi_w]],
                   colWidths=[content_w / 4] * 4)
        tw.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")), ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")), ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(tw)
        story.append(Spacer(1, 8))
        wt = transcripts[transcripts["source"] == "whisper"].merge(videos[["video_id", "channel_id"]], on="video_id") \
                        .merge(channels[["channel_id", "channel_name"]], on="channel_id")
        wt = wt[wt["audio_seconds"].fillna(0) > 0]
        pace = (wt.groupby("channel_name")["word_count"].sum() / (wt.groupby("channel_name")["audio_seconds"].sum() / 60)) if not wt.empty else None
        story.append(pair(
            cell("14. Spoken Languages (auto-detected)", lambda: chart_languages(wsum),
                 "Language Whisper detected in the audio of each analysed video.", half_w),
            cell("15. Speaking Pace by Channel", lambda: chart_pace_by_channel(pace),
                 "Words per minute in the analysed audio - how dense each creator's delivery is.", half_w)
            if pace is not None else [Paragraph("<i>No pace data yet.</i>", chart_note_style)]))
        spoken = ins.spoken_topics(transcripts[transcripts["source"].isin(["whisper", "captions"])][["video_id", "transcript_text"]])
        if not spoken.empty:
            story.append(KeepTogether(cell(
                "16. Topics the Creators Talk About", lambda: chart_spoken_topics(spoken),
                "Phrases that recur across the competitors' transcripts - what the market already hears from them.", content_w * 0.97)))
    else:
        story.append(Paragraph(f"<i>No audio has been analysed by Whisper yet in this run. ({captions_cnt} video(s) analyzed via YouTube captions, {unavail_cnt} unavailable/restricted).</i>", body_style))
        if captions_cnt > 0:
            spoken = ins.spoken_topics(transcripts[transcripts["source"].isin(["whisper", "captions"])][["video_id", "transcript_text"]])
            if not spoken.empty:
                story.append(Spacer(1, 6))
                story.append(KeepTogether(cell(
                    "16. Topics the Creators Talk About (from YouTube Captions)", lambda: chart_spoken_topics(spoken),
                    "Phrases that recur across the competitors' caption transcripts - what the market already hears from them.", content_w * 0.97)))

    # ------------------------------------------------------------------ Section 5: relevance
    story.append(PageBreak())
    story.append(Paragraph("5. Content Relevance: What the Video Says vs. What Viewers Write", heading_style))
    if whisper_cnt > 0:
        rel_desc = (f"Video audio is transcribed with OpenAI Whisper ({whisper_cnt} video(s)) and YouTube captions ({captions_cnt} video(s)). "
                    "Transcript passages and viewer comments are embedded with a multilingual "
                    "language model and compared: <b>comment alignment</b> is how close comments are to some part of the video, <b>on-topic share</b> is the "
                    "fraction of comments clearly about the video's content, and <b>title alignment</b> checks that the transcript delivers what the title promises. "
                    "The index is a similarity measure - generic praise such as &quot;nice video&quot; naturally lowers it - not a quality verdict.")
    elif captions_cnt > 0:
        rel_desc = (f"Analysis utilizes YouTube captions ({captions_cnt} video(s)) and video titles/descriptions (Whisper audio acquisition pending/restricted). "
                    "Caption passages and viewer comments are embedded with a multilingual language model and compared: "
                    "<b>comment alignment</b> measures discussion correlation with captions, <b>on-topic share</b> reflects content relevance, "
                    "and <b>title alignment</b> measures title faithfulness. The index is a similarity measure, not a quality verdict.")
    else:
        rel_desc = ("Analysis currently utilizes video titles, descriptions, and viewer comments (transcripts pending or unavailable). "
                    "Embedded representations compare viewer feedback against stated video topics. The index is a similarity measure, not a quality verdict.")
    story.append(Paragraph(rel_desc, body_style))

    if not rel.empty:
        kpi2 = [("AVG RELEVANCE INDEX", f"{rel['relevance_index'].mean():.0f}/100"),
                ("AVG ON-TOPIC COMMENTS", f"{rel['on_topic_share'].mean() * 100:.0f}%"),
                ("VIDEOS SCORED", f"{len(rel)}"),
                ("WHISPER TRANSCRIPTS", f"{whisper_cnt}")]
        t2 = Table([[Paragraph(l, kpi_label) for l, _ in kpi2], [Paragraph(v, kpi_value) for _, v in kpi2]],
                   colWidths=[content_w / 4] * 4)
        t2.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(t2)
        story.append(Spacer(1, 8))
        story.append(pair(
            cell("17. Average Relevance Index by Channel", lambda: chart_relevance_by_channel(rel),
                 "Higher means that channel's audience discusses what its videos actually teach.", half_w),
            cell("18. Where the Video Text Comes From", lambda: chart_transcript_sources(videos, transcripts),
                 "Whisper audio transcription is preferred; captions/title are fallbacks and are upgraded on later runs.", half_w)))
        story.append(KeepTogether(cell(
            "19. Best- and Least-Aligned Videos", lambda: chart_relevance_extremes(rel),
            "Videos whose comments track the content most closely (green) and least closely (red).", content_w * 0.97)))
    else:
        story.append(Paragraph("<i>Relevance has not been computed yet - run the full pipeline.</i>", body_style))

    # ------------------------------------------------------------------ Section 5: topics for next video
    story.append(PageBreak())
    story.append(Paragraph("6. Recommended Topics for Your Next Video", heading_style))
    if whisper_cnt > 0:
        topics_desc = ("Based on OpenAI Whisper transcripts, YouTube captions, and viewer comments together: topics are ranked by "
                       "how many viewers mention them (liked comments weigh more), how many explicitly ask for them, how little the "
                       "existing transcripts already cover them, and the audience's sentiment around them.")
    elif captions_cnt > 0:
        topics_desc = ("Based on lawful YouTube captions and viewer comments together (Whisper transcripts pending/restricted): "
                       "topics are ranked by viewer demand, explicit requests, caption content coverage gaps, and audience sentiment.")
    else:
        topics_desc = ("Based on video titles, descriptions, and viewer comments together (audio transcripts pending/restricted): "
                       "candidate topics are mined from viewer requests and ranked by audience demand and sentiment.")
    story.append(Paragraph(topics_desc, body_style))
    topics_global = topics_all[topics_all["scope"] == "ALL"] if not topics_all.empty else topics_all
    if not topics_global.empty:
        story.append(KeepTogether(cell(
            "20. Opportunity Ranking of Candidate Topics", lambda: chart_topics(topics_global),
            "Score = 40% demand + 30% explicit requests + 20% content gap + 10% audience sentiment.", content_w * 0.97)))
        cs = ParagraphStyle('TopicCell', parent=styles['Normal'], fontSize=7.8, leading=9.6)
        rows = [["#", "Topic", "Score", "Mentions", "Requests", "Already covered", "Mood"]]
        for r in topics_global.head(10).itertuples(index=False):
            rows.append([str(r.rank_no), Paragraph(f"<b>{escape(clean_text(r.topic, 40))}</b>", cs), f"{r.score:.0f}",
                         f"{int(r.mentions)}", f"{int(r.request_count)}", f"{r.transcript_coverage * 100:.0f}% of videos",
                         f"{(r.avg_polarity + 1) / 2 * 100:.0f}% pos."])
        tt = Table(rows, colWidths=[22, content_w - 22 - 4 * 52 - 82, 52, 52, 52, 82, 52], repeatRows=1)
        tt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C05621")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (2, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFF7ED")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0"))]))
        story.append(Spacer(1, 4))
        story.append(KeepTogether([Paragraph("Top 10 topics - the numbers behind the ranking", chart_title_style), tt]))

        story.append(Spacer(1, 8))
        ev = [Paragraph("Why the top picks", chart_title_style)]
        for r in topics_global.head(3).itertuples(index=False):
            ev.append(Paragraph(f"<b>{r.rank_no}. {escape(clean_text(r.topic, 40))}</b> - {escape(r.evidence)}", body_style))
        story.append(KeepTogether(ev))

        per_channel = topics_all[topics_all["scope"] != "ALL"]
        if not per_channel.empty:
            crow = [["Channel", "Top topics viewers want next"]]
            for name, g in per_channel.groupby("channel_name"):
                crow.append([Paragraph(escape(clean_text(name, 30)), cs),
                             Paragraph(escape(", ".join(clean_text(x, 28) for x in g.sort_values("rank_no")["topic"].head(3))), cs)])
            ct_ = Table(crow, colWidths=[140, content_w - 140], repeatRows=1)
            ct_.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0"))]))
            story.append(Spacer(1, 8))
            story.append(KeepTogether([Paragraph("Per-competitor view: what each audience asks for", chart_title_style), ct_]))
    else:
        story.append(Paragraph("<i>No topic suggestions yet - run the full pipeline.</i>", body_style))

    # ------------------------------------------------------------------ Section 6: anomalies
    story.append(Paragraph("7. Engagement Anomaly Alerts (last 7 days)", heading_style))
    unit = {"views": "views/hr", "likes": "likes/hr", "comments": "comments/hr"}
    if not alerts.empty:
        ac = ParagraphStyle('AlertCell', parent=styles['Normal'], fontSize=7.8, leading=9.6)
        arows = [["Detected (UTC)", "Severity", "Channel", "Video", "Metric", "Rate", "x baseline"]]
        for r in alerts.head(10).itertuples(index=False):
            arows.append([f"{r.detected_at:%d %b %H:%M}", r.severity or "-", Paragraph(escape(clean_text(r.channel_name, 22)), ac),
                          Paragraph(escape(clean_text(r.title, 48)), ac), r.metric,
                          f"{r.velocity:,.0f} {unit.get(r.metric, '')}", f"{r.ratio:.1f}x"])
        at = Table(arows, colWidths=[76, 52, 66, content_w - 76 - 52 - 66 - 50 - 88 - 50, 50, 88, 50], repeatRows=1)
        at.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C53030")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFF5F5")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0"))]))
        story.append(at)
    else:
        story.append(Paragraph("No anomalies in the last 7 days - all tracked videos are within normal baseline ranges.", body_style))

    # ------------------------------------------------------------------ Section 7: recommendations
    story.append(Paragraph("8. Content Recommendations from Whisper Audio + Viewer Comments", heading_style))
    recs = ins.build_recommendations(data)
    rec_html = ("Based on what the competitors say (Whisper), what viewers write, aspect-level sentiment, comment relevance and engagement:<br/>"
                + "<br/>".join(
        f"{n}. <b>{escape(head)}</b> {escape(body)}" for n, (head, body) in enumerate(recs, 1)))
    story.append(Paragraph(rec_html, body_style))

    doc.build(story)
    print(f"PDF report generated: {PDF_PATH}")


if __name__ == "__main__":
    generate_professional_pdf()