"""MediaIntel Streamlit dashboard: Overview, Sentiment, Performance, Relevance & Next Topics,
Channel Comparison, Alerts and System Health. Reads the same database the pipeline writes."""

import importlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402
from core import channel_manager  # noqa: E402

# Ensure channel_manager module is refreshed if updated on Streamlit Cloud without container reboot
try:
    importlib.reload(channel_manager)
except Exception:
    pass

from core import insights as report_helpers  # noqa: E402  (scorecard / Whisper / topic helpers shared with the PDF)

LOGO = Path(__file__).resolve().parent / "assets" / "mediaintel_logo.png"

DB_NAME = config.db_label()
engine = config.get_engine()


@st.cache_resource(show_spinner=False)
def _ensure_schema_once():
    """Lets the dashboard open even before the first pipeline run (creates / upgrades tables, never deletes)."""
    try:
        from core import db_setup
        db_setup.ensure_schema(verbose=False)
    except Exception:
        pass
    return True


_ensure_schema_once()




@st.cache_data(ttl=30)
def load_data():
    """Tracked channels only; every table filtered to them."""
    channels_df = config.read_sql("SELECT * FROM channels WHERE is_tracked = TRUE")
    videos_df = config.read_sql("SELECT v.* FROM videos v JOIN channels c ON c.channel_id = v.channel_id WHERE c.is_tracked = TRUE")
    comments_df = config.read_sql("""SELECT m.comment_id, m.video_id FROM comments m JOIN videos v ON v.video_id = m.video_id
                                     JOIN channels c ON c.channel_id = v.channel_id WHERE c.is_tracked = TRUE""")
    insights_df = config.read_sql("""SELECT s.* FROM sentiment_insights s JOIN comments m ON m.comment_id = s.comment_id
                                     JOIN videos v ON v.video_id = m.video_id JOIN channels c ON c.channel_id = v.channel_id
                                     WHERE c.is_tracked = TRUE""")
    return channels_df, videos_df, comments_df, insights_df


@st.cache_data(ttl=30)
def load_extra():
    """Tables added by the Whisper / relevance / topics / anomaly stages (empty frames if not built yet)."""
    def safe(sql):
        try:
            return config.read_sql(sql)
        except Exception:
            return pd.DataFrame()
    return {
        "aspects": safe("SELECT comment_id, aspect, sentiment_label, polarity, confidence FROM aspect_sentiments"),
        "relevance": safe("""SELECT r.*, v.title, v.channel_id, c.channel_name FROM video_relevance r
                             JOIN videos v ON v.video_id = r.video_id JOIN channels c ON c.channel_id = v.channel_id
                             WHERE c.is_tracked = TRUE"""),
        "topics": safe("SELECT * FROM topic_suggestions ORDER BY scope, rank_no"),
        "transcripts": safe("SELECT video_id, source, language, model_name, word_count, audio_seconds, transcript_text FROM transcripts"),
        "alerts": safe("""SELECT a.id, a.detected_at, a.metric, a.velocity, a.baseline, a.ratio, a.severity, a.emailed, v.title, c.channel_name
                          FROM anomaly_alerts a JOIN videos v ON v.video_id = a.video_id
                          JOIN channels c ON c.channel_id = a.channel_id WHERE c.is_tracked = TRUE
                          ORDER BY a.detected_at DESC"""),
    }


channels, videos, comments, insights = load_data()
extra = load_extra()
aspects_df, relevance_df, topics_df = extra["aspects"], extra["relevance"], extra["topics"]
transcripts_df, alerts_df = extra["transcripts"], extra["alerts"]

# Bridge insights -> videos correctly via comments (comment_id -> video_id)
insights = insights.merge(comments, on="comment_id", how="left")

st.set_page_config(
    page_title="MediaIntel | Competitive Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME / CSS
# ---------------------------------------------------------------------------
ACCENT = "#8b5cf6"
PALETTE = ["#8b5cf6", "#22d3ee", "#f472b6", "#34d399", "#fbbf24", "#60a5fa"]
SENTIMENT_COLORS = {"Positive": "#34d399", "Negative": "#f87171", "Neutral": "#94a3b8"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

#MainMenu, footer {{ visibility: hidden; }}
header {{
    background: transparent !important;
}}
[data-testid="stSidebarCollapsedControl"] {{
    visibility: visible !important;
    display: flex !important;
}}


.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}}

/* ---- Header ---- */
.hero {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 22px 28px;
    border-radius: 16px;
    background: linear-gradient(120deg, rgba(139,92,246,0.18), rgba(34,211,238,0.10));
    border: 1px solid rgba(148,163,184,0.15);
    margin-bottom: 22px;
}}
.hero-title {{
    font-size: 26px;
    font-weight: 800;
    color: #f8fafc;
    margin: 0;
}}
.hero-subtitle {{
    font-size: 14px;
    color: #94a3b8;
    margin-top: 4px;
}}
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52, 211, 153, 0.12);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
}}
.dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 0 3px rgba(52,211,153,0.2);
}}

/* ---- KPI cards ---- */
.kpi-card {{
    background: #131a2a;
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: transform .15s ease, border-color .15s ease;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    border-color: rgba(139,92,246,0.45);
}}
.kpi-icon {{
    font-size: 20px;
    opacity: 0.85;
}}
.kpi-label {{
    color: #94a3b8;
    font-size: 12.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 10px;
}}
.kpi-value {{
    color: #f8fafc;
    font-size: 28px;
    font-weight: 800;
    margin-top: 2px;
}}
.kpi-accent {{
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}}

/* ---- Section title ---- */
.section-title {{
    font-size: 18px;
    font-weight: 700;
    color: #f1f5f9;
    margin: 6px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 14px !important;
}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background: #0e1421;
    border-right: 1px solid rgba(148,163,184,0.12);
}}
.sidebar-heading {{
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8b5cf6;
    margin: 4px 0 10px 0;
}}

/* ---- Tabs ---- */
button[data-baseweb="tab"] {{
    font-weight: 600;
    font-size: 14.5px;
}}

hr {{ border-color: rgba(148,163,184,0.12); }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
    margin=dict(l=10, r=10, t=30, b=10),
)


def style_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def kpi_card(icon, label, value, accent):
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-accent" style="background:{accent};"></div>
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def section_title(icon, text):
    st.markdown(f'<div class="section-title">{icon} {text}</div>', unsafe_allow_html=True)


def empty_state(msg):
    st.info(msg, icon="ℹ️")


# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <div>
        <p class="hero-title">📡 MediaIntel Command Center</p>
        <p class="hero-subtitle">Competitive video intelligence · RoBERTa sentiment mapping · Real-time engagement telemetry</p>
    </div>
    <div class="status-pill"><span class="dot"></span>Live · Updated {datetime.now().strftime('%H:%M:%S')}</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR — FILTERS
# ---------------------------------------------------------------------------
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=170)
    st.markdown('<div class="sidebar-heading">🎯 Data Scope & Filters</div>', unsafe_allow_html=True)

    data_scope = st.radio(
        "Analysis Scope",
        ["🎯 Active Analysis Set (10 Recent / Channel)", "🏛️ All-Time Warehouse Records"],
        index=0,
        help="Active Analysis Set filters to the deterministic 10 most recent videos per channel for this run. All-Time Warehouse shows historical accumulation."
    )

    selected_channel = st.selectbox(
        "Competitor channel",
        ["All Channels"] + sorted(channels["channel_name"].unique().tolist()),
    )

    sentiment_options = sorted(insights["sentiment_label"].dropna().unique().tolist())
    selected_sentiments = st.multiselect("Sentiment", sentiment_options, default=sentiment_options)

    aspect_options = sorted(insights["aspect_category"].dropna().unique().tolist())
    selected_aspects = st.multiselect("Aspect category", aspect_options, default=aspect_options)

    search_term = st.text_input("Search video title", placeholder="e.g. tutorial, review…")

    st.markdown("---")
    if st.button("↺ Reset filters", width="stretch"):
        st.rerun()

    st.markdown("---")
    st.caption(f"Connected to **{DB_NAME}**")
    st.caption(f"{len(channels)} channels · {len(videos):,} videos tracked")

# ---------------------------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------------------------
if "Active" in data_scope and not videos.empty:
    sorted_v = videos.sort_values(
        by=["channel_id", "published_at", "video_id"],
        ascending=[True, False, True]
    )
    base_videos = sorted_v.groupby("channel_id").head(10).copy()
else:
    base_videos = videos.copy()

filtered_videos = base_videos.copy()
if selected_channel != "All Channels":
    selected_cid = channels.loc[channels["channel_name"] == selected_channel, "channel_id"].values[0]
    filtered_videos = filtered_videos[filtered_videos["channel_id"] == selected_cid]

if search_term:
    filtered_videos = filtered_videos[filtered_videos["title"].str.contains(search_term, case=False, na=False)]

filtered_insights = insights[insights["video_id"].isin(filtered_videos["video_id"])]
if selected_sentiments:
    filtered_insights = filtered_insights[filtered_insights["sentiment_label"].isin(selected_sentiments)]
if selected_aspects:
    filtered_insights = filtered_insights[filtered_insights["aspect_category"].isin(selected_aspects)]

# ---------------------------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("📺", "Tracked Competitors", len(channels), PALETTE[0])
with k2:
    if "Active" in data_scope:
        kpi_card("🎬", "Active Audited Videos", f"{len(filtered_videos):,} <span style='font-size:12px;color:#94a3b8;font-weight:400;'>({len(videos)} in warehouse)</span>", PALETTE[1])
    else:
        kpi_card("🎬", "Warehouse Videos", f"{len(filtered_videos):,}", PALETTE[1])
with k3:
    if "Active" in data_scope:
        kpi_card("💬", "Active Audited Comments", f"{len(filtered_insights):,} <span style='font-size:12px;color:#94a3b8;font-weight:400;'>({len(insights)} in warehouse)</span>", PALETTE[2])
    else:
        kpi_card("💬", "Warehouse Comments", f"{len(filtered_insights):,}", PALETTE[2])
with k4:
    avg_conf = (filtered_insights["sentiment_score"].mean() * 100) if not filtered_insights.empty else 0
    kpi_card("🧠", "Mean AI Confidence", f"{avg_conf:.1f}%", PALETTE[3])

# Channel coverage warnings
if not channels.empty:
    present_cids = set(videos["channel_id"].dropna().unique()) if not videos.empty else set()
    zero_video_chans = [r.channel_name for r in channels.itertuples() if r.channel_id not in present_cids]
    if zero_video_chans:
        st.warning(f"⚠️ {len(zero_video_chans)} tracked channel(s) currently have no collected videos: {', '.join(zero_video_chans)}. Use 'Run Full Analysis' in System Health to ingest.")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_overview, tab_sentiment, tab_performance, tab_whisper, tab_relevance, tab_compare, tab_alerts, tab_system = st.tabs(
    ["📊 Overview", "🧠 Sentiment Intelligence", "📈 Video Performance", "🎙️ Whisper Audio Analysis",
     "🎯 Relevance & Next Topics", "⚔️ Channel Comparison", "🚨 Alerts", "🗄️ System Health"]
)

# ===================== OVERVIEW =====================
with tab_overview:
    oc1, oc2 = st.columns(2)
    with oc1:
        with st.container(border=True):
            section_title("🏆", "Top Videos by Views")
            if not filtered_videos.empty:
                top_views = filtered_videos.sort_values("view_count", ascending=True).tail(5)
                fig = px.bar(top_views, x="view_count", y="title", orientation="h",
                             color="view_count", color_continuous_scale=["#312e81", "#8b5cf6"])
                fig.update_layout(showlegend=False, yaxis_title="", coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig), width="stretch")
            else:
                empty_state("No video records match the current filters.")
    with oc2:
        with st.container(border=True):
            section_title("🎭", "Sentiment Split")
            if not filtered_insights.empty:
                sc = filtered_insights["sentiment_label"].value_counts().reset_index()
                sc.columns = ["Sentiment", "Total"]
                fig = px.pie(sc, names="Sentiment", values="Total", hole=0.6,
                             color="Sentiment", color_discrete_map=SENTIMENT_COLORS)
                st.plotly_chart(style_fig(fig), width="stretch")
            else:
                empty_state("No sentiment records match the current filters.")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        section_title("📋", "Filtered Video Explorer")
        if not filtered_videos.empty:
            st.dataframe(
                filtered_videos[["title", "view_count", "like_count", "comment_count", "published_at"]]
                .sort_values("view_count", ascending=False),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "⬇ Export filtered videos as CSV",
                filtered_videos.to_csv(index=False).encode("utf-8"),
                file_name="mediaintel_filtered_videos.csv",
                mime="text/csv",
            )
        else:
            empty_state("No video records match the current filters.")

    with st.expander("🎯 View Per-Channel Coverage Ledger (Active Analysis Set: Targets vs. Collected)", expanded=False):
        st.caption("Active run telemetry: Tracks the current analysis set (10 most recent videos, up to 2 transcription candidates). Historical warehouse records are preserved across runs.")
        try:
            cov_summary_df = channel_manager.get_channel_run_coverage_df()
            st.dataframe(cov_summary_df, width="stretch", hide_index=True)
        except Exception:
            pass

# ===================== SENTIMENT INTELLIGENCE =====================
with tab_sentiment:
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        with st.container(border=True):
            section_title("🧩", "Aspect-Based Feedback Distribution")
            if not filtered_insights.empty:
                aspect_counts = filtered_insights["aspect_category"].value_counts().reset_index()
                aspect_counts.columns = ["Aspect Category", "Count"]
                fig = px.bar(aspect_counts, x="Aspect Category", y="Count", color="Aspect Category",
                             color_discrete_sequence=PALETTE)
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig), width="stretch")
            else:
                empty_state("No aspect records available.")
    with r1c2:
        with st.container(border=True):
            section_title("⚖️", "Sentiment Polarity Breakdown")
            if not filtered_insights.empty:
                sentiment_counts = filtered_insights["sentiment_label"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Total"]
                fig = px.pie(sentiment_counts, names="Sentiment", values="Total", hole=0.5,
                             color="Sentiment", color_discrete_map=SENTIMENT_COLORS)
                st.plotly_chart(style_fig(fig), width="stretch")
            else:
                empty_state("No sentiment records available.")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        with st.container(border=True):
            section_title("🎯", "Avg AI Confidence by Sentiment Class")
            if not filtered_insights.empty:
                conf_df = filtered_insights.groupby("sentiment_label")["sentiment_score"].mean().reset_index()
                conf_df.columns = ["Sentiment", "Confidence Score"]
                conf_df["Confidence Score"] *= 100
                fig = px.bar(conf_df, x="Sentiment", y="Confidence Score", color="Sentiment",
                             color_discrete_map=SENTIMENT_COLORS)
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig), width="stretch")
            else:
                empty_state("No confidence metrics found.")
    with r2c2:
        with st.container(border=True):
            section_title("🔗", "Aspect-Level Sentiment (RoBERTa ABSA)")
            asp_f = aspects_df.merge(comments, on="comment_id", how="left") if not aspects_df.empty else aspects_df
            if not asp_f.empty:
                asp_f = asp_f[asp_f["video_id"].isin(filtered_videos["video_id"])]
            if not asp_f.empty:
                ct = asp_f.groupby(["aspect", "sentiment_label"]).size().reset_index(name="Comments")
                fig = px.bar(ct, x="aspect", y="Comments", color="sentiment_label", barmode="stack",
                             color_discrete_map=SENTIMENT_COLORS, labels={"aspect": "Aspect", "sentiment_label": "Sentiment"})
                fig.update_layout(xaxis_tickangle=-20)
                st.plotly_chart(style_fig(fig), width="stretch")
                st.caption("Each comment is split into clauses; every clause gets its own aspect and sentiment.")
            elif not filtered_insights.empty:
                st.dataframe(pd.crosstab(filtered_insights["aspect_category"], filtered_insights["sentiment_label"]),
                             width="stretch")
            else:
                empty_state("No crosstab telemetry available.")

    with st.container(border=True):
        section_title("🧮", "Aspect Scorecard — net sentiment per channel  (% positive − % negative)")
        score_tbl = report_helpers.aspect_scorecard(aspects_df, comments, videos, channels)
        if score_tbl.empty:
            empty_state("Run the pipeline to build the aspect scorecard.")
        else:
            fig = px.imshow(score_tbl, text_auto=True, aspect="auto", zmin=-100, zmax=100,
                            color_continuous_scale=["#f87171", "#1e293b", "#34d399"],
                            labels={"x": "Aspect", "y": "", "color": "Net sentiment"})
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig), width="stretch")
            st.caption("+100 = every comment about that aspect is positive, −100 = every one is negative. "
                       "One comment can contribute to several aspects.")

    with st.container(border=True):
        section_title("🔎", "Aspect Evidence Explorer — the exact words behind each aspect sentiment")
        if aspects_df.empty or "evidence" not in aspects_df:
            empty_state("No aspect evidence stored yet.")
        else:
            ev = aspects_df.merge(comments, on="comment_id", how="left").merge(
                videos[["video_id", "title", "channel_id"]], on="video_id", how="left").merge(
                channels[["channel_id", "channel_name"]], on="channel_id", how="left")
            ev = ev[ev["video_id"].isin(filtered_videos["video_id"])]
            e1, e2 = st.columns(2)
            with e1:
                asp_pick = st.selectbox("Aspect", sorted(ev["aspect"].dropna().unique().tolist()), key="ev_aspect")
            with e2:
                sen_pick = st.selectbox("Sentiment", ["Negative", "Neutral", "Positive"], index=0, key="ev_sent")
            sel = ev[(ev["aspect"] == asp_pick) & (ev["sentiment_label"] == sen_pick)].sort_values("confidence", ascending=False).head(15)
            if sel.empty:
                empty_state("No clauses for this combination.")
            else:
                st.dataframe(sel[["channel_name", "title", "evidence", "polarity", "confidence"]], width="stretch",
                             hide_index=True, column_config={
                    "channel_name": "Channel", "title": "Video", "evidence": "Clause (evidence)",
                    "polarity": st.column_config.NumberColumn("Polarity", format="%+.2f"),
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f")})

# ===================== VIDEO PERFORMANCE =====================
with tab_performance:
    p1c1, p1c2 = st.columns(2)
    with p1c1:
        with st.container(border=True):
            section_title("👁️", "Top Videos by View Count")
            if not filtered_videos.empty:
                top_views = filtered_videos.sort_values("view_count", ascending=True).tail(5)
                fig = px.bar(top_views, x="view_count", y="title", orientation="h",
                             color="view_count", color_continuous_scale=["#1e3a8a", "#60a5fa"])
                fig.update_layout(showlegend=False, yaxis_title="", coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig), width="stretch")
            else:
                empty_state("No video records available.")
    with p1c2:
        with st.container(border=True):
            section_title("👍", "Top Videos by Like Count")
            if not filtered_videos.empty:
                top_likes = filtered_videos.sort_values("like_count", ascending=True).tail(5)
                fig = px.bar(top_likes, x="like_count", y="title", orientation="h",
                             color="like_count", color_continuous_scale=["#065f46", "#34d399"])
                fig.update_layout(showlegend=False, yaxis_title="", coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig), width="stretch")
            else:
                empty_state("No video records available.")

    p2c1, p2c2 = st.columns(2)
    with p2c1:
        with st.container(border=True):
            section_title("📉", "View Count Distribution Trend")
            if not filtered_videos.empty:
                fig = px.line(filtered_videos.reset_index(), y="view_count", markers=True,
                               line_shape="spline", color_discrete_sequence=[ACCENT])
                fig.update_layout(xaxis_title="Video Index")
                st.plotly_chart(style_fig(fig), width="stretch")
            else:
                empty_state("No trend records available.")
    with p2c2:
        with st.container(border=True):
            section_title("💬", "Comment Volume Leaderboard")
            if not filtered_videos.empty:
                top_comms = filtered_videos.sort_values("comment_count", ascending=False).head(5)
                fig = px.bar(top_comms, x="title", y="comment_count", color="comment_count",
                             color_continuous_scale=["#581c87", "#c084fc"])
                fig.update_layout(showlegend=False, xaxis_tickangle=-25, coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig), width="stretch")
            else:
                empty_state("No comment telemetry available.")

# ===================== WHISPER AUDIO ANALYSIS =====================
with tab_whisper:
    section_title("🎙️", "Whisper Audio Analysis — what the competitors actually SAY in their videos")
    st.caption("The first minutes of each video's audio are streamed straight into memory and analysed by OpenAI Whisper — "
               "no audio file is ever saved, only the resulting text. Videos Whisper has not reached yet use YouTube captions "
               "as a stop-gap and are upgraded on later runs.")
    tx = filtered_videos.merge(transcripts_df, on="video_id", how="left") if not transcripts_df.empty else pd.DataFrame()
    summary = report_helpers.whisper_summary(
        transcripts_df[transcripts_df["video_id"].isin(filtered_videos["video_id"])] if not transcripts_df.empty else transcripts_df,
        filtered_videos)
    if tx.empty or summary["videos"] == 0 and transcripts_df.empty:
        empty_state("No audio has been analysed yet — run the pipeline (run.bat).")
    else:
        wk1, wk2, wk3, wk4 = st.columns(4)
        with wk1:
            kpi_card("🎬", "Videos analysed by Whisper", f"{summary['videos']}  ({summary['share']:.0f}%)", PALETTE[0])
        with wk2:
            kpi_card("⏱️", "Audio analysed", f"{summary['minutes']:.0f} min", PALETTE[1])
        with wk3:
            kpi_card("🌐", "Languages detected", f"{len(summary['languages'])}", PALETTE[2])
        with wk4:
            kpi_card("🗣️", "Speaking pace", f"{summary['pace']:.0f} wpm" if summary["pace"] else "—", PALETTE[3])
        st.markdown("<br>", unsafe_allow_html=True)

        wc1, wc2 = st.columns(2)
        with wc1:
            with st.container(border=True):
                section_title("🌐", "Spoken Languages (auto-detected by Whisper)")
                if summary["languages"]:
                    lang_df = pd.DataFrame({"Language": list(summary["languages"].keys()), "Videos": list(summary["languages"].values())})
                    fig = px.pie(lang_df, names="Language", values="Videos", hole=0.55, color_discrete_sequence=PALETTE)
                    st.plotly_chart(style_fig(fig), width="stretch")
                else:
                    empty_state("No Whisper transcripts yet.")
        with wc2:
            with st.container(border=True):
                section_title("📡", "Audio Coverage by Channel")
                if not tx.empty:
                    cov = tx.assign(whisper=(tx["source"] == "whisper")).groupby("channel_id").agg(
                        analysed=("whisper", "sum"), total=("video_id", "count")).reset_index().merge(
                        channels[["channel_id", "channel_name"]], on="channel_id")
                    cov["Not yet analysed by Whisper"] = cov["total"] - cov["analysed"]
                    fig = px.bar(cov.rename(columns={"analysed": "Analysed by Whisper"}), x="channel_name",
                                 y=["Analysed by Whisper", "Not yet analysed by Whisper"], barmode="stack",
                                 color_discrete_sequence=["#8b5cf6", "#475569"], labels={"channel_name": "", "value": "Videos"})
                    fig.update_layout(legend_title_text="")
                    st.plotly_chart(style_fig(fig), width="stretch")

        wt1, wt2 = st.columns(2)
        with wt1:
            with st.container(border=True):
                section_title("💬", "Topics the Creators Talk About (from the transcripts)")
                spoken = report_helpers.spoken_topics(tx[tx["source"].isin(["whisper", "captions"])][["video_id", "transcript_text"]]) if not tx.empty else pd.DataFrame()
                if spoken.empty:
                    empty_state("Not enough transcript text to extract topics yet.")
                else:
                    fig = px.bar(spoken.iloc[::-1], x="videos", y="topic", orientation="h", color="videos",
                                 color_continuous_scale=["#4c1d95", "#8b5cf6", "#22d3ee"], labels={"videos": "Videos mentioning it", "topic": ""})
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(style_fig(fig), width="stretch")
        with wt2:
            with st.container(border=True):
                section_title("🧭", "What Viewers Ask For vs. What Videos Cover")
                scope_w = "ALL" if selected_channel == "All Channels" else channels.loc[channels["channel_name"] == selected_channel, "channel_id"].values[0]
                tp_w = topics_df[topics_df["scope"] == scope_w].head(8) if not topics_df.empty else topics_df
                if tp_w.empty:
                    empty_state("Topic analysis has not run yet.")
                else:
                    cmp_df = pd.DataFrame({"Topic": list(tp_w["topic"]) * 2,
                                           "Share": list(tp_w["transcript_coverage"] * 100) + [100 - v * 100 for v in tp_w["transcript_coverage"]],
                                           "Type": ["Already covered in videos"] * len(tp_w) + ["Not covered yet (opportunity)"] * len(tp_w)})
                    fig = px.bar(cmp_df.iloc[::-1], x="Share", y="Topic", color="Type", orientation="h", barmode="stack",
                                 color_discrete_sequence=["#475569", "#f472b6"], labels={"Share": "% of videos", "Topic": ""})
                    fig.update_layout(legend_title_text="")
                    st.plotly_chart(style_fig(fig), width="stretch")

        with st.container(border=True):
            section_title("📜", "Transcript Viewer")
            have = tx[tx["transcript_text"].notna()] if not tx.empty else pd.DataFrame()
            if have.empty:
                empty_state("No transcripts stored yet.")
            else:
                pick = st.selectbox("Video", have["title"].tolist(), key="tx_pick")
                row = have[have["title"] == pick].iloc[0]
                mins = (row["audio_seconds"] or 0) / 60 if pd.notna(row["audio_seconds"]) else 0
                st.caption(f"Source: **{row['source']}** · model: {row['model_name']} · language: {row['language']} · "
                           f"{int(row['word_count'] or 0)} words · {mins:.1f} min of audio analysed")
                st.text_area("Transcript", str(row["transcript_text"] or ""), height=220)

# ===================== RELEVANCE & NEXT TOPICS =====================
with tab_relevance:
    section_title("🎯", "Content Relevance — what the video says vs. what viewers write")
    st.caption("Whisper transcript passages and viewer comments are embedded with a multilingual model and compared. "
               "The index (0–100) is a similarity measure: how closely the discussion follows the video's real content. "
               "Generic praise ('nice video') naturally lowers it; it is not a quality verdict.")

    rel_f = relevance_df[relevance_df["video_id"].isin(filtered_videos["video_id"])] if not relevance_df.empty else relevance_df
    if rel_f.empty:
        empty_state("Relevance has not been computed yet — run the full pipeline (run.bat).")
    else:
        src_by_video = transcripts_df.set_index("video_id")["source"] if not transcripts_df.empty else pd.Series(dtype=str)
        fv_src = filtered_videos["video_id"].map(src_by_video).fillna("none").replace({"failed": "none"})
        whisper_pct = 100 * (fv_src == "whisper").mean() if len(fv_src) else 0

        rk1, rk2, rk3, rk4 = st.columns(4)
        with rk1:
            kpi_card("🎯", "Avg Relevance Index", f"{rel_f['relevance_index'].mean():.0f}/100", PALETTE[0])
        with rk2:
            kpi_card("🗣️", "On-Topic Comments", f"{rel_f['on_topic_share'].mean() * 100:.0f}%", PALETTE[1])
        with rk3:
            kpi_card("🎬", "Videos Scored", f"{len(rel_f)}", PALETTE[2])
        with rk4:
            kpi_card("🎙️", "Whisper Coverage", f"{whisper_pct:.0f}%", PALETTE[3])
        st.markdown("<br>", unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            with st.container(border=True):
                section_title("📡", "Average Relevance by Channel")
                by_ch = rel_f.groupby("channel_name")["relevance_index"].mean().reset_index().sort_values("relevance_index", ascending=False)
                fig = px.bar(by_ch, x="channel_name", y="relevance_index", color="channel_name",
                             color_discrete_sequence=PALETTE, labels={"channel_name": "", "relevance_index": "Relevance index"})
                fig.update_layout(showlegend=False, yaxis_range=[0, 100])
                st.plotly_chart(style_fig(fig), width="stretch")
        with rc2:
            with st.container(border=True):
                section_title("🎙️", "Where the Video Text Comes From")
                names = {"whisper": "Whisper (audio)", "captions": "YouTube captions", "none": "Title only"}
                sc = fv_src.map(names).value_counts().reset_index()
                sc.columns = ["Source", "Videos"]
                fig = px.pie(sc, names="Source", values="Videos", hole=0.55, color="Source",
                             color_discrete_map={"Whisper (audio)": "#8b5cf6", "YouTube captions": "#60a5fa", "Title only": "#94a3b8"})
                st.plotly_chart(style_fig(fig), width="stretch")
                st.caption("Whisper is preferred; videos beyond the time budget use captions and are upgraded on later runs.")

        rc3, rc4 = st.columns(2)
        with rc3:
            with st.container(border=True):
                section_title("🏅", "Best- and Least-Aligned Videos")
                ordered = rel_f.sort_values("relevance_index")
                k = min(5, max(len(ordered) // 2, 1))
                ext = pd.concat([ordered.head(k), ordered.tail(k)]).drop_duplicates("video_id")
                ext["label"] = ext["channel_name"].str[:12] + " · " + ext["title"].str[:42]
                fig = px.bar(ext, x="relevance_index", y="label", orientation="h", color="relevance_index",
                             color_continuous_scale=["#f87171", "#fbbf24", "#34d399"], range_color=[0, 100])
                fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Relevance index")
                st.plotly_chart(style_fig(fig), width="stretch")
        with rc4:
            with st.container(border=True):
                section_title("🔬", "Relevance vs. Audience Sentiment")
                vid_pol = insights.groupby("video_id")["polarity"].mean().rename("avg_polarity") if "polarity" in insights else pd.Series(dtype=float)
                sc_df = rel_f.merge(vid_pol, on="video_id", how="left").dropna(subset=["avg_polarity"])
                if not sc_df.empty:
                    fig = px.scatter(sc_df, x="relevance_index", y="avg_polarity", size="n_comments", color="channel_name",
                                     hover_name="title", color_discrete_sequence=PALETTE,
                                     labels={"relevance_index": "Relevance index", "avg_polarity": "Avg sentiment (−1…+1)"})
                    st.plotly_chart(style_fig(fig), width="stretch")
                else:
                    empty_state("Sentiment polarity not available yet.")

        with st.container(border=True):
            section_title("📋", "Per-Video Relevance Breakdown")
            tbl = rel_f[["channel_name", "title", "relevance_index", "comment_alignment", "on_topic_share",
                         "title_alignment", "n_comments", "text_source"]].sort_values("relevance_index", ascending=False)
            tbl["on_topic_share"] = tbl["on_topic_share"] * 100
            st.dataframe(tbl, width="stretch", hide_index=True, column_config={
                "channel_name": "Channel", "title": "Video",
                "relevance_index": st.column_config.ProgressColumn("Relevance index", min_value=0, max_value=100, format="%.0f"),
                "comment_alignment": st.column_config.NumberColumn("Comment alignment", format="%.2f"),
                "on_topic_share": st.column_config.NumberColumn("On-topic share", format="%.0f%%"),
                "title_alignment": st.column_config.NumberColumn("Title alignment", format="%.2f"),
                "n_comments": "Comments", "text_source": "Text source"})

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("💡", "Topics for Your Next Video  ·  from Whisper transcripts + comments")
    scope_id = "ALL"
    if selected_channel != "All Channels":
        scope_id = channels.loc[channels["channel_name"] == selected_channel, "channel_id"].values[0]
    tp = topics_df[topics_df["scope"] == scope_id] if not topics_df.empty else topics_df
    st.caption(f"Scope: **{selected_channel}** · Score = 40% demand + 30% explicit requests + 20% content gap + 10% audience sentiment.")
    if tp.empty:
        empty_state("No topic suggestions yet — run the full pipeline (run.bat).")
    else:
        tc1, tc2 = st.columns([3, 2])
        with tc1:
            with st.container(border=True):
                section_title("🏁", "Opportunity Ranking")
                top10 = tp.head(10).iloc[::-1]
                fig = px.bar(top10, x="score", y="topic", orientation="h", color="score",
                             color_continuous_scale=["#4c1d95", "#f472b6", "#fbbf24"],
                             hover_data={"mentions": True, "request_count": True, "transcript_coverage": ":.0%"})
                fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Opportunity score (0–100)")
                st.plotly_chart(style_fig(fig), width="stretch")
        with tc2:
            with st.container(border=True):
                section_title("🧭", "Demand vs. Existing Coverage")
                fig = px.scatter(tp, x="mentions", y=tp["transcript_coverage"] * 100, size="score", text="topic",
                                 color="score", color_continuous_scale=["#4c1d95", "#f472b6", "#fbbf24"],
                                 labels={"mentions": "Viewer mentions", "y": "Covered in videos (%)"})
                fig.update_traces(textposition="top center")
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig), width="stretch")
                st.caption("Bottom-right = loud demand, little existing coverage → your biggest opportunity.")

        with st.container(border=True):
            section_title("🧾", "Ranked Topics & Evidence")
            tp_show = tp[["rank_no", "topic", "score", "mentions", "request_count", "transcript_coverage", "evidence"]].copy()
            tp_show["transcript_coverage"] = tp_show["transcript_coverage"] * 100
            st.dataframe(tp_show, width="stretch", hide_index=True, column_config={
                "rank_no": "#", "topic": "Topic",
                "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                "mentions": "Mentions", "request_count": "Requests",
                "transcript_coverage": st.column_config.NumberColumn("Already covered", format="%.0f%%"),
                "evidence": "Evidence"})


# ===================== CHANNEL COMPARISON =====================
with tab_compare:
    section_title("⚔️", "Competitor Comparison — all tracked channels side by side")
    st.caption("This tab always compares every channel (the sidebar channel filter does not apply).")
    if videos.empty:
        empty_state("No video data yet.")
    else:
        v = videos.merge(channels[["channel_id", "channel_name"]], on="channel_id")
        agg = v.groupby("channel_name").agg(Videos=("video_id", "count"), AvgViews=("view_count", "mean"),
                                           AvgLikes=("like_count", "mean"), AvgComments=("comment_count", "mean"),
                                           TotalViews=("view_count", "sum")).reset_index()
        agg["EngagementRate"] = (agg["AvgLikes"] + agg["AvgComments"]) / agg["AvgViews"].replace(0, pd.NA) * 100
        ins_c = insights.merge(videos[["video_id", "channel_id"]], on="video_id", how="left", suffixes=("", "_v"))
        ins_c = ins_c.merge(channels[["channel_id", "channel_name"]], left_on="channel_id", right_on="channel_id", how="left") \
            if "channel_id" in ins_c else ins_c
        if not ins_c.empty and "channel_name" in ins_c:
            mix = ins_c.groupby(["channel_name", "sentiment_label"]).size().reset_index(name="n")
            mix["share"] = mix["n"] / mix.groupby("channel_name")["n"].transform("sum") * 100
            pos = mix[mix["sentiment_label"] == "Positive"].set_index("channel_name")["share"]
            agg["PositiveShare"] = agg["channel_name"].map(pos).fillna(0)
        if not relevance_df.empty:
            agg["Relevance"] = agg["channel_name"].map(relevance_df.groupby("channel_name")["relevance_index"].mean())

        cc1, cc2 = st.columns(2)
        with cc1:
            with st.container(border=True):
                section_title("👁️", "Average Views per Video")
                fig = px.bar(agg.sort_values("AvgViews", ascending=False), x="channel_name", y="AvgViews", color="channel_name",
                             color_discrete_sequence=PALETTE, labels={"channel_name": "", "AvgViews": "Avg views"})
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig), width="stretch")
        with cc2:
            with st.container(border=True):
                section_title("🔥", "Engagement Rate  (likes + comments) / views")
                fig = px.bar(agg.sort_values("EngagementRate", ascending=False), x="channel_name", y="EngagementRate",
                             color="channel_name", color_discrete_sequence=PALETTE,
                             labels={"channel_name": "", "EngagementRate": "Engagement rate (%)"})
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig), width="stretch")
        cc3, cc4 = st.columns(2)
        with cc3:
            with st.container(border=True):
                section_title("🎭", "Audience Sentiment Mix")
                if not ins_c.empty and "channel_name" in ins_c:
                    fig = px.bar(mix, x="channel_name", y="share", color="sentiment_label", barmode="stack",
                                 color_discrete_map=SENTIMENT_COLORS, labels={"channel_name": "", "share": "% of comments", "sentiment_label": "Sentiment"})
                    st.plotly_chart(style_fig(fig), width="stretch")
                else:
                    empty_state("No sentiment data yet.")
        with cc4:
            with st.container(border=True):
                section_title("🕸️", "Strength Profile (normalised 0–100)")
                metrics = [c for c in ["AvgViews", "EngagementRate", "PositiveShare", "Relevance"] if c in agg.columns]
                labels_ = {"AvgViews": "Reach", "EngagementRate": "Engagement", "PositiveShare": "Positivity", "Relevance": "Relevance"}
                fig = go.Figure()
                for i, r in agg.iterrows():
                    vals = []
                    for m in metrics:
                        mx = agg[m].max()
                        vals.append(0 if not mx or pd.isna(r[m]) else 100 * float(r[m]) / float(mx))
                    fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=[labels_[m] for m in metrics] + [labels_[metrics[0]]],
                                                  fill="toself", name=r["channel_name"], opacity=0.55,
                                                  line=dict(color=PALETTE[i % len(PALETTE)])))
                fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], gridcolor="#334155"), bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(style_fig(fig), width="stretch")
        with st.container(border=True):
            section_title("📋", "Comparison Table")
            show = agg.rename(columns={"channel_name": "Channel", "AvgViews": "Avg views", "AvgLikes": "Avg likes",
                                       "AvgComments": "Avg comments", "TotalViews": "Total views",
                                       "EngagementRate": "Engagement %", "PositiveShare": "Positive %", "Relevance": "Relevance idx"})
            st.dataframe(show.round(1), width="stretch", hide_index=True)

# ===================== ALERTS =====================
with tab_alerts:
    section_title("🚨", "Engagement Anomaly Alerts — views · likes · comments")
    st.caption("A metric is flagged when its hourly rate exceeds 3× the channel's own baseline "
               "(MEDIUM ≥ 3×, HIGH ≥ 5×, CRITICAL ≥ 10×). The RPA bot e-mails new alerts with the PDF report attached.")

    with st.container(border=True):
        section_title("⚡", "Live Velocity vs. Channel Baseline  (views per hour — how close is each video to a spike?)")
        vv = filtered_videos.dropna(subset=["published_at"]).copy()
        if vv.empty:
            empty_state("No videos to measure yet.")
        else:
            hrs = ((pd.Timestamp.now("UTC").tz_localize(None) - pd.to_datetime(vv["published_at"])).dt.total_seconds() / 3600).clip(lower=1)
            vv["views_per_hr"] = vv["view_count"] / hrs
            vv["baseline"] = vv.groupby("channel_id")["views_per_hr"].transform("mean")
            vv["ratio"] = vv["views_per_hr"] / vv["baseline"].replace(0, pd.NA)
            vv = vv.merge(channels[["channel_id", "channel_name"]], on="channel_id", how="left")
            top_v = vv.sort_values("ratio", ascending=False).head(12).copy()
            top_v["label"] = top_v["channel_name"].str[:12] + " · " + top_v["title"].str[:34]
            top_v["status"] = top_v["ratio"].apply(lambda r: "CRITICAL" if r >= 10 else "HIGH" if r >= 5 else "MEDIUM" if r >= 3 else "Normal")
            fig = px.bar(top_v.iloc[::-1], x="ratio", y="label", orientation="h", color="status",
                         color_discrete_map={"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#fbbf24", "Normal": "#64748b"},
                         labels={"ratio": "× channel baseline", "label": ""})
            fig.add_vline(x=3, line_dash="dash", line_color="#94a3b8", annotation_text="alert threshold (3×)")
            st.plotly_chart(style_fig(fig), width="stretch")

    if alerts_df.empty:
        st.success("No anomalies recorded yet — every tracked video is within its normal baseline range.", icon="✅")
    else:
        metric_pick = st.multiselect("Metric", ["views", "likes", "comments"], default=["views", "likes", "comments"], key="al_metric")
        al = alerts_df[alerts_df["metric"].isin(metric_pick)]
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            kpi_card("🚨", "Alerts (all time)", f"{len(al)}", "#f87171")
        with a2:
            kpi_card("🔥", "High / Critical", f"{int(al['severity'].isin(['HIGH', 'CRITICAL']).sum())}", "#f97316")
        with a3:
            kpi_card("📧", "E-mailed", f"{int((al['emailed'] == 1).sum())}", "#34d399")
        with a4:
            kpi_card("⏳", "Waiting to send", f"{int((al['emailed'] == 0).sum())}", "#fbbf24")
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            section_title("📈", "Strongest Spikes (× channel baseline)")
            top_al = al.sort_values("ratio", ascending=False).head(10).copy()
            top_al["label"] = top_al["channel_name"].str[:12] + " · " + top_al["title"].str[:36] + " [" + top_al["metric"] + "]"
            fig = px.bar(top_al.iloc[::-1], x="ratio", y="label", orientation="h", color="metric",
                         color_discrete_sequence=PALETTE, labels={"ratio": "× baseline", "label": ""})
            st.plotly_chart(style_fig(fig), width="stretch")
        with st.container(border=True):
            section_title("🧾", "Alert Ledger")
            led = al[["detected_at", "severity", "channel_name", "title", "metric", "velocity", "baseline", "ratio", "emailed"]].copy()
            led["emailed"] = led["emailed"].map({1: "✅ sent", 0: "⏳ pending"})
            st.dataframe(led, width="stretch", hide_index=True, column_config={
                "detected_at": "Detected (UTC)", "severity": "Severity", "channel_name": "Channel", "title": "Video", "metric": "Metric",
                "velocity": st.column_config.NumberColumn("Rate /hr", format="%.1f"),
                "baseline": st.column_config.NumberColumn("Baseline /hr", format="%.1f"),
                "ratio": st.column_config.NumberColumn("× baseline", format="%.1f"), "emailed": "E-mail"})

# ===================== SYSTEM HEALTH =====================
with tab_system:
    s1, s2, s3 = st.columns(3)
    with s1:
        with st.container(border=True):
            section_title("📺", "Tracked Channel Registry")
            st.dataframe(channels[["channel_name", "is_tracked"]], width="stretch", hide_index=True)
    with s2:
        with st.container(border=True):
            section_title("🧾", "Sample Processed Insights")
            if not insights.empty:
                st.dataframe(
                    insights[["comment_id", "sentiment_label", "aspect_category", "model_name"]].head(5),
                    width="stretch", hide_index=True,
                )
            else:
                empty_state("No log entries available.")
    with s3:
        with st.container(border=True):
            section_title("🩺", "Warehouse Health")
            st.metric("Channels Partition", len(channels))
            st.metric("Videos Partition", len(videos))
            st.metric("Insights Partition", len(insights))
            st.metric("Aspect Rows (ABSA)", len(aspects_df))
            wh_tx_cnt = int((transcripts_df["source"] == "whisper").sum()) if not transcripts_df.empty else 0
            cap_tx_cnt = int((transcripts_df["source"] == "captions").sum()) if not transcripts_df.empty else 0
            unavail_tx_cnt = int((transcripts_df["source"] == "audio_unavailable").sum()) if not transcripts_df.empty else 0
            st.metric("Whisper Transcripts", wh_tx_cnt)
            st.metric("Captions Transcripts", cap_tx_cnt)
            if unavail_tx_cnt > 0:
                st.metric("Audio Unavailable Records", unavail_tx_cnt)
            st.metric("Topic Suggestions", len(topics_df))

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        section_title("⚙️", "Online Channel & Ingestion Management")
        st.caption("Secure administrator controls for competitor onboarding, tracking toggles, and cloud ingestion telemetry.")

        import hmac
        import re

        admin_secret = (
            config.env("MEDIAINTEL_ADMIN_PASSWORD")
            or config.env("ADMIN_PASSWORD")
        )
        if not admin_secret:
            try:
                admin_secret = st.secrets.get("MEDIAINTEL_ADMIN_PASSWORD") or st.secrets.get("ADMIN_PASSWORD")
            except Exception:
                pass

        if not st.session_state.get("admin_authenticated", False):
            st.markdown("#### 🔒 Administrator Sign In")
            st.caption("Authentication is required to modify channel tracking or add new channels.")
            col_pwd, col_btn = st.columns([3, 1])
            with col_pwd:
                entered_pwd = st.text_input(
                    "Admin Password",
                    type="password",
                    placeholder="Enter server administrator password…",
                    label_visibility="collapsed",
                    key="admin_pwd_field",
                )
            with col_btn:
                if st.button("🔑 Sign In", width="stretch"):
                    if not admin_secret:
                        st.warning("Server administrator password is not set (configure MEDIAINTEL_ADMIN_PASSWORD).")
                    elif hmac.compare_digest(entered_pwd.strip().encode("utf-8"), admin_secret.strip().encode("utf-8")):
                        st.session_state["admin_authenticated"] = True
                        st.rerun()
                    else:
                        st.error("Invalid administrator password.")
        else:
            top_bar_1, top_bar_2 = st.columns([4, 1])
            with top_bar_1:
                st.markdown(
                    '<span class="status-pill"><span class="dot"></span> Authenticated as Administrator</span>',
                    unsafe_allow_html=True,
                )
            with top_bar_2:
                if st.button("🚪 Sign Out", width="stretch"):
                    st.session_state["admin_authenticated"] = False
                    st.rerun()

            st.markdown("<hr>", unsafe_allow_html=True)

            gh_repo = config.env("GITHUB_REPOSITORY") or "Harinath2112/MediaIntel"
            gh_token = config.env("GITHUB_TOKEN") or config.env("GH_PAT")
            if not gh_token:
                try:
                    gh_token = st.secrets.get("GITHUB_TOKEN") or st.secrets.get("GH_PAT")
                except Exception:
                    pass

            # -----------------------------------------------------------
            # Section 1: Channel Onboarding & Tracking Controls
            # -----------------------------------------------------------
            try:
                all_channels_df = config.read_sql("SELECT channel_id, channel_name, is_tracked FROM channels ORDER BY is_tracked DESC, channel_name ASC")
            except Exception:
                all_channels_df = pd.DataFrame(columns=["channel_id", "channel_name", "is_tracked"])

            col_onboard, col_tracking = st.columns([3, 2])

            with col_onboard:
                st.markdown("##### ➕ Competitor Channel Onboarding")
                subtab_single, subtab_batch = st.tabs(["➕ Single Channel", "📑 Batch Onboard (2–5+ Channels)"])

                with subtab_single:
                    new_cid_input = st.text_input("Channel ID, URL, or @Handle", placeholder="e.g. UCJskGeByzRRSvmOyZOz61ig, @ApnaCollegeOfficial, https://youtube.com/...", key="add_cid_in")
                    new_cname_input = st.text_input("Display Label (Optional)", placeholder="e.g. take U forward", key="add_cname_in")

                    if st.button("Register & Track Channel", width="stretch", key="btn_add_chan_single"):
                        yt_key = config.env("YOUTUBE_API_KEY") or (st.secrets.get("YOUTUBE_API_KEY") if hasattr(st, "secrets") else None)
                        cid, resolved_name, err = channel_manager.resolve_channel_identifier(new_cid_input, api_key=yt_key)
                        if err or not cid:
                            st.error(err or "Invalid YouTube channel identifier.")
                        else:
                            label = new_cname_input.strip() or resolved_name or cid
                            res = channel_manager.register_single_channel(cid, channel_name=label, is_tracked=True)
                            if res["success"]:
                                st.cache_data.clear()
                                st.success(res["message"])
                                st.rerun()
                            else:
                                st.error(res["message"])

                with subtab_batch:
                    st.caption("Paste multiple channels (one per line, comma-separated, or JSON array). Example: `UC... - Channel Name` or `@handle` or full URL.")
                    batch_text = st.text_area(
                        "Batch Channel Entries",
                        placeholder="UCJskGeByzRRSvmOyZOz61ig - take U forward\nhttps://www.youtube.com/channel/UCBwmMxybNva6P_5VmxjzwqA - Apna College\n@GateSmashers\n@CodeWithHarry",
                        height=130,
                        key="add_batch_text_in"
                    )

                    b_c1, b_c2 = st.columns([2, 1])
                    with b_c1:
                        track_default = st.checkbox("Track channels immediately upon registration", value=True, key="batch_track_def")
                    with b_c2:
                        btn_batch = st.button("🚀 Register Batch", width="stretch", key="btn_add_chan_batch")

                    if btn_batch:
                        if not batch_text.strip():
                            st.warning("Please enter at least one channel to register.")
                        else:
                            yt_key = config.env("YOUTUBE_API_KEY") or (st.secrets.get("YOUTUBE_API_KEY") if hasattr(st, "secrets") else None)
                            batch_results = channel_manager.register_batch_channels(batch_text, api_key=yt_key, default_tracked=track_default)
                            if not batch_results:
                                st.warning("No valid channel identifiers found in input.")
                            else:
                                success_count = sum(1 for r in batch_results if r["success"])
                                st.cache_data.clear()
                                if success_count == len(batch_results):
                                    st.success(f"Successfully processed all {success_count} channel(s)!")
                                else:
                                    st.info(f"Processed {len(batch_results)} entries: {success_count} succeeded, {len(batch_results) - success_count} failed.")

                                with st.expander("View Batch Registration Ledger", expanded=True):
                                    for r in batch_results:
                                        icon = "✅" if r["success"] else "⚠️"
                                        st.markdown(f"- {icon} **{r.get('channel_name') or r.get('raw')}**: {r.get('message')}")
                                st.rerun()

            with col_tracking:
                st.markdown("##### 🔀 Channel Tracking Controls")
                if not all_channels_df.empty:
                    channel_map = {
                        f"{r.channel_name} ({'Tracked' if r.is_tracked else 'Untracked'})": (r.channel_id, bool(r.is_tracked))
                        for r in all_channels_df.itertuples()
                    }
                    selected_toggle_label = st.selectbox("Select Channel", list(channel_map.keys()), key="toggle_sel_ch")
                    target_cid, current_state = channel_map[selected_toggle_label]
                    new_state = not current_state
                    btn_label = "🚫 Disable Tracking" if current_state else "✅ Enable Tracking"

                    if st.button(btn_label, width="stretch", key="btn_toggle_track"):
                        if channel_manager.set_channel_tracking(target_cid, new_state):
                            st.cache_data.clear()
                            st.success(f"Updated {target_cid} to {'Tracked' if new_state else 'Untracked'}.")
                            st.rerun()
                        else:
                            st.error("Failed to update tracking status.")
                    st.caption("ℹ️ Disabled channels are excluded from cloud ingestion. Historical videos, comments, and analytics are preserved.")
                else:
                    st.info("No channels in database yet.")

            # Section 1.5: Permanent Channel Deletion (Danger Zone - Admin Only)
            if not all_channels_df.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("⚠️ Danger Zone: Delete Channel Permanently", expanded=False):
                    st.markdown("##### 🗑️ Delete Channel Permanently")
                    st.caption("Permanently purge a competitor channel and all its dependent videos, comments, RoBERTa sentiment scores, transcripts, and storage assets from the database.")

                    del_channel_map = {
                        f"{r.channel_name} (ID: {r.channel_id})": (r.channel_id, r.channel_name)
                        for r in all_channels_df.itertuples()
                    }
                    sel_del_label = st.selectbox(
                        "Select Channel to Delete",
                        list(del_channel_map.keys()),
                        key="del_sel_ch",
                        help="Choose the competitor channel you wish to permanently purge."
                    )
                    del_cid, del_cname = del_channel_map[sel_del_label]

                    # Concurrency check & preview with dynamic module refresh protection
                    if not hasattr(channel_manager, "preview_channel_deletion"):
                        try:
                            importlib.reload(channel_manager)
                        except Exception:
                            pass

                    del_preview = channel_manager.preview_channel_deletion(del_cid, gh_token=gh_token)

                    st.markdown(f"**Target Channel:** `{del_cname}` &nbsp;&nbsp;|&nbsp;&nbsp; **YouTube Channel ID:** `{del_cid}`")

                    # Preview record counts
                    p_counts = del_preview.get("record_counts", {})
                    completed_tx = del_preview.get("completed_transcripts", p_counts.get("transcripts", 0))
                    pending_tx = del_preview.get("pending_audio_records", 0)
                    active_tx = del_preview.get("active_set_transcripts", completed_tx)
                    tx_help = f"Completed Transcripts: {completed_tx} (Active Set: {active_tx}/2) | Pending Audio/Unavailable Records: {pending_tx} | Total Table Rows to Purge: {p_counts.get('transcripts', 0)}"

                    dc1, dc2, dc3, dc4 = st.columns(4)
                    with dc1:
                        st.metric("Videos (Warehouse)", f"{p_counts.get('videos', 0):,}")
                        st.metric("Comments", f"{p_counts.get('comments', 0):,}")
                    with dc2:
                        st.metric("Sentiment Insights", f"{p_counts.get('sentiment_insights', 0):,}")
                        st.metric("Aspect Records", f"{p_counts.get('aspect_sentiments', 0):,}")
                    with dc3:
                        if pending_tx > 0:
                            st.metric(
                                "Transcripts (Completed)",
                                f"{completed_tx:,}",
                                delta=f"{pending_tx} audio pending",
                                delta_color="inverse",
                                help=tx_help,
                            )
                        else:
                            st.metric(
                                "Transcripts (Warehouse)",
                                f"{completed_tx:,}",
                                help=tx_help,
                            )
                        st.metric("Audio Storage Objects", f"{del_preview.get('storage_audio_count', 0):,}")
                    with dc4:
                        st.metric("Relevance & Alerts", f"{p_counts.get('video_relevance', 0) + p_counts.get('anomaly_alerts', 0):,}")
                        st.metric("Total Dependent Records", f"{del_preview.get('total_records', 0):,}")

                    with st.expander("📋 Detailed Relational Table Breakdown", expanded=False):
                        breakdown_data = [
                            {"Table": "channels", "Records to Purge": p_counts.get("channels", 1), "Scope / Notes": "Channel metadata"},
                            {"Table": "videos", "Records to Purge": p_counts.get("videos", 0), "Scope / Notes": "All stored videos in warehouse"},
                            {"Table": "comments", "Records to Purge": p_counts.get("comments", 0), "Scope / Notes": "All collected comments"},
                            {"Table": "sentiment_insights", "Records to Purge": p_counts.get("sentiment_insights", 0), "Scope / Notes": "RoBERTa sentiment mappings"},
                            {"Table": "aspect_sentiments", "Records to Purge": p_counts.get("aspect_sentiments", 0), "Scope / Notes": "ABSA aspect rows"},
                            {"Table": "transcripts", "Records to Purge": p_counts.get("transcripts", 0), "Scope / Notes": f"{completed_tx} completed, {pending_tx} audio pending/unavailable"},
                            {"Table": "video_audio_assets", "Records to Purge": p_counts.get("video_audio_assets", 0), "Scope / Notes": "Storage asset mappings"},
                            {"Table": "video_stats_history", "Records to Purge": p_counts.get("video_stats_history", 0), "Scope / Notes": "Engagement velocity history"},
                            {"Table": "video_relevance", "Records to Purge": p_counts.get("video_relevance", 0), "Scope / Notes": "Relevance & alignment scores"},
                            {"Table": "anomaly_alerts", "Records to Purge": p_counts.get("anomaly_alerts", 0), "Scope / Notes": "Channel/video anomalies"},
                            {"Table": "topic_suggestions", "Records to Purge": p_counts.get("topic_suggestions", 0), "Scope / Notes": "Channel-scoped topic suggestions"},
                            {"Table": "channel_run_coverage", "Records to Purge": p_counts.get("channel_run_coverage", 0), "Scope / Notes": "Coverage ledger telemetry"},
                        ]
                        st.dataframe(pd.DataFrame(breakdown_data), width="stretch", hide_index=True)

                    st.warning(
                        "⚠️ **IRREVERSIBLE ACTION:** All records shown above will be permanently deleted using a single database transaction. "
                        "Shared global topics (`scope='all'`) and other tracked channels will not be touched."
                    )

                    if del_preview.get("is_locked"):
                        st.error(f"🚫 **Deletion is temporarily blocked:** {del_preview.get('lock_reason')}")

                    del_confirm_input = st.text_input(
                        "To confirm permanent deletion, type DELETE in uppercase:",
                        placeholder="DELETE",
                        key="del_confirm_text_input"
                    )

                    btn_del_perm = st.button(
                        f"🗑️ Permanently Delete '{del_cname}' and All Associated Data",
                        type="primary",
                        disabled=(del_confirm_input.strip() != "DELETE" or del_preview.get("is_locked", False)),
                        key="btn_exec_permanent_delete",
                        width="stretch"
                    )

                    if btn_del_perm:
                        if not hasattr(channel_manager, "delete_channel_permanently"):
                            try:
                                importlib.reload(channel_manager)
                            except Exception:
                                pass

                        del_result = channel_manager.delete_channel_permanently(
                            channel_id=del_cid,
                            confirmation=del_confirm_input,
                            admin_user=True,
                            gh_token=gh_token
                        )
                        if del_result.get("success"):
                            st.cache_data.clear()
                            st.success(del_result.get("message"))
                            storage_info = del_result.get("storage_cleanup", {})
                            if not storage_info.get("success") and storage_info.get("failed_paths"):
                                st.warning(f"Database records were purged, but storage cleanup reported: {storage_info.get('error')}. You can retry below.")
                                st.session_state["pending_storage_purge"] = storage_info.get("failed_paths")
                            st.rerun()
                        else:
                            st.error(del_result.get("error") or del_result.get("message"))

                    if "pending_storage_purge" in st.session_state and st.session_state["pending_storage_purge"]:
                        st.markdown("---")
                        st.markdown("##### 🗄️ Pending Audio Storage Purge")
                        st.caption(f"{len(st.session_state['pending_storage_purge'])} audio object(s) awaiting removal from Supabase Storage.")
                        if st.button("🔄 Retry Storage Audio Purge", key="btn_retry_storage_purge"):
                            from core import storage_audio
                            retry_res = storage_audio.retry_storage_cleanup("mediaintel-audio", st.session_state["pending_storage_purge"])
                            if retry_res.get("success"):
                                st.success(f"Purged {retry_res.get('deleted_count')} audio storage object(s).")
                                del st.session_state["pending_storage_purge"]
                                st.rerun()
                            else:
                                st.error(f"Retry failed: {retry_res.get('error')}")

            st.markdown("<hr>", unsafe_allow_html=True)

            # -----------------------------------------------------------
            # Section 2: Prominent Run Full Analysis Engine
            # -----------------------------------------------------------
            st.markdown("##### 🚀 On-Demand Cloud Analysis Engine")
            st.caption("Execute serverless intelligence in GitHub Actions without running Python or Whisper on your laptop.")

            branch_ref = config.env("GITHUB_REF_NAME")
            if not branch_ref:
                try:
                    branch_ref = st.secrets.get("GITHUB_REF_NAME")
                except Exception:
                    pass
            branch_ref = branch_ref or "main"

            p_c1, p_c2, p_c3 = st.columns([2, 1, 1])
            with p_c1:
                tracked_df = all_channels_df[all_channels_df["is_tracked"] == True] if not all_channels_df.empty else pd.DataFrame()
                target_mode = st.radio(
                    "Competitor Scope",
                    [f"All Enabled Channels ({len(tracked_df)} channels)", "Selected Channel(s) Only"],
                    horizontal=True,
                    key="scope_radio"
                )
                target_cids_list = []
                if "Selected" in target_mode and not tracked_df.empty:
                    sel_chans = st.multiselect(
                        "Select Competitors for this Run",
                        options=tracked_df["channel_name"].tolist(),
                        default=tracked_df["channel_name"].tolist()[:2],
                        key="sel_run_chans"
                    )
                    target_cids_list = tracked_df[tracked_df["channel_name"].isin(sel_chans)]["channel_id"].tolist()
            with p_c2:
                v_limit = st.number_input("Max Videos / Channel", min_value=1, max_value=50, value=10, step=1, key="v_limit_in")
            with p_c3:
                s_limit = st.number_input(
                    "Max RoBERTa Comments (0 for All)",
                    min_value=0,
                    max_value=8000,
                    value=0,
                    step=500,
                    key="s_limit_in",
                    help="0 analyzes all pending comments incrementally across all channels without truncation."
                )

            btn_run_analysis = st.button("🚀 Run Full Analysis", type="primary", width="stretch", key="btn_run_full_analysis")

            if btn_run_analysis:
                if not gh_token:
                    st.error("⚠️ GitHub personal access token (`GITHUB_TOKEN`) is not configured. Please add `GITHUB_TOKEN = 'ghp_...'` in Streamlit Secrets or environment variables to dispatch workflows.")
                else:
                    with st.spinner("Dispatching cloud intelligence pipeline to GitHub Actions..."):
                        dispatch_res = channel_manager.dispatch_cloud_pipeline(
                            gh_token=gh_token,
                            gh_repo=gh_repo,
                            ref=branch_ref,
                            channel_ids=target_cids_list,
                            video_limit=v_limit,
                            comments_per_video=100,
                            transcripts_per_channel=2,
                            sentiment_limit=s_limit,
                            run_whisper=True,
                            force=False,
                        )
                        if dispatch_res["success"]:
                            st.session_state["pipeline_dispatched"] = True
                            st.session_state["pipeline_dispatched_at"] = datetime.now().strftime("%H:%M:%S")
                            st.success(f"✅ {dispatch_res['message']} (Workflow Queued in GitHub Actions)")
                        elif dispatch_res.get("status_code") == 409:
                            active_r = dispatch_res.get("active_run", {})
                            run_url = active_r.get("html_url", f"https://github.com/{gh_repo}/actions")
                            st.warning(f"⚠️ {dispatch_res['message']} [Click here to view the active run in GitHub Actions]({run_url}).")
                        else:
                            st.error(f"❌ {dispatch_res['message']}")

            # -----------------------------------------------------------
            # Section 3: Live Processing Progress & Job Telemetry
            # -----------------------------------------------------------
            st.markdown("<hr>", unsafe_allow_html=True)
            stat_h1, stat_h2 = st.columns([3, 1])
            with stat_h1:
                st.markdown("##### 📡 Cloud Pipeline Processing Status")
            with stat_h2:
                if st.button("🔄 Poll Status", width="stretch", key="btn_poll_pipeline"):
                    st.rerun()

            if gh_token:
                recent_runs = channel_manager.get_latest_workflow_runs(gh_token, gh_repo, limit=2)
                if recent_runs:
                    latest_run = recent_runs[0]
                    run_status = latest_run.get("status", "unknown")
                    run_conclusion = latest_run.get("conclusion")
                    run_id = latest_run.get("id")
                    run_url = latest_run.get("html_url", "#")

                    if run_status in ("queued", "waiting"):
                        badge = f'<span class="status-pill" style="color:#fbbf24; border-color:rgba(251,191,36,0.3); background:rgba(251,191,36,0.12);"><span class="dot" style="background:#fbbf24;"></span> Run #{run_id}: Queued</span>'
                    elif run_status == "in_progress":
                        badge = f'<span class="status-pill" style="color:#60a5fa; border-color:rgba(96,165,250,0.3); background:rgba(96,165,250,0.12);"><span class="dot" style="background:#60a5fa;"></span> Run #{run_id}: Running in Cloud</span>'
                    elif run_conclusion == "success":
                        badge = f'<span class="status-pill"><span class="dot"></span> Run #{run_id}: Completed Successfully</span>'
                    elif run_conclusion in ("failure", "timed_out"):
                        badge = f'<span class="status-pill" style="color:#f87171; border-color:rgba(248,113,113,0.3); background:rgba(248,113,113,0.12);"><span class="dot" style="background:#f87171;"></span> Run #{run_id}: Failed (Check Logs)</span>'
                    else:
                        badge = f'<span class="status-pill"><span class="dot"></span> Run #{run_id}: {run_status} ({run_conclusion or "running"})</span>'

                    st.markdown(f"{badge} &nbsp;&nbsp; 🔗 [Open Run #{run_id} in GitHub Actions ↗]({run_url})", unsafe_allow_html=True)

                    jobs = channel_manager.get_workflow_run_jobs(run_id, gh_token, gh_repo)
                    if jobs:
                        j_cols = st.columns(len(jobs))
                        for idx, job in enumerate(jobs):
                            with j_cols[idx]:
                                j_status = job.get("status", "")
                                j_conc = job.get("conclusion") or j_status
                                j_icon = "✅" if j_conc == "success" else ("🔄" if j_status == "in_progress" else ("⏳" if j_status == "queued" else ("❌" if j_conc == "failure" else "⚪")))
                                st.markdown(f"**{j_icon} {job.get('name')}**\n\n`{j_conc.upper()}`")
                else:
                    st.info("No cloud pipeline runs found yet in GitHub Actions.")
            else:
                st.caption("ℹ️ Configure `GITHUB_TOKEN` in Streamlit Secrets to view live GitHub Actions run status and job stages.")

            # -----------------------------------------------------------
            # Section 4: Channel Ingestion Status & Coverage Table
            # -----------------------------------------------------------
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("##### 🎯 Per-Channel Ingestion & Transcription Coverage Ledger (Active Analysis Set)")
            st.caption("Detailed breakdown of active run targets vs collected videos, comments, and transcripts for the current analysis set (10 videos, 2 transcription candidates).")
            try:
                coverage_df = channel_manager.get_channel_run_coverage_df()
                st.dataframe(coverage_df, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Could not load channel coverage ledger: {type(exc).__name__}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📊 All-Time Channel Telemetry & Storage State (All Warehouse Records)")
            st.caption("Cumulative historical inventory across all ingestion runs in the Supabase PostgreSQL warehouse, including all videos, comments, and completed transcripts.")
            try:
                telemetry_df = channel_manager.get_channel_telemetry_df()
                st.dataframe(telemetry_df, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Could not load channel telemetry: {type(exc).__name__}")

            st.markdown("<br>", unsafe_allow_html=True)
            ref_c1, ref_c2 = st.columns([1, 1])
            with ref_c1:
                if st.button("🔄 Refresh Results & Reload Data", width="stretch", key="btn_refresh_results"):
                    try:
                        channel_manager.update_all_channel_run_coverage()
                        st.cache_data.clear()
                        st.success("Dashboard cache cleared. Fresh data loaded.")
                    except Exception as exc:
                        st.error(f"Failed to refresh coverage ledger: {type(exc).__name__} ({exc})")
                        st.cache_data.clear()
                    st.rerun()
            with ref_c2:
                st.caption("ℹ️ Once GitHub Actions indicates 'Completed Successfully', click 'Refresh Results' to display newly analyzed videos and sentiment.")

st.markdown(
    f'<p style="text-align:center; color:#475569; font-size:12px; margin-top:24px;">'
    f"MediaIntel Dashboard · Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True,
)
