"""Stage 1 - YouTube ingestion.

Pulls the latest uploads of every tracked channel (metadata, statistics, comments), keeps a
statistics snapshot per run (for spike detection) and stores everything in the relational database (SQLite by default, MySQL optional).

Efficiency for many channels:
  * one batched videos.list call per channel (up to 50 videos) instead of one call per video
  * comments of all videos are fetched in parallel worker threads
  * all rows of a video are written in a single bulk statement
"""
from __future__ import annotations

import html
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402
from sqlalchemy import text  # noqa: E402

VIDEOS_PER_CHANNEL = min(max(config.setting_int("VIDEOS_PER_CHANNEL"), 1), 50)
COMMENT_PAGES = max(config.env_int("COMMENT_PAGES", 1), 1)       # 1 page = up to 100 comments per video
WORKERS = max(config.env_int("INGEST_WORKERS", 4), 1)


# ---------------------------------------------------------------------------
# YouTube API wrapper (kept thin so tests can swap in a fake)
# ---------------------------------------------------------------------------
class QuotaExceeded(RuntimeError):
    pass


class YouTubeClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._local = threading.local()

    def _svc(self):
        # googleapiclient service objects are not thread-safe -> one per thread
        if not hasattr(self._local, "svc"):
            from googleapiclient.discovery import build
            self._local.svc = build("youtube", "v3", developerKey=self.api_key, cache_discovery=False)
        return self._local.svc

    @staticmethod
    def _run(request):
        from googleapiclient.errors import HttpError
        try:
            return request.execute()
        except HttpError as exc:
            if getattr(exc, "status_code", None) == 403 and "quota" in str(exc).lower():
                raise QuotaExceeded("YouTube API daily quota exceeded") from exc
            raise

    def uploads_playlist_id(self, channel_id: str):
        resp = self._run(self._svc().channels().list(part="contentDetails", id=channel_id))
        items = resp.get("items", [])
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"] if items else None

    def recent_video_ids(self, playlist_id: str, limit: int) -> list[str]:
        resp = self._run(self._svc().playlistItems().list(
            part="contentDetails", playlistId=playlist_id, maxResults=limit))
        return [i["contentDetails"]["videoId"] for i in resp.get("items", [])]

    def video_details(self, video_ids: list[str]) -> list[dict]:
        resp = self._run(self._svc().videos().list(
            part="snippet,statistics,contentDetails", id=",".join(video_ids), maxResults=50))
        out = []
        for it in resp.get("items", []):
            st, sn = it.get("statistics", {}), it.get("snippet", {})
            out.append({
                "video_id": it["id"],
                "title": html.unescape(sn.get("title", ""))[:250],
                "description": html.unescape(sn.get("description", ""))[:4000],
                "published_at": _parse_ts(sn.get("publishedAt")),
                "duration_sec": _parse_duration(it.get("contentDetails", {}).get("duration")),
                "view_count": int(st.get("viewCount", 0)),
                "like_count": int(st.get("likeCount", 0)),
                "comment_count": int(st.get("commentCount", 0)),
            })
        return out

    def top_comments(self, video_id: str, pages: int = 1) -> list[dict]:
        rows, token = [], None
        for _ in range(pages):
            resp = self._run(self._svc().commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                textFormat="plainText", pageToken=token))
            for item in resp.get("items", []):
                c = item["snippet"]["topLevelComment"]["snippet"]
                rows.append({
                    "comment_id": item["id"], "video_id": video_id,
                    "author": c.get("authorDisplayName", ""), "text": c.get("textDisplay", ""),
                    "likes": int(c.get("likeCount", 0)), "published_at": _parse_ts(c.get("publishedAt")),
                })
            token = resp.get("nextPageToken")
            if not token:
                break
        return rows


def make_client() -> YouTubeClient:
    if config.env_bool("MEDIAINTEL_TEST_FAKES"):   # offline self-test hook
        sys.path.insert(0, str(config.ROOT / "tests"))
        import fakes
        return fakes.FakeYouTube()
    key = config.env("YOUTUBE_API_KEY")
    if not key:
        print("ERROR: YOUTUBE_API_KEY is missing from your .env file.")
        sys.exit(1)
    return YouTubeClient(key)


def _parse_ts(value):
    if not value:
        return None
    return datetime.strptime(value.replace("Z", "")[:19], "%Y-%m-%dT%H:%M:%S")


def _parse_duration(iso):
    """ISO-8601 'PT1H2M3S' -> seconds."""
    if not iso:
        return None
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return None
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


# ---------------------------------------------------------------------------
# Database writes
# ---------------------------------------------------------------------------
VIDEO_COLS = ["video_id", "channel_id", "title", "description", "duration_sec", "view_count", "like_count",
              "comment_count", "published_at"]
COMMENT_COLS = ["comment_id", "video_id", "author_name", "comment_text", "like_count", "published_at", "is_processed"]


def _comment_rows(comments):
    return [{"comment_id": c["comment_id"], "video_id": c["video_id"], "author_name": c["author"][:250],
             "comment_text": c["text"], "like_count": c["likes"], "published_at": c["published_at"], "is_processed": False}
            for c in comments]


def ingest_competitor_data() -> None:
    print("Initializing YouTube data ingestion...")
    from core import db_setup
    db_setup.ensure_schema(verbose=False)      # lets this script also run on its own, on a brand-new database
    db_setup.sync_channels()
    client = make_client()
    engine = config.get_engine()

    with engine.connect() as conn:
        channels = conn.execute(text(
            "SELECT channel_id, channel_name FROM channels WHERE is_tracked = TRUE")).fetchall()
    
    if not channels:
        print("WARNING: no tracked channels in the database. Add rows to the `channels` table first.")
        return

    # Optional single or multi-channel mode for controlled cloud ingestion.
    selected_channel_input = config.env("MEDIAINTEL_INGEST_CHANNEL_ID")
    if selected_channel_input:
        target_ids = {c.strip() for c in selected_channel_input.split(",") if c.strip()}
        if target_ids:
            channels = [
                row for row in channels
                if row[0] in target_ids
            ]
            if not channels:
                raise SystemExit(
                    "ERROR: None of the selected channels are tracked or exist in database."
                )

    # Optional per-channel video limit for cloud testing.
    video_limit = VIDEOS_PER_CHANNEL
    if config.env("MEDIAINTEL_INGEST_VIDEO_LIMIT"):
        video_limit = min(
            max(config.env_int("MEDIAINTEL_INGEST_VIDEO_LIMIT", 2), 1),
            50
        )


    now = config.utcnow()
    upsert_video = config.upsert_sql("videos", VIDEO_COLS, ["video_id"],
                                     ["title", "description", "duration_sec", "view_count", "like_count", "comment_count"])
    upsert_comment = config.upsert_sql("comments", COMMENT_COLS, ["comment_id"], ["like_count"])
    insert_history = text("""INSERT INTO video_stats_history (video_id, view_count, like_count, comment_count, captured_at)
                             VALUES (:video_id, :view_count, :like_count, :comment_count, :captured_at)""")
    total_videos = total_comments = failed_channels = 0
    try:
        for channel_id, channel_name in channels:
            print(f"\nChannel: {channel_name} ({channel_id})")
            try:
                playlist = client.uploads_playlist_id(channel_id)
                if not playlist:
                    print(f"  ERROR: could not find uploads playlist for {channel_name} - check its channel_id")
                    failed_channels += 1
                    continue
                video_ids = client.recent_video_ids(playlist, video_limit)
                details = client.video_details(video_ids) if video_ids else []
            except QuotaExceeded:
                raise
            except Exception as exc:
                print(f"  ERROR: API failure for {channel_name}: {str(exc)[:200]}")
                failed_channels += 1
                continue

            rows = [dict(d, channel_id=channel_id) for d in details]
            with engine.begin() as conn:
                if rows:
                    conn.execute(upsert_video, rows)
                    conn.execute(insert_history, [
                        {"video_id": r["video_id"], "view_count": r["view_count"],
                         "like_count": r["like_count"], "comment_count": r["comment_count"],
                         "captured_at": now} for r in rows])
            total_videos += len(rows)
            print(f"  {len(rows)} videos stored")

            # comments of every video of this channel, in parallel
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(client.top_comments, r["video_id"], COMMENT_PAGES): r["video_id"]
                           for r in rows if r["comment_count"] > 0}
                for fut in as_completed(futures):
                    vid = futures[fut]
                    try:
                        comments = fut.result()
                    except QuotaExceeded:
                        raise
                    except Exception:
                        print(f"  comments unavailable for {vid} (disabled or restricted)")
                        continue
                    if comments:
                        with engine.begin() as conn:
                            conn.execute(upsert_comment, _comment_rows(comments))
                        total_comments += len(comments)
    except QuotaExceeded:
        print("\nERROR: YouTube API quota exhausted. Data collected so far is saved; try again tomorrow.")
        sys.exit(2)

    print(f"\nData ingestion complete: {total_videos} videos, {total_comments} comments synced"
          + (f" ({failed_channels} channel(s) failed)." if failed_channels else "."))
    if failed_channels == len(channels):
        print("ERROR: no channel could be read - check YOUTUBE_API_KEY (valid? YouTube Data API v3 enabled?) and the channel IDs.")
        sys.exit(3)


if __name__ == "__main__":
    ingest_competitor_data()
