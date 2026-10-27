"""Online Channel Management, Batch Onboarding, and GitHub Actions Cloud Pipeline Orchestrator.

Provides:
- Resolution of YouTube channel IDs, URLs, handles, and JSON batch lists.
- Safe idempotent database upserts for channels.
- Tracking status toggles (non-destructive).
- Secure GitHub Actions workflow dispatch with deduplication.
- Live progress polling for cloud pipeline runs and job stages.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from core import config

__all__ = [
    "parse_channel_input",
    "resolve_channel_identifier",
    "register_single_channel",
    "register_batch_channels",
    "set_channel_tracking",
    "get_channel_telemetry_df",
    "update_all_channel_run_coverage",
    "get_channel_run_coverage_df",
    "get_latest_workflow_runs",
    "get_workflow_run_jobs",
    "dispatch_cloud_pipeline",
    "dispatch_rpa_reporting",
    "check_channel_concurrency",
    "preview_channel_deletion",
    "delete_channel_permanently",
]


def parse_channel_input(raw_text: str) -> list[dict[str, str]]:
    """Parse raw input text into a list of candidate items.
    
    Supports:
    - JSON array of objects (e.g. [{"channel_id": "...", "name": "..."}])
    - JSON array of strings (e.g. ["UC...", "@handle"])
    - Multiline or comma-separated URLs, handles, or IDs.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return []

    # 1. Try parsing as JSON first
    if (raw_text.startswith("[") and raw_text.endswith("]")) or (raw_text.startswith("{") and raw_text.endswith("}")):
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                results = []
                for item in parsed:
                    if isinstance(item, dict):
                        cid = item.get("channel_id") or item.get("id") or item.get("url") or ""
                        name = item.get("name") or item.get("channel_name") or item.get("title") or ""
                        if cid:
                            results.append({"raw": str(cid).strip(), "name": str(name).strip()})
                    elif isinstance(item, str) and item.strip():
                        results.append({"raw": item.strip(), "name": ""})
                if results:
                    return results
            elif isinstance(parsed, dict):
                cid = parsed.get("channel_id") or parsed.get("id") or parsed.get("url") or ""
                name = parsed.get("name") or parsed.get("channel_name") or parsed.get("title") or ""
                if cid:
                    return [{"raw": str(cid).strip(), "name": str(name).strip()}]
        except Exception:
            pass  # Fall through to line-by-line parsing

    # 2. Line-by-line or comma-separated parsing
    lines = [line.strip() for line in re.split(r"[\r\n,]+", raw_text) if line.strip()]
    items = []
    for line in lines:
        # Check if line contains a label: "UC... - Channel Name" or "UC... | Channel Name" or tab
        parts = re.split(r"\s+[-|–—]\s+|\t+", line, maxsplit=1)
        if len(parts) == 2:
            items.append({"raw": parts[0].strip(), "name": parts[1].strip()})
        else:
            items.append({"raw": line, "name": ""})
    return items


def resolve_channel_identifier(
    raw_identifier: str,
    api_key: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve a raw YouTube channel identifier (ID, URL, or handle) to (channel_id, channel_name, error).
    
    Returns:
        (channel_id, resolved_name, error_message)
    """
    raw = (raw_identifier or "").strip()
    if not raw:
        return None, None, "Identifier cannot be empty"

    # 1. Direct regex match for standard 24-character YouTube Channel ID (starts with UC)
    match_uc = re.search(r"(UC[A-Za-z0-9_-]{22})", raw)
    if match_uc:
        channel_id = match_uc.group(1)
        # If API key available and no display name provided, fetch channel title
        title = None
        if api_key:
            try:
                import requests
                resp = requests.get(
                    "https://www.googleapis.com/youtube/v3/channels",
                    params={"part": "snippet", "id": channel_id, "key": api_key},
                    timeout=6,
                )
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    if items:
                        title = items[0]["snippet"].get("title")
            except Exception:
                pass
        return channel_id, title, None

    # 2. Check for YouTube Handle: @handle or youtube.com/@handle
    handle_match = re.search(r"@([A-Za-z0-9_.-]+)", raw)
    if handle_match:
        handle = handle_match.group(1)
        if not api_key:
            return None, None, f"YouTube API key required to resolve handle '@{handle}' to Channel ID. Please provide the 24-character 'UC...' Channel ID."

        try:
            import requests
            resp = requests.get(
                "https://www.googleapis.com/youtube/v3/channels",
                params={"part": "id,snippet", "forHandle": handle, "key": api_key},
                timeout=8,
            )
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                if items:
                    cid = items[0]["id"]
                    title = items[0]["snippet"].get("title", f"@{handle}")
                    return cid, title, None
                return None, None, f"Handle '@{handle}' was not found on YouTube."
            elif resp.status_code == 403:
                return None, None, "YouTube API quota exceeded or key invalid while resolving handle."
            else:
                return None, None, f"YouTube API returned HTTP {resp.status_code} while resolving handle '@{handle}'."
        except Exception as exc:
            return None, None, f"Network error resolving handle '@{handle}': {type(exc).__name__}"

    return None, None, f"Could not extract a valid 24-character YouTube Channel ID (starting with 'UC') from: '{raw}'"


def register_single_channel(
    channel_id: str,
    channel_name: str | None = None,
    is_tracked: bool = True,
) -> dict[str, Any]:
    """Register or update a channel in the database idempotently without deleting historical data."""
    if not channel_id or not re.match(r"^UC[A-Za-z0-9_-]{22}$", channel_id):
        return {
            "success": False,
            "channel_id": channel_id,
            "channel_name": channel_name or channel_id,
            "status": "invalid_id",
            "message": "Invalid YouTube Channel ID. Must be 24 characters starting with 'UC'.",
        }

    engine = config.get_engine()
    with engine.connect() as conn:
        existing = conn.execute(
            text("SELECT channel_id, channel_name, is_tracked FROM channels WHERE channel_id = :cid"),
            {"cid": channel_id},
        ).fetchone()

    name = (channel_name or (existing[1] if existing else None) or channel_id)[:100].strip()

    if existing:
        # Channel exists: update tracking status and name if provided
        config.execute(
            "UPDATE channels SET channel_name = :name, is_tracked = :tracked WHERE channel_id = :cid",
            {"name": name, "tracked": is_tracked, "cid": channel_id},
        )
        return {
            "success": True,
            "channel_id": channel_id,
            "channel_name": name,
            "status": "updated",
            "message": f"Channel '{name}' is already registered. Tracking updated to {'Enabled' if is_tracked else 'Disabled'}.",
        }
    else:
        # Channel is new: insert
        config.execute(
            config.upsert_sql(
                "channels",
                ["channel_id", "channel_name", "is_tracked"],
                ["channel_id"],
                ["channel_name", "is_tracked"],
            ),
            {"channel_id": channel_id, "channel_name": name, "is_tracked": is_tracked},
        )
        return {
            "success": True,
            "channel_id": channel_id,
            "channel_name": name,
            "status": "created",
            "message": f"Channel '{name}' ({channel_id}) successfully registered and tracking enabled.",
        }


def register_batch_channels(
    raw_batch_text: str,
    api_key: str | None = None,
    default_tracked: bool = True,
) -> list[dict[str, Any]]:
    """Parse, resolve, and register multiple YouTube channels in a single batch.
    
    Prevents duplicates within the batch and reports individual item status.
    """
    candidates = parse_channel_input(raw_batch_text)
    if not candidates:
        return []

    results = []
    seen_in_batch: set[str] = set()

    for item in candidates:
        raw_id = item["raw"]
        custom_name = item.get("name") or None

        # Resolve ID and title
        cid, resolved_name, err = resolve_channel_identifier(raw_id, api_key=api_key)
        if err or not cid:
            results.append({
                "raw": raw_id,
                "success": False,
                "channel_id": None,
                "channel_name": custom_name or raw_id,
                "status": "resolution_error",
                "message": err or "Failed to resolve channel identifier.",
            })
            continue

        if cid in seen_in_batch:
            results.append({
                "raw": raw_id,
                "success": True,
                "channel_id": cid,
                "channel_name": custom_name or resolved_name or cid,
                "status": "duplicate_in_batch",
                "message": f"Channel ID '{cid}' appeared multiple times in this batch. Skipped duplicate.",
            })
            continue

        seen_in_batch.add(cid)
        final_name = custom_name or resolved_name or cid
        reg_result = register_single_channel(cid, channel_name=final_name, is_tracked=default_tracked)
        reg_result["raw"] = raw_id
        results.append(reg_result)

    return results


def set_channel_tracking(channel_id: str, is_tracked: bool) -> bool:
    """Toggle tracking status for a channel without deleting historical data."""
    try:
        config.execute(
            "UPDATE channels SET is_tracked = :tracked WHERE channel_id = :cid",
            {"tracked": is_tracked, "cid": channel_id},
        )
        return True
    except Exception:
        return False


def get_channel_telemetry_df():
    """Retrieve full status, video, comment, sentiment, and transcript telemetry for all registered channels."""
    sql = """
        SELECT
            c.channel_name AS "Channel",
            c.channel_id AS "Channel ID",
            CASE WHEN c.is_tracked THEN 'Active' ELSE 'Inactive' END AS "Tracking",
            COUNT(DISTINCT v.video_id) AS "Videos",
            COUNT(DISTINCT m.comment_id) AS "Comments",
            COUNT(DISTINCT s.comment_id) AS "Sentiments",
            COUNT(DISTINCT CASE WHEN t.source IN ('whisper', 'faster_whisper') AND t.transcript_text IS NOT NULL AND TRIM(t.transcript_text) != '' THEN t.video_id END) AS "Whisper Transcripts",
            COUNT(DISTINCT CASE WHEN t.source IN ('captions', 'youtube_manual_caption', 'youtube_auto_caption') AND t.transcript_text IS NOT NULL AND TRIM(t.transcript_text) != '' THEN t.video_id END) AS "Captions",
            COUNT(DISTINCT CASE WHEN t.source IN ('whisper', 'faster_whisper', 'captions', 'youtube_manual_caption', 'youtube_auto_caption') AND t.transcript_text IS NOT NULL AND TRIM(t.transcript_text) != '' THEN t.video_id END) AS "Total Completed Transcripts",
            COUNT(DISTINCT CASE WHEN t.source IN ('audio_unavailable', 'awaiting_authorized_audio', 'failed') THEN t.video_id END) AS "Audio Unavailable",
            MAX(v.published_at) AS "Latest Video"
        FROM channels c
        LEFT JOIN videos v ON v.channel_id = c.channel_id
        LEFT JOIN comments m ON m.video_id = v.video_id
        LEFT JOIN sentiment_insights s ON s.comment_id = m.comment_id
        LEFT JOIN transcripts t ON t.video_id = v.video_id
        GROUP BY c.channel_id, c.channel_name, c.is_tracked
        ORDER BY c.is_tracked DESC, c.channel_name ASC
    """
    return config.read_sql(sql)


def get_active_analysis_videos_df(
    limit_per_channel: int = 10,
    channel_ids: list[str] | None = None,
):
    """Retrieve exactly up to limit_per_channel most recent videos per tracked channel.
    
    Deterministic order: published_at DESC, video_id ASC.
    Preserves all historical records in warehouse while scoping current analysis set.
    """
    query = """
        SELECT v.*, c.channel_name
        FROM videos v
        JOIN channels c ON c.channel_id = v.channel_id
        WHERE c.is_tracked = TRUE
    """
    df = config.read_sql(query)
    if df.empty:
        return df

    if channel_ids:
        cid_set = set(channel_ids)
        df = df[df["channel_id"].isin(cid_set)]
        if df.empty:
            return df

    # Deterministic sort: published_at DESC, video_id ASC
    df = df.sort_values(
        by=["channel_id", "published_at", "video_id"],
        ascending=[True, False, True]
    )
    return df.groupby("channel_id").head(limit_per_channel).reset_index(drop=True)


def get_transcription_candidate_videos(
    limit_per_channel: int = 2,
    channel_ids: list[str] | None = None,
):
    """Select the latest published videos per tracked channel for transcription.
    
    Deterministic ranking: published_at DESC, video_id ASC.
    """
    active_v = get_active_analysis_videos_df(10, channel_ids=channel_ids)
    if active_v.empty:
        return active_v

    # Deterministic ranking within each channel: published_at DESC, video_id ASC
    sorted_df = active_v.sort_values(
        by=["channel_id", "published_at", "video_id"],
        ascending=[True, False, True]
    )
    return sorted_df.groupby("channel_id").head(limit_per_channel).reset_index(drop=True)


def update_all_channel_run_coverage(
    video_limit: int = 10,
    target_cids: list[str] | None = None,
) -> None:
    """Compute and persist per-channel analysis coverage telemetry to channel_run_coverage table."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    channels_df = config.read_sql("SELECT channel_id, channel_name FROM channels WHERE is_tracked = TRUE ORDER BY channel_name ASC")
    if channels_df.empty:
        return

    if target_cids:
        c_set = set(target_cids)
        channels_df = channels_df[channels_df["channel_id"].isin(c_set)]

    active_videos_df = get_active_analysis_videos_df(limit_per_channel=video_limit)
    candidate_videos_df = get_transcription_candidate_videos(limit_per_channel=2)

    active_vids = set(active_videos_df["video_id"].tolist()) if not active_videos_df.empty else set()
    candidate_vids = set(candidate_videos_df["video_id"].tolist()) if not candidate_videos_df.empty else set()

    # Load comments, sentiment insights, and transcripts for active set
    comments_df = config.read_sql("SELECT comment_id, video_id FROM comments")
    insights_df = config.read_sql("SELECT comment_id FROM sentiment_insights")
    transcripts_df = config.read_sql("SELECT video_id, source, transcript_text FROM transcripts")

    active_comments = comments_df[comments_df["video_id"].isin(active_vids)] if not comments_df.empty else comments_df
    active_comment_ids = set(active_comments["comment_id"].tolist()) if not active_comments.empty else set()
    active_insights = insights_df[insights_df["comment_id"].isin(active_comment_ids)] if not insights_df.empty else insights_df
    candidate_transcripts = transcripts_df[transcripts_df["video_id"].isin(candidate_vids)] if not transcripts_df.empty else transcripts_df

    coverage_rows = []
    for row in channels_df.itertuples():
        cid = row.channel_id
        cname = row.channel_name

        c_videos = active_videos_df[active_videos_df["channel_id"] == cid] if not active_videos_df.empty else active_videos_df
        selected_v = len(c_videos)
        target_v = video_limit

        c_vids_set = set(c_videos["video_id"].tolist()) if not c_videos.empty else set()
        c_comments = active_comments[active_comments["video_id"].isin(c_vids_set)] if not active_comments.empty else active_comments
        collected_c = len(c_comments)
        target_c = selected_v * 100

        c_comment_ids = set(c_comments["comment_id"].tolist()) if not c_comments.empty else set()
        c_insights = active_insights[active_insights["comment_id"].isin(c_comment_ids)] if not active_insights.empty else active_insights
        analyzed_c = len(c_insights)

        c_cand = candidate_videos_df[candidate_videos_df["channel_id"] == cid] if not candidate_videos_df.empty else candidate_videos_df
        trans_candidates = min(selected_v, 2)
        c_cand_set = set(c_cand["video_id"].tolist()) if not c_cand.empty else set()

        completed_tx = 0
        if not candidate_transcripts.empty and c_cand_set:
            completed_tx = len(candidate_transcripts[
                candidate_transcripts["video_id"].isin(c_cand_set) &
                candidate_transcripts["source"].isin(["whisper", "faster_whisper", "captions", "youtube_manual_caption", "youtube_auto_caption"]) &
                candidate_transcripts["transcript_text"].notna() &
                (candidate_transcripts["transcript_text"].str.strip() != "")
            ])

        # Pipeline Stage 1: Video Ingestion
        if selected_v == 0:
            v_status = "No Videos Found"
        elif selected_v >= target_v:
            v_status = f"Completed ({selected_v}/{target_v})"
        else:
            v_status = f"Partial ({selected_v}/{target_v})"

        # Pipeline Stage 2: Comment Collection
        if selected_v == 0:
            c_status = "N/A"
        elif collected_c > 0:
            c_status = f"Collected ({collected_c})"
        else:
            c_status = "Comments Disabled/Zero"

        # Pipeline Stage 3: Sentiment Analysis
        if collected_c == 0:
            s_status = "N/A"
        elif analyzed_c >= collected_c:
            s_status = f"Analyzed ({analyzed_c}/{collected_c})"
        else:
            s_status = f"Analyzing ({analyzed_c}/{collected_c})"

        # Pipeline Stage 4: Whisper Transcription
        if trans_candidates == 0:
            tx_status = "N/A"
        elif completed_tx >= trans_candidates:
            tx_status = f"Completed ({completed_tx}/{trans_candidates})"
        else:
            tx_status = f"Audio Pending ({completed_tx}/{trans_candidates})"

        # Detailed ledger notes / missing data reason (non-alarming, accurate)
        reasons = []
        if selected_v == 0:
            reasons.append("No public videos found on channel")
        elif selected_v < target_v:
            reasons.append(f"Channel published only {selected_v} eligible videos")
        if selected_v > 0 and collected_c == 0:
            reasons.append("Comments disabled or 0 comments published")
        if trans_candidates > completed_tx:
            reasons.append(f"Whisper audio pending upload ({completed_tx}/{trans_candidates})")

        missing_reason = "; ".join(reasons) if reasons else "None"

        # Overall Status: Ingestion status is NOT penalized by pending whisper audio!
        if selected_v == 0:
            status = "No Data"
        elif selected_v >= target_v and collected_c > 0:
            if completed_tx >= trans_candidates:
                status = "Complete"
            else:
                status = "Ingestion Complete"
        else:
            status = "Partial Ingestion"

        coverage_rows.append({
            "channel_id": cid,
            "channel_name": cname,
            "target_videos": target_v,
            "selected_videos": selected_v,
            "target_comments": target_c,
            "collected_comments": collected_c,
            "transcription_candidates": trans_candidates,
            "completed_transcripts": completed_tx,
            "video_status": v_status,
            "comment_status": c_status,
            "sentiment_status": s_status,
            "transcription_status": tx_status,
            "missing_data_reason": missing_reason,
            "last_run_status": status,
            "updated_at": now,
        })

    if coverage_rows:
        cols = [
            "channel_id", "channel_name", "target_videos", "selected_videos",
            "target_comments", "collected_comments", "transcription_candidates",
            "completed_transcripts", "video_status", "comment_status",
            "sentiment_status", "transcription_status",
            "missing_data_reason", "last_run_status", "updated_at"
        ]
        upsert_stmt = config.upsert_sql(
            "channel_run_coverage",
            cols,
            ["channel_id"],
            ["channel_name", "target_videos", "selected_videos", "target_comments",
             "collected_comments", "transcription_candidates", "completed_transcripts",
             "video_status", "comment_status", "sentiment_status", "transcription_status",
             "missing_data_reason", "last_run_status", "updated_at"]
        )
        with config.get_engine().begin() as conn:
            conn.execute(upsert_stmt, coverage_rows)


def get_channel_run_coverage_df():
    """Retrieve full per-channel analysis coverage table for Streamlit and auditing."""
    try:
        sql = """
            SELECT
                channel_name AS "Channel Name",
                target_videos AS "Target Videos",
                selected_videos AS "Selected Videos",
                video_status AS "Video Status",
                target_comments AS "Target Comments",
                collected_comments AS "Collected Comments",
                comment_status AS "Comment Status",
                sentiment_status AS "Sentiment Status",
                transcription_candidates AS "Transcription Candidates",
                completed_transcripts AS "Completed Transcripts",
                transcription_status AS "Transcription Status",
                last_run_status AS "Last Run Status",
                missing_data_reason AS "Missing Data Reason"
            FROM channel_run_coverage
            ORDER BY channel_name ASC
        """
        df = config.read_sql(sql)
        if not df.empty and "Video Status" in df.columns:
            return df
    except Exception:
        pass

    # If empty or not yet generated or missing new columns, build dynamically
    update_all_channel_run_coverage()
    return config.read_sql("""
        SELECT
            channel_name AS "Channel Name",
            target_videos AS "Target Videos",
            selected_videos AS "Selected Videos",
            video_status AS "Video Status",
            target_comments AS "Target Comments",
            collected_comments AS "Collected Comments",
            comment_status AS "Comment Status",
            sentiment_status AS "Sentiment Status",
            transcription_candidates AS "Transcription Candidates",
            completed_transcripts AS "Completed Transcripts",
            transcription_status AS "Transcription Status",
            last_run_status AS "Last Run Status",
            missing_data_reason AS "Missing Data Reason"
        FROM channel_run_coverage
        ORDER BY channel_name ASC
    """)


def get_latest_workflow_runs(
    gh_token: str,
    gh_repo: str = "Harinath2112/MediaIntel",
    workflow_file: str = "mediaintel-cloud-pipeline.yml",
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Fetch recent runs of the cloud pipeline workflow from GitHub Actions."""
    if not gh_token:
        return []

    try:
        import requests
        url = f"https://api.github.com/repos/{gh_repo}/actions/workflows/{workflow_file}/runs"
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {gh_token}",
                "Accept": "application/vnd.github.v3+json",
            },
            params={"per_page": limit},
            timeout=8,
        )
        if resp.status_code == 200:
            return resp.json().get("workflow_runs", [])
    except Exception:
        pass
    return []


def get_workflow_run_jobs(
    run_id: int | str,
    gh_token: str,
    gh_repo: str = "Harinath2112/MediaIntel",
) -> list[dict[str, Any]]:
    """Fetch individual jobs for a specific workflow run."""
    if not gh_token or not run_id:
        return []

    try:
        import requests
        url = f"https://api.github.com/repos/{gh_repo}/actions/runs/{run_id}/jobs"
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {gh_token}",
                "Accept": "application/vnd.github.v3+json",
            },
            timeout=8,
        )
        if resp.status_code == 200:
            return resp.json().get("jobs", [])
    except Exception:
        pass
    return []


def dispatch_cloud_pipeline(
    gh_token: str,
    gh_repo: str = "Harinath2112/MediaIntel",
    workflow_file: str = "mediaintel-cloud-pipeline.yml",
    ref: str = "main",
    channel_ids: str | list[str] = "",
    video_limit: int = 10,
    comments_per_video: int = 100,
    transcripts_per_channel: int = 2,
    sentiment_limit: int = 0,
    run_whisper: bool = False,
    run_rpa: bool = True,
    send_email: bool = False,
    force: bool = False,
    run_ingestion: bool = True,
    run_ai: bool = True,
    force_retranscribe: bool = False,
    whisper_model: str = "tiny",
    video_id: str | None = None,
) -> dict[str, Any]:
    """Dispatch the GitHub Actions cloud pipeline with deduplication protection."""
    if not gh_token:
        return {
            "success": False,
            "status_code": 401,
            "message": "GitHub personal access token (GITHUB_TOKEN) is not configured in secrets.",
        }

    # Normalize channel_ids
    if isinstance(channel_ids, list):
        channel_id_param = ",".join(c.strip() for c in channel_ids if c.strip())
    else:
        channel_id_param = str(channel_ids or "").strip()

    import requests

    # Check for actively running workflows to prevent duplicate triggers
    if not force:
        runs = get_latest_workflow_runs(gh_token, gh_repo, workflow_file, limit=2)
        for r in runs:
            if r.get("status") in ("queued", "in_progress", "waiting"):
                return {
                    "success": False,
                    "status_code": 409,
                    "active_run": r,
                    "message": f"A cloud pipeline run (#{r.get('id')}) is already in progress ({r.get('status')}).",
                }

    url = f"https://api.github.com/repos/{gh_repo}/actions/workflows/{workflow_file}/dispatches"
    payload = {
        "ref": ref,
        "inputs": {
            "run_ingestion": bool(run_ingestion),
            "run_ai": bool(run_ai),
            "run_whisper": bool(run_whisper),
            "run_rpa": bool(run_rpa),
            "send_email": bool(send_email),
            "channel_id": channel_id_param,
            "video_limit": str(video_limit),
            "comments_per_video": str(comments_per_video),
            "transcripts_per_channel": str(transcripts_per_channel),
            "sentiment_limit": str(sentiment_limit),
            "force_retranscribe": bool(force_retranscribe),
            "whisper_model": str(whisper_model),
            "video_id": str(video_id or "").strip(),
        },
    }

    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {gh_token}",
                "Accept": "application/vnd.github.v3+json",
            },
            json=payload,
            timeout=10,
        )
        if resp.status_code == 204:
            return {
                "success": True,
                "status_code": 204,
                "message": f"Cloud intelligence pipeline dispatched to GitHub Actions on branch '{ref}'.",
            }
        else:
            return {
                "success": False,
                "status_code": resp.status_code,
                "message": f"GitHub Actions API returned HTTP {resp.status_code}: {resp.text[:200]}",
            }
    except Exception as exc:
        return {
            "success": False,
            "status_code": 500,
            "message": f"Failed to dispatch workflow: {type(exc).__name__}",
        }


def dispatch_transcription_retry(
    gh_token: str,
    gh_repo: str = "Harinath2112/MediaIntel",
    workflow_file: str = "mediaintel-cloud-pipeline.yml",
    ref: str = "main",
    video_id: str | None = None,
    channel_ids: list[str] | str | None = None,
    transcripts_per_channel: int = 2,
    whisper_model: str = "tiny",
    force_retranscribe: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Dispatch transcription-only workflow without re-running ingestion, AI sentiment, or RPA."""
    return dispatch_cloud_pipeline(
        gh_token=gh_token,
        gh_repo=gh_repo,
        workflow_file=workflow_file,
        ref=ref,
        channel_ids=channel_ids,
        video_id=video_id,
        transcripts_per_channel=transcripts_per_channel,
        run_ingestion=False,
        run_ai=False,
        run_whisper=True,
        run_rpa=False,
        send_email=False,
        force_retranscribe=force_retranscribe,
        whisper_model=whisper_model,
        force=force,
    )


def dispatch_rpa_reporting(
    gh_token: str,
    gh_repo: str = "Harinath2112/MediaIntel",
    workflow_file: str = "mediaintel-cloud-pipeline.yml",
    ref: str = "main",
    send_email: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Dispatch standalone Cloud RPA report generation (and optional alert email).

    Generates the 8-page PDF report, strategy report, and alert summary directly from
    the latest committed Supabase PostgreSQL records, skipping YouTube ingestion,
    RoBERTa sentiment analysis, and Whisper transcription.
    """
    return dispatch_cloud_pipeline(
        gh_token=gh_token,
        gh_repo=gh_repo,
        workflow_file=workflow_file,
        ref=ref,
        run_ingestion=False,
        run_ai=False,
        run_whisper=False,
        run_rpa=True,
        send_email=send_email,
        force=force,
    )


def check_channel_concurrency(
    channel_id: str,
    gh_token: str | None = None,
    gh_repo: str = "Harinath2112/MediaIntel",
) -> tuple[bool, str | None]:
    """Check if any active GitHub Actions pipeline or audio processing lease is currently running for this channel."""
    # 1. Check GitHub Actions runs if token is available
    if gh_token:
        try:
            runs = get_latest_workflow_runs(gh_token, gh_repo=gh_repo, limit=3)
            for r in runs:
                if r.get("status") in ("queued", "in_progress", "waiting"):
                    return True, f"GitHub Actions cloud pipeline run #{r.get('id')} is currently {r.get('status')}."
        except Exception:
            pass

    # 2. Check for active audio claims in database
    try:
        active_claims = config.read_sql("""
            SELECT COUNT(*) AS cnt
            FROM video_audio_assets a
            JOIN videos v ON v.video_id = a.video_id
            WHERE v.channel_id = :cid
              AND a.status = 'processing'
              AND a.lease_expires_at > CURRENT_TIMESTAMP
        """, {"cid": channel_id})
        if not active_claims.empty and int(active_claims.iloc[0]["cnt"]) > 0:
            return True, "Audio worker currently has an active processing lease on this channel's video(s)."
    except Exception:
        pass

    return False, None


def preview_channel_deletion(
    channel_id: str,
    gh_token: str | None = None,
) -> dict[str, Any]:
    """Inspect and return record counts across all 12 tables for a channel deletion preview.

    Non-destructive inspection before permanent deletion.
    """
    cid = (channel_id or "").strip()
    if not cid:
        return {"exists": False, "channel_id": cid, "error": "Invalid channel ID"}

    try:
        # 1. Fetch channel row
        ch_df = config.read_sql("SELECT channel_id, channel_name, is_tracked FROM channels WHERE channel_id = :cid", {"cid": cid})
        if ch_df.empty:
            return {
                "exists": False,
                "channel_id": cid,
                "channel_name": None,
                "message": f"Channel '{cid}' does not exist in database.",
            }

        cname = ch_df.iloc[0]["channel_name"]
        is_tracked = bool(ch_df.iloc[0]["is_tracked"])

        # 2. Concurrency check
        is_locked, lock_reason = check_channel_concurrency(cid, gh_token=gh_token)

        # 3. Query record counts across all dependent tables
        counts: dict[str, int] = {}

        v_df = config.read_sql("SELECT COUNT(*) AS cnt FROM videos WHERE channel_id = :cid", {"cid": cid})
        counts["videos"] = int(v_df.iloc[0]["cnt"]) if not v_df.empty else 0

        c_df = config.read_sql("""
            SELECT COUNT(*) AS cnt FROM comments
            WHERE video_id IN (SELECT video_id FROM videos WHERE channel_id = :cid)
        """, {"cid": cid})
        counts["comments"] = int(c_df.iloc[0]["cnt"]) if not c_df.empty else 0

        s_df = config.read_sql("""
            SELECT COUNT(*) AS cnt FROM sentiment_insights
            WHERE comment_id IN (
                SELECT c.comment_id FROM comments c
                JOIN videos v ON v.video_id = c.video_id
                WHERE v.channel_id = :cid
            )
        """, {"cid": cid})
        counts["sentiment_insights"] = int(s_df.iloc[0]["cnt"]) if not s_df.empty else 0

        asp_df = config.read_sql("""
            SELECT COUNT(*) AS cnt FROM aspect_sentiments
            WHERE comment_id IN (
                SELECT c.comment_id FROM comments c
                JOIN videos v ON v.video_id = c.video_id
                WHERE v.channel_id = :cid
            )
        """, {"cid": cid})
        counts["aspect_sentiments"] = int(asp_df.iloc[0]["cnt"]) if not asp_df.empty else 0

        # Detailed transcript inspection: raw table rows vs completed text vs pending audio
        tx_rows_df = config.read_sql("""
            SELECT video_id, source, transcript_text FROM transcripts
            WHERE video_id IN (SELECT video_id FROM videos WHERE channel_id = :cid)
        """, {"cid": cid})
        counts["transcripts"] = len(tx_rows_df)

        completed_tx = 0
        pending_audio_tx = 0
        active_completed_tx = 0

        if not tx_rows_df.empty:
            is_completed = (
                tx_rows_df["source"].fillna("").astype(str).str.lower().isin(["whisper", "faster_whisper", "captions", "youtube_manual_caption", "youtube_auto_caption"]) &
                tx_rows_df["transcript_text"].notna() &
                (tx_rows_df["transcript_text"].astype(str).str.strip() != "")
            )
            completed_tx = int(is_completed.sum())
            pending_audio_tx = len(tx_rows_df) - completed_tx

            # Active candidate transcripts (top 2 by view count in active 10)
            try:
                cand_vids_df = get_transcription_candidate_videos(2, channel_ids=[cid])
                cand_vids = set(cand_vids_df["video_id"].tolist()) if not cand_vids_df.empty else set()
                if cand_vids:
                    active_completed_tx = int((is_completed & tx_rows_df["video_id"].isin(cand_vids)).sum())
            except Exception:
                active_completed_tx = completed_tx

        vaa_df = config.read_sql("""
            SELECT bucket, object_path FROM video_audio_assets
            WHERE video_id IN (SELECT video_id FROM videos WHERE channel_id = :cid)
        """, {"cid": cid})
        counts["video_audio_assets"] = len(vaa_df)
        audio_paths = vaa_df["object_path"].tolist() if not vaa_df.empty and "object_path" in vaa_df.columns else []

        vsh_df = config.read_sql("""
            SELECT COUNT(*) AS cnt FROM video_stats_history
            WHERE video_id IN (SELECT video_id FROM videos WHERE channel_id = :cid)
        """, {"cid": cid})
        counts["video_stats_history"] = int(vsh_df.iloc[0]["cnt"]) if not vsh_df.empty else 0

        vr_df = config.read_sql("""
            SELECT COUNT(*) AS cnt FROM video_relevance
            WHERE video_id IN (SELECT video_id FROM videos WHERE channel_id = :cid)
        """, {"cid": cid})
        counts["video_relevance"] = int(vr_df.iloc[0]["cnt"]) if not vr_df.empty else 0

        aa_df = config.read_sql("""
            SELECT COUNT(*) AS cnt FROM anomaly_alerts
            WHERE channel_id = :cid
               OR video_id IN (SELECT video_id FROM videos WHERE channel_id = :cid)
        """, {"cid": cid})
        counts["anomaly_alerts"] = int(aa_df.iloc[0]["cnt"]) if not aa_df.empty else 0

        # Check remaining channels if this channel is deleted
        ch_count_df = config.read_sql("SELECT COUNT(*) AS cnt FROM channels WHERE channel_id != :cid", {"cid": cid})
        rem_chans = int(ch_count_df.iloc[0]["cnt"]) if not ch_count_df.empty else 0
        if rem_chans == 0:
            top_df = config.read_sql("SELECT COUNT(*) AS cnt FROM topic_suggestions")
        else:
            top_df = config.read_sql("""
                SELECT COUNT(*) AS cnt FROM topic_suggestions
                WHERE scope = :cid OR UPPER(scope) = 'ALL'
            """, {"cid": cid})
        counts["topic_suggestions"] = int(top_df.iloc[0]["cnt"]) if not top_df.empty else 0

        cov_df = config.read_sql("""
            SELECT COUNT(*) AS cnt FROM channel_run_coverage
            WHERE channel_id = :cid
        """, {"cid": cid})
        counts["channel_run_coverage"] = int(cov_df.iloc[0]["cnt"]) if not cov_df.empty else 0

        counts["channels"] = 1
        total_relational = sum(counts.values())

        return {
            "exists": True,
            "channel_id": cid,
            "channel_name": cname,
            "is_tracked": is_tracked,
            "record_counts": counts,
            "total_records": total_relational,
            "completed_transcripts": completed_tx,
            "pending_audio_records": pending_audio_tx,
            "active_set_transcripts": active_completed_tx,
            "storage_audio_objects": audio_paths,
            "storage_audio_count": len(audio_paths),
            "is_locked": is_locked,
            "lock_reason": lock_reason,
        }
    except Exception as exc:
        return {
            "exists": True,
            "channel_id": cid,
            "channel_name": cid,
            "is_tracked": True,
            "record_counts": {},
            "total_records": 0,
            "completed_transcripts": 0,
            "pending_audio_records": 0,
            "active_set_transcripts": 0,
            "storage_audio_objects": [],
            "storage_audio_count": 0,
            "is_locked": False,
            "lock_reason": None,
            "error": f"Preview notice: {type(exc).__name__}: {exc}",
        }


def delete_channel_permanently(
    channel_id: str,
    confirmation: str,
    admin_user: bool = True,
    gh_token: str | None = None,
    force_unlocked: bool = False,
) -> dict[str, Any]:
    """Permanently delete a channel and all its associated records in a single database transaction.

    Enforces:
    - Administrator authorization
    - Exact 'DELETE' typed confirmation
    - Concurrency safety (blocks deletion if a workflow or audio claim is active)
    - Full child-to-parent relational cascade in an atomic SQL transaction
    - Isolated channel scoping: Never deletes shared global topics (scope='all') or other channels
    - Safe Supabase Storage audio asset cleanup with retry support
    - Local target_channels.json synchronization
    - Idempotent execution
    """
    # 1. Admin authorization check
    if not admin_user:
        return {
            "success": False,
            "error": "Unauthorized: Administrator privileges are required to permanently delete channels.",
        }

    # 2. Confirmation check
    if (confirmation or "").strip() != "DELETE":
        return {
            "success": False,
            "error": "Safety confirmation failed: You must type 'DELETE' exactly in uppercase to confirm permanent deletion.",
        }

    cid = (channel_id or "").strip()
    if not cid:
        return {"success": False, "error": "Invalid channel ID provided."}

    # 3. Concurrency check
    is_locked, lock_reason = check_channel_concurrency(cid, gh_token=gh_token)
    if is_locked and not force_unlocked:
        return {
            "success": False,
            "error": f"Deletion blocked by concurrency safeguard: {lock_reason}",
        }

    # 4. Preview record counts before deletion
    preview = preview_channel_deletion(cid, gh_token=gh_token)
    if not preview.get("exists"):
        return {
            "success": True,
            "already_deleted": True,
            "channel_id": cid,
            "channel_name": preview.get("channel_name"),
            "message": f"Channel '{cid}' does not exist or has already been deleted.",
        }

    channel_name = preview["channel_name"]
    counts = preview["record_counts"]
    audio_paths = preview["storage_audio_objects"]

    # 5. Execute relational deletion in a single atomic transaction
    engine = config.get_engine()
    try:
        with engine.begin() as conn:
            # 1. aspect_sentiments
            conn.execute(text("""
                DELETE FROM aspect_sentiments
                WHERE comment_id IN (
                    SELECT c.comment_id FROM comments c
                    JOIN videos v ON v.video_id = c.video_id
                    WHERE v.channel_id = :cid
                )
            """), {"cid": cid})

            # 2. sentiment_insights
            conn.execute(text("""
                DELETE FROM sentiment_insights
                WHERE comment_id IN (
                    SELECT c.comment_id FROM comments c
                    JOIN videos v ON v.video_id = c.video_id
                    WHERE v.channel_id = :cid
                )
            """), {"cid": cid})

            # 3. comments
            conn.execute(text("""
                DELETE FROM comments
                WHERE video_id IN (
                    SELECT v.video_id FROM videos v
                    WHERE v.channel_id = :cid
                )
            """), {"cid": cid})

            # 4. transcripts
            conn.execute(text("""
                DELETE FROM transcripts
                WHERE video_id IN (
                    SELECT v.video_id FROM videos v
                    WHERE v.channel_id = :cid
                )
            """), {"cid": cid})

            # 5. video_audio_assets
            conn.execute(text("""
                DELETE FROM video_audio_assets
                WHERE video_id IN (
                    SELECT v.video_id FROM videos v
                    WHERE v.channel_id = :cid
                )
            """), {"cid": cid})

            # 6. video_stats_history
            conn.execute(text("""
                DELETE FROM video_stats_history
                WHERE video_id IN (
                    SELECT v.video_id FROM videos v
                    WHERE v.channel_id = :cid
                )
            """), {"cid": cid})

            # 7. video_relevance
            conn.execute(text("""
                DELETE FROM video_relevance
                WHERE video_id IN (
                    SELECT v.video_id FROM videos v
                    WHERE v.channel_id = :cid
                )
            """), {"cid": cid})

            # 8. anomaly_alerts
            conn.execute(text("""
                DELETE FROM anomaly_alerts
                WHERE channel_id = :cid
                   OR video_id IN (
                       SELECT v.video_id FROM videos v
                       WHERE v.channel_id = :cid
                   )
            """), {"cid": cid})

            # 9. topic_suggestions
            # Check remaining channels
            rem_count = conn.execute(text("SELECT COUNT(*) FROM channels WHERE channel_id != :cid"), {"cid": cid}).scalar()
            if int(rem_count or 0) == 0:
                # Last channel: wipe ALL topic suggestions completely (channel-scoped and global)
                conn.execute(text("DELETE FROM topic_suggestions"))
            else:
                # Invalidate channel-scoped topics and global suggestions that incorporated this channel
                conn.execute(text("""
                    DELETE FROM topic_suggestions
                    WHERE scope = :cid OR UPPER(scope) = 'ALL'
                """), {"cid": cid})

            # 10. videos
            conn.execute(text("""
                DELETE FROM videos
                WHERE channel_id = :cid
            """), {"cid": cid})

            # 11. channel_run_coverage
            conn.execute(text("""
                DELETE FROM channel_run_coverage
                WHERE channel_id = :cid
            """), {"cid": cid})

            # 12. channels
            conn.execute(text("""
                DELETE FROM channels
                WHERE channel_id = :cid
            """), {"cid": cid})

    except Exception as exc:
        return {
            "success": False,
            "error": f"Database transaction failed and was completely rolled back: {type(exc).__name__}: {exc}",
        }

    # 6. Synchronize target_channels.json if it exists locally
    try:
        if config.CHANNELS_FILE.exists():
            current_json = json.loads(config.CHANNELS_FILE.read_text(encoding="utf-8"))
            if isinstance(current_json, list):
                updated_json = [item for item in current_json if (item.get("channel_id") or "").strip() != cid]
                config.CHANNELS_FILE.write_text(json.dumps(updated_json, indent=2), encoding="utf-8")
    except Exception:
        pass

    # 7. Safe Supabase Storage audio cleanup
    storage_result = {"success": True, "deleted_count": 0, "failed_paths": [], "error": None}
    if audio_paths:
        try:
            from core import storage_audio
            storage_result = storage_audio.delete_storage_audio_objects("mediaintel-audio", audio_paths)
        except Exception as exc:
            storage_result = {
                "success": False,
                "deleted_count": 0,
                "failed_paths": audio_paths,
                "error": f"Storage cleanup exception: {type(exc).__name__}: {exc}",
            }

    return {
        "success": True,
        "already_deleted": False,
        "channel_id": cid,
        "channel_name": channel_name,
        "deleted_counts": counts,
        "storage_cleanup": storage_result,
        "message": f"Channel '{channel_name}' ({cid}) and all associated records permanently deleted.",
    }

