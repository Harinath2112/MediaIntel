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
    """Retrieve full status, video, comment, and transcript telemetry for all registered channels."""
    sql = """
        SELECT
            c.channel_name AS "Channel",
            c.channel_id AS "Channel ID",
            CASE WHEN c.is_tracked THEN 'Active' ELSE 'Inactive' END AS "Tracking",
            COUNT(DISTINCT v.video_id) AS "Videos",
            COUNT(DISTINCT m.comment_id) AS "Comments",
            COUNT(DISTINCT t.video_id) AS "Transcripts",
            MAX(v.published_at) AS "Latest Video"
        FROM channels c
        LEFT JOIN videos v ON v.channel_id = c.channel_id
        LEFT JOIN comments m ON m.video_id = v.video_id
        LEFT JOIN transcripts t ON t.video_id = v.video_id AND t.source IN ('whisper', 'captions')
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
    """Rank the 10 selected active videos per channel by view count and select top 2.
    
    Deterministic ranking: view_count DESC, published_at DESC, video_id ASC.
    """
    active_v = get_active_analysis_videos_df(10, channel_ids=channel_ids)
    if active_v.empty:
        return active_v

    # Deterministic ranking within each channel: view_count DESC, published_at DESC, video_id ASC
    sorted_df = active_v.sort_values(
        by=["channel_id", "view_count", "published_at", "video_id"],
        ascending=[True, False, False, True]
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

    # Load comments and transcripts for active set
    comments_df = config.read_sql("SELECT comment_id, video_id FROM comments")
    transcripts_df = config.read_sql("SELECT video_id, source, transcript_text FROM transcripts")

    active_comments = comments_df[comments_df["video_id"].isin(active_vids)] if not comments_df.empty else comments_df
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

        c_cand = candidate_videos_df[candidate_videos_df["channel_id"] == cid] if not candidate_videos_df.empty else candidate_videos_df
        trans_candidates = min(selected_v, 2)
        c_cand_set = set(c_cand["video_id"].tolist()) if not c_cand.empty else set()

        completed_tx = 0
        if not candidate_transcripts.empty and c_cand_set:
            completed_tx = len(candidate_transcripts[
                candidate_transcripts["video_id"].isin(c_cand_set) &
                candidate_transcripts["source"].isin(["whisper", "captions"]) &
                candidate_transcripts["transcript_text"].notna() &
                (candidate_transcripts["transcript_text"].str.strip() != "")
            ])

        reasons = []
        if selected_v == 0:
            reasons.append("No public videos found on channel")
        elif selected_v < target_v:
            reasons.append(f"Channel published only {selected_v} eligible videos")
        if selected_v > 0 and collected_c == 0:
            reasons.append("Comments disabled or 0 comments published")
        if trans_candidates > completed_tx:
            reasons.append("Audio unavailable in Supabase Storage")

        missing_reason = "; ".join(reasons) if reasons else "None"

        if selected_v == 0:
            status = "No Data"
        elif completed_tx == trans_candidates and selected_v >= target_v:
            status = "Completed"
        elif completed_tx < trans_candidates and selected_v > 0:
            status = "Pending Audio"
        else:
            status = "Partial"

        coverage_rows.append({
            "channel_id": cid,
            "channel_name": cname,
            "target_videos": target_v,
            "selected_videos": selected_v,
            "target_comments": target_c,
            "collected_comments": collected_c,
            "transcription_candidates": trans_candidates,
            "completed_transcripts": completed_tx,
            "missing_data_reason": missing_reason,
            "last_run_status": status,
            "updated_at": now,
        })

    if coverage_rows:
        cols = [
            "channel_id", "channel_name", "target_videos", "selected_videos",
            "target_comments", "collected_comments", "transcription_candidates",
            "completed_transcripts", "missing_data_reason", "last_run_status", "updated_at"
        ]
        upsert_stmt = config.upsert_sql(
            "channel_run_coverage",
            cols,
            ["channel_id"],
            ["channel_name", "target_videos", "selected_videos", "target_comments",
             "collected_comments", "transcription_candidates", "completed_transcripts",
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
                target_comments AS "Target Comments",
                collected_comments AS "Collected Comments",
                transcription_candidates AS "Transcription Candidates",
                completed_transcripts AS "Completed Transcripts",
                missing_data_reason AS "Missing Data Reason",
                last_run_status AS "Last Run Status"
            FROM channel_run_coverage
            ORDER BY channel_name ASC
        """
        df = config.read_sql(sql)
        if not df.empty:
            return df
    except Exception:
        pass

    # If empty or not yet generated, build dynamically
    update_all_channel_run_coverage()
    return config.read_sql("""
        SELECT
            channel_name AS "Channel Name",
            target_videos AS "Target Videos",
            selected_videos AS "Selected Videos",
            target_comments AS "Target Comments",
            collected_comments AS "Collected Comments",
            transcription_candidates AS "Transcription Candidates",
            completed_transcripts AS "Completed Transcripts",
            missing_data_reason AS "Missing Data Reason",
            last_run_status AS "Last Run Status"
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
    sentiment_limit: int = 800,
    run_whisper: bool = True,
    force: bool = False,
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
            "run_ingestion": True,
            "run_ai": True,
            "run_whisper": bool(run_whisper),
            "channel_id": channel_id_param,
            "video_limit": str(video_limit),
            "comments_per_video": str(comments_per_video),
            "transcripts_per_channel": str(transcripts_per_channel),
            "sentiment_limit": str(sentiment_limit),
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
