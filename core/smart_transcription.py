"""MediaIntel — Smart 15-Minute Transcription Strategy.

Reusable duration-based transcription strategy:
  Rule 1 (Videos < 15 minutes / < 900s):
    - Complete YouTube captions preferred.
    - If captions unavailable, transcribe full video audio with Faster-Whisper.
    - Recorded as full transcript (is_truncated=False).

  Rule 2 (Videos >= 15 minutes / >= 900s):
    - Complete YouTube captions preferred (full transcript, no Whisper).
    - If captions unavailable, select a maximum of 15 minutes (900s) of representative audio.
    - Prefer informative chapters / speech segments when identifiable.
    - Otherwise distribute segments across beginning, middle, and end.
    - Omit intros, outros, and sponsor advertisements where reasonably detectable.
    - Combine segment transcripts in chronological order.
    - Recorded as partial transcript (is_truncated=True).

Important:
  Segment selections are heuristic representative samples and are not claimed
  to be objectively optimal or exhaustive without content evidence.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Thresholds
SMART_TRANSCRIPTION_THRESHOLD_SEC = 900  # 15 minutes
MAX_PARTIAL_AUDIO_SEC = 900              # 15 minutes

# Regex patterns for non-content, promos, intros, outros, sponsors
EXCLUDED_CHAPTER_PATTERNS = [
    r"\bintro\b",
    r"\bintroduction\b",
    r"\bwelcome\b",
    r"\bchannel intro\b",
    r"\bsponsor\b",
    r"\bsponsored\b",
    r"\bad\b",
    r"\bpromo\b",
    r"\badvertisement\b",
    r"\bpromotion\b",
    r"\boutro\b",
    r"\bcredits\b",
    r"\bthanks\b",
    r"\bthank you\b",
    r"\bsubscribe\b",
    r"\blike and subscribe\b",
    r"\bpatreon\b",
    r"\bdiscord\b",
    r"\bsocials\b",
    r"\blinks\b",
    r"\bmerch\b",
    r"\bmusic\b",
    r"\bend screen\b",
]
_EXCLUDE_RE = re.compile("|".join(EXCLUDED_CHAPTER_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class AudioSegment:
    """An audio segment selected for transcription."""
    start_sec: float
    end_sec: float
    duration_sec: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "duration_sec": round(self.duration_sec, 2),
            "label": self.label,
        }


@dataclass
class TranscriptionPlan:
    """Deterministic transcription plan based on video duration and metadata."""
    video_id: str
    video_duration_sec: int
    is_partial: bool
    segments: list[AudioSegment]
    total_audio_seconds: int
    strategy: str  # "full_video" | "chapters" | "distributed_temporal"
    description: str

    @property
    def is_truncated(self) -> bool:
        """Accurate truncated/partial indicator:
        - If 100% of the video audio is transcribed: False
        - If any meaningful portion of the video is omitted: True
        """
        if not self.is_partial:
            return False
        # If a single segment covers 100% of the video duration without gaps
        if (
            len(self.segments) == 1
            and self.segments[0].start_sec <= 0.5
            and self.segments[0].end_sec >= self.video_duration_sec - 0.5
            and self.total_audio_seconds >= self.video_duration_sec - 0.5
        ):
            return False
        return True

    def get_segment_timestamps(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.segments]

    def format_summary(self) -> str:
        parts = []
        for s in self.segments:
            parts.append(f"{format_seconds(s.start_sec)}-{format_seconds(s.end_sec)} ({s.label})")
        return "; ".join(parts)


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    s = max(0, int(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_description_timestamps(description: str | None) -> list[dict[str, Any]]:
    """Extract chapter timestamps formatted as (MM:SS or HH:MM:SS Title) from video description."""
    if not description:
        return []

    lines = description.splitlines()
    ts_pattern = re.compile(r"(?:^|\s)(?:(\d{1,2}):)?(\d{2}):(\d{2})(?:\s*[-–:]\s*|\s+)(.+)$")
    raw_entries = []

    for line in lines:
        match = ts_pattern.search(line.strip())
        if match:
            h_str, m_str, s_str, title = match.groups()
            h = int(h_str) if h_str else 0
            m = int(m_str)
            s = int(s_str)
            start_sec = h * 3600 + m * 60 + s
            clean_title = title.strip()
            if clean_title:
                raw_entries.append((start_sec, clean_title))

    if len(raw_entries) < 2:
        return []

    # Sort and convert to chapter dicts
    raw_entries.sort(key=lambda x: x[0])
    chapters = []
    for i in range(len(raw_entries)):
        start_sec, title = raw_entries[i]
        end_sec = raw_entries[i + 1][0] if i + 1 < len(raw_entries) else start_sec + 300.0
        if end_sec > start_sec:
            chapters.append({
                "title": clean_title,
                "start_time": float(start_sec),
                "end_time": float(end_sec),
            })
    return chapters


def filter_informative_chapters(chapters: list[dict[str, Any]] | None, video_duration: float) -> list[dict[str, Any]]:
    """Filter out intro, outro, sponsor, and promotional chapters."""
    if not chapters:
        return []

    informative = []
    for ch in chapters:
        title = str(ch.get("title") or "").strip()
        start = float(ch.get("start_time", 0.0))
        end = float(ch.get("end_time", 0.0))
        if end <= start or start >= video_duration:
            continue
        end = min(end, video_duration)
        duration = end - start
        if duration < 5.0:  # Ignore trivial blips
            continue

        if _EXCLUDE_RE.search(title):
            continue

        informative.append({
            "title": title,
            "start_time": start,
            "end_time": end,
            "duration": duration,
        })

    return informative


def plan_transcription_segments(
    video_duration_sec: int | float | None,
    chapters: list[dict[str, Any]] | None = None,
    description: str | None = None,
    max_partial_sec: int = MAX_PARTIAL_AUDIO_SEC,
    video_id: str = "",
    full_if_within_budget: bool = False,
) -> TranscriptionPlan:
    """Plan audio segments for transcription following MediaIntel Smart 15-Minute Strategy.

    - Videos < 900s: full video (1 segment, is_partial=False, is_truncated=False).
    - Videos exactly 900s:
        * If 100% of the audio is transcribed (full_if_within_budget=True):
          1 complete segment [0, 900], is_partial=False, is_truncated=False.
        * If any meaningful portion is omitted (intro/outro trimmed or chapters):
          representative segments <= 900s, is_partial=True, is_truncated=True.
    - Videos > 900s: max 15 minutes (<= 900s) representative audio (is_partial=True, is_truncated=True).
      Prefers informative chapters if available; otherwise distributes across
      beginning, middle, and end, omitting intros/outros.
    """
    if video_duration_sec is None or video_duration_sec <= 0:
        duration = float(max_partial_sec)
        is_unknown_duration = True
    else:
        duration = float(video_duration_sec)
        is_unknown_duration = False

    # Rule 1: Videos where 100% of the audio is transcribed within budget without omissions
    if not is_unknown_duration and (duration < SMART_TRANSCRIPTION_THRESHOLD_SEC or (duration <= SMART_TRANSCRIPTION_THRESHOLD_SEC and full_if_within_budget and not chapters)):
        seg = AudioSegment(
            start_sec=0.0,
            end_sec=duration,
            duration_sec=duration,
            label="Complete video",
        )
        return TranscriptionPlan(
            video_id=video_id,
            video_duration_sec=int(duration),
            is_partial=False,
            segments=[seg],
            total_audio_seconds=int(duration),
            strategy="full_video",
            description=f"Complete video transcription ({int(duration)}s <= 15 minutes).",
        )

    # Rule 2: Videos 15 minutes or longer (>= 900 seconds)
    budget = float(min(MAX_PARTIAL_AUDIO_SEC, max_partial_sec, duration))

    # Check for informative chapters
    candidate_chapters = chapters or parse_description_timestamps(description)
    informative_chapters = filter_informative_chapters(candidate_chapters, duration)

    if len(informative_chapters) >= 2:
        # Chapter-based representative selection
        # Select up to 3-4 chapters distributed across the timeline
        n_ch = len(informative_chapters)
        if n_ch <= 3:
            selected_ch = informative_chapters
        else:
            # Pick first informative chapter, middle chapter(s), and final content chapter
            mid_idx = n_ch // 2
            selected_ch = [
                informative_chapters[0],
                informative_chapters[mid_idx],
                informative_chapters[-1],
            ]

        # Allocate budget evenly across selected chapters
        per_ch_budget = budget / len(selected_ch)
        segments: list[AudioSegment] = []

        for ch in selected_ch:
            ch_start = ch["start_time"]
            ch_end = ch["end_time"]
            ch_dur = ch_end - ch_start
            take_dur = min(per_ch_budget, ch_dur)
            title = ch.get("title") or "Chapter"
            segments.append(
                AudioSegment(
                    start_sec=round(ch_start, 2),
                    end_sec=round(ch_start + take_dur, 2),
                    duration_sec=round(take_dur, 2),
                    label=f"Chapter: {title[:40]}",
                )
            )

        # Sort and ensure non-overlapping
        segments.sort(key=lambda s: s.start_sec)
        total_sec = sum(s.duration_sec for s in segments)
        return TranscriptionPlan(
            video_id=video_id,
            video_duration_sec=int(duration),
            is_partial=True,
            segments=segments,
            total_audio_seconds=int(total_sec),
            strategy="chapters",
            description=(
                f"Partial transcript (>= 15 minutes): {len(segments)} representative chapter segments "
                f"totaling {int(total_sec)}s (selection based on identified video chapters; intros/outros/ads omitted)."
            ),
        )

    # Distributed temporal sampling across Beginning, Middle, and End
    # Intro/outro trimming
    intro_skip = min(30.0, max(0.0, 0.02 * duration))
    outro_skip = min(45.0, max(0.0, 0.02 * duration))
    content_end = max(intro_skip + 1.0, duration - outro_skip)
    effective_dur = content_end - intro_skip

    if effective_dur <= budget:
        # If trimming intros/outros leaves less than or equal to 900s, take 3 even partitions
        seg_dur = effective_dur / 3.0
        s1 = AudioSegment(
            start_sec=round(intro_skip, 2),
            end_sec=round(intro_skip + seg_dur, 2),
            duration_sec=round(seg_dur, 2),
            label="Beginning (intro omitted)",
        )
        s2 = AudioSegment(
            start_sec=round(intro_skip + seg_dur, 2),
            end_sec=round(intro_skip + 2 * seg_dur, 2),
            duration_sec=round(seg_dur, 2),
            label="Middle / core transition",
        )
        s3 = AudioSegment(
            start_sec=round(intro_skip + 2 * seg_dur, 2),
            end_sec=round(content_end, 2),
            duration_sec=round(content_end - (intro_skip + 2 * seg_dur), 2),
            label="End / synthesis (outro omitted)",
        )
        segments = [s1, s2, s3]
    else:
        # 3 representative segments of 300s each across Beginning, Middle, End
        target_per_seg = budget / 3.0  # 300.0 seconds

        # 1. Beginning
        start_1 = intro_skip
        end_1 = start_1 + target_per_seg
        s1 = AudioSegment(
            start_sec=round(start_1, 2),
            end_sec=round(end_1, 2),
            duration_sec=round(target_per_seg, 2),
            label="Beginning (intro omitted)",
        )

        # 3. End
        end_3 = content_end
        start_3 = max(end_1 + 30.0, end_3 - target_per_seg)
        s3 = AudioSegment(
            start_sec=round(start_3, 2),
            end_sec=round(end_3, 2),
            duration_sec=round(end_3 - start_3, 2),
            label="End / synthesis (outro omitted)",
        )

        # 2. Middle
        mid_center = (start_1 + end_3) / 2.0
        start_2 = max(end_1 + 15.0, mid_center - (target_per_seg / 2.0))
        end_2 = min(start_3 - 15.0, start_2 + target_per_seg)
        dur_2 = end_2 - start_2
        if dur_2 < 10.0:
            # Rebalance evenly if video is very close to 900s
            span = (content_end - intro_skip) / 3.0
            start_1 = intro_skip
            end_1 = start_1 + span
            start_2 = end_1
            end_2 = start_2 + span
            start_3 = end_2
            end_3 = content_end
            s1 = AudioSegment(round(start_1, 2), round(end_1, 2), round(end_1 - start_1, 2), "Beginning (intro omitted)")
            s2 = AudioSegment(round(start_2, 2), round(end_2, 2), round(end_2 - start_2, 2), "Middle / core transition")
            s3 = AudioSegment(round(start_3, 2), round(end_3, 2), round(end_3 - start_3, 2), "End / synthesis (outro omitted)")
        else:
            s2 = AudioSegment(
                start_sec=round(start_2, 2),
                end_sec=round(end_2, 2),
                duration_sec=round(dur_2, 2),
                label="Middle / core transition",
            )

        segments = [s1, s2, s3]

    # Guarantee total audio <= 900s, strictly chronological, non-overlapping
    segments.sort(key=lambda s: s.start_sec)
    total_audio = sum(s.duration_sec for s in segments)
    if total_audio > max_partial_sec:
        # Scale back last segment if needed
        overflow = total_audio - max_partial_sec
        last = segments[-1]
        segments[-1] = AudioSegment(
            start_sec=last.start_sec,
            end_sec=round(last.end_sec - overflow, 2),
            duration_sec=round(last.duration_sec - overflow, 2),
            label=last.label,
        )
        total_audio = sum(s.duration_sec for s in segments)

    return TranscriptionPlan(
        video_id=video_id,
        video_duration_sec=int(duration),
        is_partial=True,
        segments=segments,
        total_audio_seconds=int(total_audio),
        strategy="distributed_temporal",
        description=(
            f"Partial transcript (>= 15 minutes): 3 representative temporal segments "
            f"across beginning, middle, and end totaling {int(total_audio)}s "
            f"(intros/outros omitted; selection is heuristic and not claimed to be objectively exhaustive)."
        ),
    )


def combine_segment_transcripts(
    segment_results: list[tuple[AudioSegment, str]],
    plan: TranscriptionPlan,
) -> str:
    """Combine segment transcripts chronologically with metadata headers for partial transcripts."""
    if not plan.is_truncated:
        # Full video (100% audio transcribed): return direct transcription text
        texts = [t.strip() for _, t in segment_results if t and t.strip()]
        return " ".join(texts).strip()

    header = (
        f"[PARTIAL TRANSCRIPT — Processed: {plan.total_audio_seconds}s of {plan.video_duration_sec}s total duration]\n"
        f"[Selected Segments: {plan.format_summary()}]\n\n"
    )

    body_sections = []
    for seg, text in segment_results:
        t_clean = text.strip()
        if t_clean:
            time_range = f"[{format_seconds(seg.start_sec)} - {format_seconds(seg.end_sec)}]"
            body_sections.append(f"{time_range} {seg.label}:\n{t_clean}")

    if not body_sections:
        return ""

    return header + "\n\n".join(body_sections)


def slice_pcm_for_plan(
    pcm_array: np.ndarray,
    plan: TranscriptionPlan,
    sample_rate: int = 16000,
) -> list[tuple[AudioSegment, np.ndarray]]:
    """Slice an in-memory PCM array according to the transcription plan."""
    results = []
    total_samples = len(pcm_array)

    for seg in plan.segments:
        start_idx = min(total_samples, int(seg.start_sec * sample_rate))
        end_idx = min(total_samples, int(seg.end_sec * sample_rate))
        if end_idx > start_idx:
            chunk = pcm_array[start_idx:end_idx]
            results.append((seg, chunk))
        else:
            # Empty fallback chunk if bounds exceed array
            results.append((seg, np.zeros(0, dtype=pcm_array.dtype)))

    return results
