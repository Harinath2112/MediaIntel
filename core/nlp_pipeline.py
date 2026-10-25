"""Stage 3 - Aspect-based sentiment analysis with RoBERTa.

For every new comment:
  1. split into clauses
  2. RoBERTa (XLM-RoBERTa) predicts Negative / Neutral / Positive probabilities for each clause
  3. each clause is matched to an aspect (Content Quality, Engagement, Relevance, Feature Request,
     Pricing / Cost, Technical Issue, General Feedback)
  4. results are written to `sentiment_insights` (one row per comment) and `aspect_sentiments`
     (one row per comment x aspect, with the clause that is the evidence for it)

All collected comments are analysed unless MAX_COMMENTS_PER_VIDEO caps it (top-liked N per video).

Incremental: only comments that are new - or were analysed by a different model (e.g. the old VADER
rows) - are processed, in chunks, so progress is saved even if you stop the run.

Usage:  python core/nlp_pipeline.py [--limit N] [--reprocess-all] [--preview N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402
from core import aspects as asp  # noqa: E402
from core import nlp_models  # noqa: E402
from sqlalchemy import bindparam, text  # noqa: E402

CHUNK = config.env_int("NLP_CHUNK", 200)    # comments per database commit
MAX_PER_VIDEO = config.setting_int("MAX_COMMENTS_PER_VIDEO")   # 0 = analyse all collected comments; N = most-liked N per video


def analyse_comments(comments: list[str], sentiment, matcher) -> list[dict]:
    """Pure function: texts in -> analysed dicts out (no database access).

    Identical clauses (e.g. the many "nice video sir") are scored once and reused - a free speed-up when every
    comment is analysed.
    """
    segs_per_comment = [asp.segment(c) for c in comments]
    unique = list(dict.fromkeys(s for segs in segs_per_comment for s in segs))
    index = {s: i for i, s in enumerate(unique)}
    probs_u = sentiment.predict_proba(unique)
    hits_u = matcher.classify(unique)

    results = []
    for segs in segs_per_comment:
        p = np.array([probs_u[index[s]] for s in segs])
        w = np.array([max(len(s), 1) for s in segs], dtype=float)

        overall = (p * w[:, None]).sum(axis=0) / w.sum()
        label_idx = int(overall.argmax())

        per_aspect: dict[str, dict] = {}
        for seg, prob, weight in zip(segs, p, w):
            for aspect, _score in hits_u[index[seg]]:
                acc = per_aspect.setdefault(aspect, {"psum": np.zeros(3), "w": 0.0, "evidence": seg, "ew": 0.0})
                acc["psum"] += prob * weight
                acc["w"] += weight
                if weight > acc["ew"]:                       # keep the clause that carries most of this aspect
                    acc["evidence"], acc["ew"] = seg, weight

        aspect_rows = []
        for aspect, acc in per_aspect.items():
            pa = acc["psum"] / acc["w"]
            aspect_rows.append({
                "aspect": aspect,
                "sentiment_label": nlp_models.CANONICAL[int(pa.argmax())],
                "polarity": float(pa[2] - pa[0]),
                "confidence": float(pa.max()),
                "weight": float(acc["w"]),
                "evidence": acc["evidence"][:300],
            })
        specific = [r for r in aspect_rows if r["aspect"] != asp.GENERAL]
        primary = max(specific or aspect_rows, key=lambda r: r["weight"])["aspect"]

        results.append({
            "sentiment_label": nlp_models.CANONICAL[label_idx],
            "sentiment_score": float(overall.max()),
            "polarity": float(overall[2] - overall[0]),
            "aspect_category": primary,
            "aspects": aspect_rows,
        })
    return results


def _pending(limit: int | None, reprocess_all: bool, model_name: str):
    """Comments still to analyse: every collected comment that was never analysed or was analysed by a different
    model (e.g. rows left over from an older version). Optional cap: MAX_COMMENTS_PER_VIDEO > 0 keeps only the
    most-liked N comments of each video."""
    df = config.read_sql("""
        SELECT c.comment_id, c.video_id, c.comment_text, COALESCE(c.like_count, 0) AS like_count,
               c.published_at, s.model_name
        FROM comments c
        JOIN videos v ON v.video_id = c.video_id
        JOIN channels ch ON ch.channel_id = v.channel_id
        LEFT JOIN sentiment_insights s ON s.comment_id = c.comment_id
        WHERE ch.is_tracked = TRUE""")
    if df.empty:
        return df
    df = df.sort_values(["video_id", "like_count", "published_at"], ascending=[True, False, False])
    if MAX_PER_VIDEO > 0:
        df["rk"] = df.groupby("video_id").cumcount()
        df = df[df["rk"] < MAX_PER_VIDEO]
    if not reprocess_all:
        df = df[df["model_name"].isna() | (df["model_name"] != model_name)]
    df = df.drop_duplicates("comment_id").sort_values("comment_id")
    if limit is not None and int(limit) > 0:
        return df.head(int(limit))
    return df


def _write_chunk(ids: list[str], results: list[dict], model_name: str) -> None:
    engine = config.get_engine()
    with engine.begin() as conn:
        for table in ("aspect_sentiments", "sentiment_insights"):
            conn.execute(text(f"DELETE FROM {table} WHERE comment_id IN :ids")
                         .bindparams(bindparam("ids", expanding=True)), {"ids": ids})
        conn.execute(text("""
            INSERT INTO sentiment_insights (comment_id, sentiment_label, sentiment_score, polarity, aspect_category, model_name)
            VALUES (:cid, :label, :score, :pol, :aspect, :model)"""), [
            {"cid": cid, "label": r["sentiment_label"], "score": r["sentiment_score"], "pol": r["polarity"],
             "aspect": r["aspect_category"], "model": model_name} for cid, r in zip(ids, results)])
        aspect_rows = [{"cid": cid, "a": a["aspect"], "l": a["sentiment_label"], "p": a["polarity"],
                        "c": a["confidence"], "e": a["evidence"]}
                       for cid, r in zip(ids, results) for a in r["aspects"]]
        if aspect_rows:
            conn.execute(text("""
                INSERT INTO aspect_sentiments (comment_id, aspect, sentiment_label, polarity, confidence, evidence)
                VALUES (:cid, :a, :l, :p, :c, :e)"""), aspect_rows)
        conn.execute(text("UPDATE comments SET is_processed = TRUE WHERE comment_id IN :ids")
                     .bindparams(bindparam("ids", expanding=True)), {"ids": ids})


def run_ai_pipeline(
    limit: int | None = None,
    reprocess_all: bool = False,
    preview: int = 0,
    time_budget_min: float | None = None,
) -> int:
    sentiment = nlp_models.get_sentiment_model()
    model_name = getattr(sentiment, "name", nlp_models.SENTIMENT_MODEL)

    print("Fetching comments that need (re)analysis...")
    df = _pending(preview or limit, reprocess_all and not preview, model_name)
    if df.empty:
        print("No new comments to analyse - everything is up to date.")
        return 0

    matcher = asp.AspectMatcher(nlp_models.get_embedder())
    print(f"{len(df)} comment(s) queued for RoBERTa/ABSA analysis ({model_name}).")

    if preview:
        sample = df.sample(min(preview, len(df)), random_state=1)
        res = analyse_comments(sample["comment_text"].tolist(), sentiment, matcher)
        for text_, r in zip(sample["comment_text"], res):
            print(f"\n> {str(text_)[:110]!r}\n  overall: {r['sentiment_label']} ({r['sentiment_score']:.2f})   "
                  f"primary aspect: {r['aspect_category']}")
            for a in r["aspects"]:
                print(f"     - {a['aspect']:<16} {a['sentiment_label']:<8} polarity {a['polarity']:+.2f}   <- \"{a['evidence'][:60]}\"")
        return len(sample)

    budget_minutes = (
        time_budget_min
        if time_budget_min is not None
        else config.env_float("NLP_TIME_BUDGET_MIN", 24.0)
    )
    time_budget_sec = budget_minutes * 60.0 if budget_minutes and budget_minutes > 0 else 0.0

    done, started = 0, time.time()
    for start in range(0, len(df), CHUNK):
        elapsed = time.time() - started
        if time_budget_sec > 0 and elapsed >= time_budget_sec and done > 0:
            print(f"\n⚠️ Safe time budget reached ({elapsed / 60.0:.1f}m / {budget_minutes:.1f}m). "
                  f"Processed and committed {done}/{len(df)} comments. Remaining {len(df) - done} comments will be processed in the next run.")
            break

        chunk = df.iloc[start:start + CHUNK]
        results = analyse_comments(chunk["comment_text"].astype(str).tolist(), sentiment, matcher)
        _write_chunk(chunk["comment_id"].tolist(), results, model_name)
        done += len(chunk)
        rate = done / max(time.time() - started, 1e-6)
        print(f"  analysed {done}/{len(df)} comments ({rate:.0f}/s)")
    print(f"Analysed and stored {done} comments with aspect-level sentiment.")
    try:
        from core import channel_manager
        channel_manager.update_all_channel_run_coverage()
    except Exception as exc:
        print(f"Notice: Coverage telemetry update: {type(exc).__name__} ({exc})")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="maximum comments to analyze (0 or omitted = all pending)")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="maximum minutes to spend before graceful incremental exit (default: 24m)")
    ap.add_argument("--reprocess-all", action="store_true", help="re-run every comment, even if already analysed")
    ap.add_argument("--preview", type=int, default=0, metavar="N",
                    help="show predictions for N random comments without saving (sanity check)")
    a = ap.parse_args()
    run_ai_pipeline(a.limit, a.reprocess_all, a.preview, time_budget_min=a.time_budget)
