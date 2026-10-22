"""Aspect taxonomy + clause segmentation for aspect-based sentiment analysis (ABSA).

A comment such as "Explanation is great but audio is terrible" is split into clauses; every
clause is matched to the aspect it talks about (multilingual embeddings + keyword evidence) and
gets its own RoBERTa sentiment - so the platform can report *"Content Quality: Positive,
Technical Issue: Negative"* for that single comment.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402

GENERAL = "General Feedback"
ASPECT_MIN_SIM = config.env_float("ASPECT_MIN_SIM", 0.30)   # below this a clause counts as general feedback
KEYWORD_BOOST = 0.15
MULTI_MARGIN = 0.06      # a second aspect must score this close to the best one
MAX_ASPECTS = 2          # aspects reported per clause
GENERAL_MARGIN = 0.10    # General Feedback wins only when it clearly beats every specific aspect

# Prototype sentences (English + Hinglish) per aspect; a clause is compared with every one of them.
PROTOTYPES: dict[str, list[str]] = {
    "Content Quality": [
        "the explanation is very clear and easy to understand",
        "great teaching, every concept is explained in depth",
        "the content is accurate and well structured",
        "the video contains wrong information",
        "samjhane ka tarika bahut accha hai",
        "bahut detail me samjhaya sir",
    ],
    "Engagement": [
        "very entertaining and engaging video",
        "loved your energy and the way you present",
        "boring and too long, I lost interest",
        "great delivery that kept me hooked till the end",
        "video bahut mazedaar tha",
        "keep it up, subscribed and liked",
    ],
    "Relevance": [
        "very useful for placements and getting a job",
        "exactly what I was looking for, so relevant",
        "this helped me in my interview and career",
        "up to date, practical real world knowledge",
        "interview ke liye bahut useful hai",
        "this is not relevant to the title of the video",
    ],
    "Feature Request": [
        "please make a video on this topic next",
        "can you make a full series or course on this",
        "we want more projects and examples, please upload",
        "next video kab aayega, waiting for it",
        "is topic par video banao sir please",
        "please cover this topic in your next video",
    ],
    "Pricing / Cost": [
        "the course is too expensive",
        "thank you for providing this content for free",
        "is it paid or free, what is the price",
        "totally worth the money",
        "free me itna accha content milta hai",
    ],
    "Technical Issue": [
        "the audio quality is bad, I cannot hear anything",
        "the video is lagging and buffering",
        "the code gives an error and is not working",
        "the screen is blurry and the font is too small",
        "awaaz nahi aa rahi sir",
    ],
    GENERAL: [
        "thank you sir, nice video",
        "good video",
        "love you sir, big fan",
        "first comment",
        "dhanyavad sir, bahut accha",
    ],
}

# Word-boundary keyword evidence (each hit adds KEYWORD_BOOST to that aspect's similarity)
KEYWORDS: dict[str, str] = {
    "Technical Issue": r"bug|glitch|error|crash|lag|laggy|buffering|audio|sound|mic|blurry|resolution|broken|not working|awaaz|awaz",
    "Pricing / Cost": r"price|cost|expensive|money|free|paid|cheap|rupees|rs|fees?|subscription|charge",
    "Feature Request": r"please make|make a video|make video|series|next video|upload|request|kab aayega|banao|bana do|full course|cover",
    "Content Quality": r"explain|explained|explanation|clear|concept|concepts|teach|teaching|understand|samjh\w*|samaj\w*|detail|wrong|correct",
    "Engagement": r"boring|entertaining|engaging|funny|fun|energy|interesting|mazedaar|mast|subscribed|subscribe",
    "Relevance": r"placement|placements|interview|job|jobs|career|useful|relevant|internship|offer|salary",
}
_KW = {a: re.compile(rf"\b(?:{p})\b", re.I) for a, p in KEYWORDS.items()}

ASPECTS = list(PROTOTYPES.keys())

_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+|\s+(?:but|however|although|though|lekin|magar|par)\s+", re.I)


def segment(comment: str, max_segments: int = 6) -> list[str]:
    """Splits a comment into clauses (sentences / contrast clauses). Always returns >= 1 segment."""
    text = re.sub(r"\s+", " ", str(comment)).strip()
    if not text:
        return [""]
    parts = [p.strip() for p in _SPLIT.split(text) if p and p.strip()]
    parts = [p[:300] for p in parts if len(p) >= 3] or [text[:300]]
    return parts[:max_segments]


class AspectMatcher:
    """Maps clauses to aspects using embedding similarity to prototypes + keyword evidence."""

    def __init__(self, embedder):
        self.embedder = embedder
        flat, owner = [], []
        for aspect, sentences in PROTOTYPES.items():
            for s in sentences:
                flat.append(s)
                owner.append(aspect)
        self.owner = np.array(owner)
        self.proto = embedder.encode(flat)
        self.aspects = ASPECTS

    def classify(self, clauses: list[str]) -> list[list[tuple[str, float]]]:
        """For every clause returns the aspect(s) it talks about as [(aspect, score), ...].

        A clause may express more than one aspect ("great explanation and so useful for placements"), so every
        aspect scoring above ASPECT_MIN_SIM and within MULTI_MARGIN of the best one is returned (at most MAX_ASPECTS).
        Pure praise ("thank you sir") and clauses without a clear aspect become General Feedback.
        """
        if not clauses:
            return []
        sims = self.embedder.encode(clauses) @ self.proto.T            # (n_clauses, n_prototypes)
        results: list[list[tuple[str, float]]] = []
        for i, clause in enumerate(clauses):
            score = {}
            for aspect in self.aspects:
                score[aspect] = float(sims[i, self.owner == aspect].max())
                if aspect in _KW and _KW[aspect].search(clause):
                    score[aspect] += KEYWORD_BOOST
            specific = {a: v for a, v in score.items() if a != GENERAL}
            best_aspect = max(specific, key=specific.get)
            best = specific[best_aspect]
            if best < ASPECT_MIN_SIM or score[GENERAL] > best + GENERAL_MARGIN:
                results.append([(GENERAL, max(score[GENERAL], best))])
                continue
            chosen = [(a, v) for a, v in sorted(specific.items(), key=lambda kv: -kv[1])
                      if v >= ASPECT_MIN_SIM and v >= best - MULTI_MARGIN][:MAX_ASPECTS]
            results.append(chosen)
        return results
