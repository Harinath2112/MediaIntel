"""Lightweight stand-ins for the RoBERTa / embedding models so the pipeline logic can be tested
offline (no downloads, no GPU).  Real models are used in production."""
import hashlib, re, time
import numpy as np

POS = {"great","good","best","love","amazing","excellent","thanks","thank","awesome","nice","helpful","useful","clear","mast","bahut","accha","worth","free"}
NEG = {"bad","terrible","worst","boring","wrong","lag","laggy","error","expensive","poor","blurry","hate","useless","not"}

class FakeSentiment:
    name = "fake-roberta"
    def predict_proba(self, texts):
        out = []
        for t in texts:
            w = set(re.findall(r"\w+", t.lower()))
            p, n = len(w & POS), len(w & NEG)
            v = np.array([0.1 + 0.5*n, 0.4, 0.1 + 0.5*p], dtype=np.float32)
            out.append(v / v.sum())
        return np.array(out)

class FakeEmbedder:
    """Hashed bag-of-words vectors: texts sharing words get a high cosine similarity."""
    DIM = 256
    def _vec(self, text):
        v = np.zeros(self.DIM, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            rng = np.random.default_rng(h % (2**32))
            v += rng.standard_normal(self.DIM).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v
    def encode(self, texts, batch_size=64):
        return np.array([self._vec(t) for t in texts], dtype=np.float32) if texts else np.zeros((0, self.DIM), np.float32)


# ---------------- fake YouTube / Whisper for offline end-to-end runs (MEDIAINTEL_TEST_FAKES=1) -------------
import random
from datetime import datetime, timedelta, timezone


class FakeYouTube:
    TOPICS = {"python": "python lists loops functions dictionaries strings", "dsa": "arrays trees recursion sorting graphs",
              "web": "html css javascript layout flexbox forms"}
    REQ = ["please make a video on system design", "machine learning series kab aayega sir please", "please upload docker tutorial",
           "waiting for react js course please make", "make a video on machine learning from scratch", "system design playlist chahiye"]
    GEN = ["thanks sir great video", "nice video sir", "very good explanation thank you"]

    def uploads_playlist_id(self, cid): return "UU" + cid
    def recent_video_ids(self, pl, n): return [f"{pl[2:]}_new{i}" for i in range(3)]

    def video_details(self, ids):
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        out = []
        for i, v in enumerate(ids):
            out.append(dict(video_id=v, title=f"Fake upload {v}", description="free course", duration_sec=600,
                            published_at=now - timedelta(hours=30 + 20 * i), view_count=50000 * (i + 1), like_count=2000 * (i + 1),
                            comment_count=40))
        return out

    def top_comments(self, vid, pages=1):
        rnd = random.Random(vid)
        words = " ".join(self.TOPICS.values()).split()
        rows = []
        for j in range(40):
            r = rnd.random()
            txt = " ".join(rnd.sample(words, 7)) if r < .4 else rnd.choice(self.GEN) if r < .65 else rnd.choice(self.REQ)
            rows.append(dict(comment_id=f"{vid}_c{j}", video_id=vid, author="u", text=txt, likes=rnd.choice([0, 0, 2, 9]),
                             published_at=datetime.now(timezone.utc).replace(tzinfo=None)))
        return rows


def fake_fetch_audio(video_id, max_seconds):
    """Stand-in for the in-memory audio stream: a tiny array that carries the video id's hash."""
    import numpy as np
    return np.full(16000, (hash(video_id) % 20000) + 1, dtype=np.int16)


def fake_whisper(pcm, max_seconds):
    time.sleep(0.05)
    words = "python arrays trees recursion html css javascript functions loops sorting".split()
    rnd = random.Random(int(pcm[0]))
    return " ".join(rnd.choice(words) for _ in range(250)), "en", 300


def fake_captions(video_id):
    return f"caption text for {video_id} about python and arrays", "en"
