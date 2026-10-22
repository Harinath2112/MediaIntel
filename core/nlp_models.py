"""Model loaders shared by the NLP, relevance and topic stages.

  * SentimentModel : RoBERTa (XLM-RoBERTa, multilingual) 3-class sentiment  -> Negative / Neutral / Positive
  * Embedder       : multilingual sentence-transformer used for aspect matching, relevance and topic merging

Both load lazily (first call) and are cached for the life of the process.
Speed on CPU: texts are sorted by length so batches have minimal padding, inference runs with
torch.inference_mode, weights are dynamically quantised to int8 (set QUANTIZE=0 to disable).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402

SENTIMENT_MODEL = config.env("SENTIMENT_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
EMBEDDING_MODEL = config.env("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
SENTIMENT_BATCH = config.env_int("SENTIMENT_BATCH", 32)
MAX_TOKENS = config.setting_int("SENTIMENT_MAX_TOKENS")
QUANTIZE = config.env_bool("QUANTIZE", True)

CANONICAL = ["Negative", "Neutral", "Positive"]


def clean_for_model(text: str, limit: int = 500) -> str:
    """Normalisation the cardiffnlp Twitter-RoBERTa family was trained with."""
    t = re.sub(r"@\w+", "@user", str(text))
    t = re.sub(r"https?://\S+", "http", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit]


class SentimentModel:
    """RoBERTa sentiment classifier. predict_proba() -> array (n, 3) ordered Negative, Neutral, Positive."""

    def __init__(self, name: str = SENTIMENT_MODEL):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "ERROR: the RoBERTa sentiment model needs PyTorch + transformers.\n"
                "       Run:  pip install -r requirements.txt") from exc
        self.torch = torch
        self.name = name
        print(f"Loading RoBERTa sentiment model '{name}' (first run downloads it once)...")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            torch.set_num_threads(max(1, __import__("os").cpu_count() or 1))
            if QUANTIZE:
                try:
                    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                except Exception as exc:  # noqa: BLE001 - quantisation is only an optimisation
                    print(f"  (int8 quantisation skipped: {exc})")
        else:
            model = model.to(self.device)
        self.model = model

        # map the model's own label order onto Negative / Neutral / Positive
        id2label = getattr(self.model.config, "id2label", {}) or {}
        order = []
        for i in range(len(id2label) or 3):
            lab = str(id2label.get(i, f"LABEL_{i}")).lower()
            if "neg" in lab:
                order.append(0)
            elif "neu" in lab:
                order.append(1)
            elif "pos" in lab:
                order.append(2)
            else:
                order.append(i)  # LABEL_0/1/2 -> cardiffnlp convention (neg, neu, pos)
        self.perm = np.argsort(order)  # column j of the output = model column perm[j]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        cleaned = [clean_for_model(t) for t in texts]
        out = np.zeros((len(cleaned), 3), dtype=np.float32)
        idx = sorted(range(len(cleaned)), key=lambda i: len(cleaned[i]))  # similar lengths -> little padding
        with torch.inference_mode():
            for s in range(0, len(idx), SENTIMENT_BATCH):
                batch_ids = idx[s: s + SENTIMENT_BATCH]
                enc = self.tokenizer([cleaned[i] for i in batch_ids], padding=True, truncation=True,
                                     max_length=MAX_TOKENS, return_tensors="pt")
                if self.device != "cpu":
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
                out[batch_ids] = probs[:, self.perm]
        return out


class Embedder:
    """Multilingual sentence embeddings (unit-length vectors, so dot product = cosine similarity)."""

    def __init__(self, name: str = EMBEDDING_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("ERROR: sentence-transformers is missing. Run: pip install -r requirements.txt") from exc
        print(f"Loading embedding model '{name}' (first run downloads it once)...")
        self.name = name
        self.model = SentenceTransformer(name)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vecs = self.model.encode([str(t)[:600] for t in texts], batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


_cache: dict = {}


def _fakes():
    """Offline self-test hook (MEDIAINTEL_TEST_FAKES=1): lightweight stand-ins, never used in normal runs."""
    sys.path.insert(0, str(config.ROOT / "tests"))
    import fakes
    return fakes


def get_sentiment_model():
    if "sentiment" not in _cache:
        _cache["sentiment"] = _fakes().FakeSentiment() if config.env_bool("MEDIAINTEL_TEST_FAKES") else SentimentModel()
    return _cache["sentiment"]


def get_embedder():
    if "embedder" not in _cache:
        _cache["embedder"] = _fakes().FakeEmbedder() if config.env_bool("MEDIAINTEL_TEST_FAKES") else Embedder()
    return _cache["embedder"]
