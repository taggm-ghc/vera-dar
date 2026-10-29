"""SEMSIM-S4 (OD-1028(48)): one shared interface for meaning-based similarity.

Two uses, calibrated separately: topic matching ("topic") and category comparison ("category").
Outputs are only "similar", "dissimilar" or "unknown". Similarity is an input to callers; it never
establishes prohibited intent and never triggers a refusal (guarded by tests/test_semantic_similarity.py).

Fail closed: model unavailable, threshold null/uncalibrated, bad input or any error -> "unknown".
Text is data only. Nothing is downloaded at import or request time; fastembed is imported lazily and
loads a model only from a local directory named by the env var in the config.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Protocol, Sequence

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "semantic_similarity.json"
USES = ("topic", "category")
_THRESHOLD_KEY = {"topic": "threshold_topic", "category": "threshold_category"}
SIMILAR, DISSIMILAR, UNKNOWN = "similar", "dissimilar", "unknown"
DEFAULT_MAX_CHARS = 2000


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedAdapter:
    """Lazy fastembed wrapper. Constructing it imports nothing; the first embed() does."""

    def __init__(self, model_name: str, model_dir: str | None):
        self.model_name, self.model_dir, self._model = model_name, model_dir, None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            if not self.model_dir:
                raise RuntimeError("model directory not configured")
            from fastembed import TextEmbedding  # lazy: optional dependency
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.model_dir,
                                        local_files_only=True)
        return [[float(x) for x in v] for v in self._model.embed(list(texts))]


class FakeEmbedder:
    """Deterministic bag-of-hashed-tokens embedder for tests (no model, no network)."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import zlib
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in str(t).lower().split():
                v[zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
            out.append(v)
        return out


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def cosine(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0 or not math.isfinite(na * nb):
        return None
    s = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return max(-1.0, min(1.0, s)) if math.isfinite(s) else None


def _clean(text, cap: int) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    return text[:cap]


def similarity_score(text_a: str, text_b: str, embedder: Embedder, max_chars: int = DEFAULT_MAX_CHARS) -> float | None:
    """Cosine of the two embeddings, or None on any failure (fail closed)."""
    try:
        a, b = _clean(text_a, max_chars), _clean(text_b, max_chars)
        if a is None or b is None:
            return None
        va, vb = embedder.embed([a, b])
        return cosine(va, vb)
    except Exception:
        return None


def default_embedder(cfg: dict | None = None, environ=os.environ) -> Embedder | None:
    """FastEmbed adapter from config/env; None if disabled or no local model dir is set."""
    try:
        cfg = cfg if cfg is not None else load_config()
        model_dir = environ.get(cfg.get("model_cache_env") or "")
        if not cfg.get("enabled") or not model_dir:
            return None
        return FastEmbedAdapter(cfg["model_name"], model_dir)
    except Exception:
        return None


def similarity_decision(use: str, text_a: str, text_b: str, embedder: Embedder | None = None,
                        cfg: dict | None = None) -> str:
    """'similar' | 'dissimilar' | 'unknown' using the threshold for `use` ('topic' or 'category')."""
    try:
        if use not in USES:
            return UNKNOWN
        cfg = cfg if cfg is not None else load_config()
        thr = cfg.get(_THRESHOLD_KEY[use])
        if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not math.isfinite(thr):
            return UNKNOWN  # null/uncalibrated
        emb = embedder if embedder is not None else default_embedder(cfg)
        if emb is None:
            return UNKNOWN
        s = similarity_score(text_a, text_b, emb, int(cfg.get("max_input_chars") or DEFAULT_MAX_CHARS))
        if s is None:
            return UNKNOWN
        return SIMILAR if s >= thr else DISSIMILAR
    except Exception:
        return UNKNOWN
