"""SEMSIM-S4 (OD-1028(48)): one shared interface for meaning-based similarity.

Two uses, calibrated separately: topic matching ("topic") and category comparison ("category").
Outputs are only "similar", "dissimilar" or "unknown". Similarity is an input to callers; it never
establishes prohibited intent and never triggers a refusal (guarded by tests/test_semantic_similarity.py).

OD-1028(26)/(48): similarity must never by itself establish prohibited intent or trigger refusal. Only
similarity_decision / similarity_decision_with_mode and the Embedder/adapter types are public; scoring helpers are
private. Structural and behavioural guards live in tests/test_semantic_similarity.py.

Shadow mode: when cfg["shadow"] is not False the decision is computed but flagged non-actionable; callers
must NOT act on it (log only). similarity_decision() then returns "unknown".

Operational: the first model load is slow and must NOT happen inside an advisory-lock DB transaction; call
warm() at startup or outside the lock. The adapter is cached per (model_dir, sha256) behind a lock.

Fail closed: model unavailable, threshold null/uncalibrated, bad input or any error -> "unknown".
Text is data only. Nothing is downloaded at import or request time; fastembed is imported lazily and
loads a model only from a local directory named by the env var in the config.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import NamedTuple, Protocol, Sequence

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "semantic_similarity.json"
USES = ("topic", "category")
_THRESHOLD_KEY = {"topic": "threshold_topic", "category": "threshold_category"}
SIMILAR, DISSIMILAR, UNKNOWN = "similar", "dissimilar", "unknown"
DEFAULT_MAX_CHARS = 2000
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class SimilarityResult(NamedTuple):
    decision: str
    mode: str  # "active" | "shadow" | "off"
    actionable: bool  # False in shadow/off: callers must not act on the decision (log only)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _real_loader(model_name: str, model_dir: str):  # UNVERIFIED against fastembed 0.9.0 (not installed in test venv)
    from fastembed import TextEmbedding  # lazy: optional dependency
    return TextEmbedding(model_name=model_name, specific_model_path=model_dir)


class FastEmbedAdapter:
    """Lazy fastembed wrapper. Constructing it imports nothing; the first embed()/load() does the slow load.

    Before loading: the sha256 of `model_file` inside `model_dir` must equal `model_sha256`. The model is loaded
    from `model_dir` itself (specific_model_path) with HF_HUB_OFFLINE=1 set only for the duration of the load."""

    def __init__(self, model_name: str, model_dir: str | None, model_sha256: str = "", model_file: str = "",
                 loader=None):
        self.model_name, self.model_dir = model_name, model_dir
        self.model_sha256, self.model_file = model_sha256, model_file
        self._loader, self._model, self._lock = loader or _real_loader, None, threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not self.model_dir or not _SHA_RE.match(self.model_sha256 or "") or not self.model_file \
                    or os.sep in self.model_file or self.model_file in (".", "..") or "/" in self.model_file:
                raise RuntimeError("model directory/hash/file not configured")
            fp = Path(self.model_dir) / self.model_file
            if not fp.is_file() or _file_sha256(fp) != self.model_sha256:
                raise RuntimeError("model file hash mismatch")
            prev = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                self._model = self._loader(self.model_name, self.model_dir)
            finally:
                if prev is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            self.load()
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


def _load_config(path: Path | str = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def _cosine(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0 or not math.isfinite(na * nb):
        return None
    s = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return max(-1.0, min(1.0, s)) if math.isfinite(s) else None


def _cap(cfg: dict) -> int | None:
    cap = cfg.get("max_input_chars", DEFAULT_MAX_CHARS)
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= DEFAULT_MAX_CHARS:
        return None
    return cap


def _clean(text, cap: int) -> str | None:
    if not isinstance(text, str):
        return None
    text = text[:cap]
    return text if text.strip() else None  # re-check emptiness AFTER slicing


def _similarity_score(text_a: str, text_b: str, embedder: Embedder, cap: int) -> float | None:
    """Cosine of the two embeddings, or None on any failure (fail closed). Private: callers get decisions only."""
    try:
        a, b = _clean(text_a, cap), _clean(text_b, cap)
        if a is None or b is None:
            return None
        va, vb = embedder.embed([a, b])
        return _cosine(va, vb)
    except Exception:
        return None


_ADAPTERS: dict = {}
_ADAPTERS_LOCK = threading.Lock()


def _valid_model_cfg(cfg: dict) -> bool:
    rev, sha = cfg.get("model_revision"), cfg.get("model_sha256")
    if not isinstance(rev, str) or not rev.strip() or "PLACEHOLDER" in rev.upper():
        return False
    name = cfg.get("model_name")
    if not isinstance(name, str) or not name.strip() or "PLACEHOLDER" in name.upper():
        return False
    return isinstance(sha, str) and bool(_SHA_RE.match(sha))


def _adapter(cfg: dict, environ=os.environ, loader=None) -> FastEmbedAdapter | None:
    """Cached adapter per (model_dir, sha256); None if disabled, placeholder values, or no model dir."""
    if cfg.get("enabled") is not True or not _valid_model_cfg(cfg):
        return None
    model_dir = environ.get(cfg.get("model_cache_env") or "")
    if not model_dir:
        return None
    key = (model_dir, cfg["model_sha256"])
    with _ADAPTERS_LOCK:
        ad = _ADAPTERS.get(key)
        if ad is None:
            ad = _ADAPTERS[key] = FastEmbedAdapter(cfg["model_name"], model_dir, cfg["model_sha256"],
                                                   str(cfg.get("model_file") or ""), loader)
        return ad


def warm(cfg: dict | None = None, environ=os.environ) -> bool:
    """Load the model now. Call at startup or OUTSIDE any advisory-lock transaction. True if loaded."""
    try:
        cfg = cfg if cfg is not None else _load_config()
        ad = _adapter(cfg, environ)
        if ad is None:
            return False
        ad.load()
        return True
    except Exception:
        return False


def similarity_decision_with_mode(use: str, text_a: str, text_b: str, embedder: Embedder | None = None,
                                  cfg: dict | None = None, environ=os.environ) -> SimilarityResult:
    """Decision plus mode. Config is read once. `enabled` must be the literal True (else unknown, even with an
    explicit embedder). shadow must be literally False to be actionable; otherwise log only (OD-1028(26)/(48))."""
    try:
        cfg = cfg if cfg is not None else _load_config()
        if not isinstance(cfg, dict) or cfg.get("enabled") is not True:
            return SimilarityResult(UNKNOWN, "off", False)
        mode = "active" if cfg.get("shadow") is False else "shadow"
        res = lambda d: SimilarityResult(d, mode, mode == "active" and d != UNKNOWN)  # noqa: E731
        if use not in USES:
            return res(UNKNOWN)
        thr = cfg.get(_THRESHOLD_KEY[use])
        if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not math.isfinite(thr) or not -1 <= thr <= 1:
            return res(UNKNOWN)  # null/uncalibrated/out of range
        cap = _cap(cfg)
        if cap is None:
            return res(UNKNOWN)
        emb = embedder if embedder is not None else _adapter(cfg, environ)
        if emb is None:
            return res(UNKNOWN)
        s = _similarity_score(text_a, text_b, emb, cap)
        if s is None:
            return res(UNKNOWN)
        return res(SIMILAR if s >= thr else DISSIMILAR)
    except Exception:
        return SimilarityResult(UNKNOWN, "off", False)


def similarity_decision(use: str, text_a: str, text_b: str, embedder: Embedder | None = None,
                        cfg: dict | None = None) -> str:
    """'similar' | 'dissimilar' | 'unknown'. Returns a non-unknown value only in active (non-shadow) mode."""
    r = similarity_decision_with_mode(use, text_a, text_b, embedder, cfg)
    return r.decision if r.mode == "active" else UNKNOWN
