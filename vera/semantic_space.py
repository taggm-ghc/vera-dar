"""Item #91 (R1 decision 11, 2026-10-10): per-document semantic space, mixed (hybrid) config.

Pure numpy, no I/O, no DB, no network. Embedding vectors are supplied by the caller (no embedding source exists
in VERA-DAR as of 2026-10-10). Every threshold comes from config/semantic_space.json; the loader refuses
enabled=true while the config is not marked CALIBRATED (CLAUDE.md calibration rule).

Per document: fit_doc_space(chunk vectors) -> DocSpace | None (None = too few chunks, weight 0, status quo only).
Per question: grade_vectors(question vectors, space) -> inside / bordering / outside / undefined.
Mixed: hybrid_score(w, framework, status_quo) blends the framework grade with the status-quo score by the
document's measured support weight w (no hard cliff at the chunk floor).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "semantic_space.json"
GRADES = ("inside", "bordering", "outside", "undefined")
_RANK = {"inside": 0, "bordering": 1, "outside": 2, "undefined": 3}
_REQUIRED = ("enabled", "calibration_status", "min_chunks_per_doc", "default_chunk_chars", "min_chunk_chars",
             "max_components", "covariance_shrinkage", "fence_multiplier", "bootstrap_refits",
             "bootstrap_fraction", "bootstrap_seed", "stability_floor", "fusion", "rrf_k", "grade_scores",
             "question_aggregate")


class SemanticSpaceConfigError(ValueError):
    pass


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    try:
        cfg = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise SemanticSpaceConfigError(f"cannot read semantic-space config {path}: {e}") from e
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise SemanticSpaceConfigError(f"semantic-space config missing keys: {missing}")
    if cfg["fusion"] not in ("linear", "rrf"):
        raise SemanticSpaceConfigError("fusion must be 'linear' or 'rrf'")
    if cfg["question_aggregate"] not in ("best", "majority"):
        raise SemanticSpaceConfigError("question_aggregate must be 'best' or 'majority'")
    if not 0 <= cfg["covariance_shrinkage"] <= 1 or not 0 < cfg["bootstrap_fraction"] <= 1:
        raise SemanticSpaceConfigError("shrinkage in [0,1] and bootstrap_fraction in (0,1] required")
    if cfg["min_chunks_per_doc"] < 3 or cfg["max_components"] < 1:
        raise SemanticSpaceConfigError("min_chunks_per_doc >= 3 and max_components >= 1 required")
    if cfg["enabled"] and (cfg["calibration_status"] != "CALIBRATED" or not str(cfg.get("calibration_note", "")).strip()):
        raise SemanticSpaceConfigError(
            "enabled=true refused: calibration_status must be CALIBRATED with a calibration_note naming the real "
            "data and spread seen (CLAUDE.md calibration rule; M2 not done as of 2026-10-10)")
    return cfg


# ------------------------------------------------------------------ chunking (R1 sizing rule, plan 2a)
def chunk_text(text: str, cfg: dict) -> list[str] | None:
    """Chunk size = min(default, length / min_chunks), never below min_chunk_chars. None when
    length < min_chunks x min_chunk_chars (excluded from the framework; status quo only). Non-overlapping."""
    text = " ".join(text.split())
    n_min = cfg["min_chunks_per_doc"]
    if len(text) < n_min * cfg["min_chunk_chars"]:
        return None
    target = max(cfg["min_chunk_chars"], min(cfg["default_chunk_chars"], len(text) // n_min))
    words = text.split(" ")
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if sum(len(x) + 1 for x in cur) >= target:
            chunks.append(" ".join(cur)); cur = []
    if cur:
        if chunks and len(" ".join(cur)) < cfg["min_chunk_chars"]:
            chunks[-1] += " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    # prefer sentence-aligned boundaries when this still leaves the floor intact
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    packed, cur = [], ""
    for s in sents:
        cur = (cur + " " + s).strip()
        if len(cur) >= target:
            packed.append(cur); cur = ""
    if cur:
        if packed and len(cur) < cfg["min_chunk_chars"]:
            packed[-1] += " " + cur
        else:
            packed.append(cur)
    return packed if len(packed) >= n_min else (chunks if len(chunks) >= n_min else None)


# ------------------------------------------------------------------ document space
@dataclass(frozen=True)
class DocSpace:
    centre: np.ndarray
    basis: np.ndarray        # (k, d) orthonormal rows
    lam: np.ndarray          # (k,) shrunk eigenvalues
    n_chunks: int
    maha_fences: tuple       # (q3, upper_fence)
    resid_fences: tuple
    stability: float
    weight: float


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=float))
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(nrm == 0):
        raise ValueError("zero-length vector")
    return x / nrm


def _pca(X: np.ndarray, k: int, shrink: float):
    mu = X.mean(axis=0)
    Xc = X - mu
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    lam = s ** 2 / max(len(X) - 1, 1)
    k = min(k, len(X) - 2, int((s > 1e-12).sum()))
    if k < 1:
        return None
    lam_k = lam[:k]
    lam_k = (1 - shrink) * lam_k + shrink * lam[:max(k, 1)].mean()   # eigenvalue shrinkage toward the mean
    return mu, vt[:k], np.maximum(lam_k, 1e-12)


def _scores(x: np.ndarray, mu, basis, lam):
    d = np.atleast_2d(x) - mu
    coords = d @ basis.T
    maha = np.sqrt((coords ** 2 / lam).sum(axis=1))
    resid = np.linalg.norm(d - coords @ basis, axis=1)
    return maha, resid


def _fences(vals: np.ndarray, mult: float):
    q1, q3 = np.percentile(vals, [25, 75])
    return float(q3), float(q3 + mult * (q3 - q1))


def principal_angle_stability(b1: np.ndarray, b2: np.ndarray) -> float:
    """Mean squared cosine of the principal angles between two row-orthonormal bases (1 = same space)."""
    k = min(len(b1), len(b2))
    sv = np.linalg.svd(b1[:k] @ b2[:k].T, compute_uv=False)
    return float((sv ** 2).mean())


def fit_doc_space(chunk_vectors, cfg: dict) -> DocSpace | None:
    """None when the document has fewer than min_chunks_per_doc chunks or no fittable space (weight 0)."""
    X = _unit(chunk_vectors)
    n = len(X)
    if n < cfg["min_chunks_per_doc"]:
        return None
    fit = _pca(X, cfg["max_components"], cfg["covariance_shrinkage"])
    if fit is None:
        return None
    mu, basis, lam = fit
    # fences from leave-one-out scores of the document's own chunks (in-sample scores are biased small)
    maha, resid = [], []
    for i in range(n):
        f = _pca(np.delete(X, i, axis=0), cfg["max_components"], cfg["covariance_shrinkage"])
        if f is None:
            return None
        m, r = _scores(X[i], *f)
        maha.append(m[0]); resid.append(r[0])
    mf = _fences(np.array(maha), cfg["fence_multiplier"])
    rf = _fences(np.array(resid), cfg["fence_multiplier"])
    # stability by seeded subsample refits
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    size = max(int(np.ceil(cfg["bootstrap_fraction"] * n)), 3)
    angs = []
    for _ in range(cfg["bootstrap_refits"]):
        idx = rng.choice(n, size=min(size, n), replace=False)
        f = _pca(X[idx], cfg["max_components"], cfg["covariance_shrinkage"])
        if f is not None:
            angs.append(principal_angle_stability(basis, f[1]))
    stab = float(np.mean(angs)) if angs else 0.0
    floor = cfg["stability_floor"]
    w = 0.0 if stab <= floor else min(1.0, (stab - floor) / (1 - floor)) if floor < 1 else 0.0
    return DocSpace(mu, basis, lam, n, mf, rf, stab, w)


def grade_vectors(question_vectors, space: DocSpace | None, cfg: dict) -> dict:
    """Grade each question vector against the document's own fences; aggregate per config."""
    if space is None or space.weight <= 0:
        return {"grade": "undefined", "per_vector": [], "fraction_inside": 0.0}
    Q = _unit(question_vectors)
    maha, resid = _scores(Q, space.centre, space.basis, space.lam)
    per = []
    for m, r in zip(maha, resid):
        g = 0
        for v, (q3, up) in ((m, space.maha_fences), (r, space.resid_fences)):
            g = max(g, 0 if v <= q3 else 1 if v <= up else 2)
        per.append(GRADES[g])
    frac = per.count("inside") / len(per)
    if cfg["question_aggregate"] == "best":
        grade = min(per, key=_RANK.__getitem__)
    else:
        grade = "inside" if frac > 0.5 else min(per, key=_RANK.__getitem__) if "bordering" in per else "outside"
    return {"grade": grade, "per_vector": per, "fraction_inside": frac}


# ------------------------------------------------------------------ mixed (hybrid) scoring
def hybrid_score(weight: float, grade: str, status_quo_score: float, cfg: dict) -> float:
    """Linear fusion: w x framework + (1 - w) x status quo. status_quo_score must already be in [0, 1]."""
    if not 0 <= weight <= 1 or not 0 <= status_quo_score <= 1:
        raise ValueError("weight and status_quo_score must lie in [0, 1]")
    return weight * cfg["grade_scores"][grade] + (1 - weight) * status_quo_score


def rrf_fuse(framework_ranked: list, status_quo_ranked: list, weights: dict, cfg: dict) -> list:
    """Weighted reciprocal-rank fusion of two doc-id rankings; weights[doc] = w (missing = 0)."""
    k = cfg["rrf_k"]
    fr = {d: i for i, d in enumerate(framework_ranked, 1)}
    sq = {d: i for i, d in enumerate(status_quo_ranked, 1)}
    out = {}
    for d in set(fr) | set(sq):
        w = weights.get(d, 0.0)
        out[d] = (w / (k + fr[d]) if d in fr else 0.0) + ((1 - w) / (k + sq[d]) if d in sq else 0.0)
    return sorted(out, key=lambda d: (-out[d], str(d)))
