"""CHUNK-TAXO-CORE-1109 (C1): pure, deterministic engine that derives a subtopic -> topic -> domain hierarchy of
semantic centers from chunk vectors and associates chunks to it.  Plan: work/CHUNK-TAXO-1109/status-plan.md, v3.0.

Contract: no LLM, no keyword or stop lists, no network, no DB, no file access except load_cfg().  Same inputs ->
byte-identical outputs (Model.dumps()).  Every threshold comes from the config `params`; the defaults in DEFAULT_PARAMS
apply only when a key is missing from the file.

Method
  1. Vectors are L2-normalised (cosine).  Chunks are processed in span_id order, so input order does not matter.
  2. Subtopics: seeded k-means++ / Lloyd, best of one run per configured seed (highest total cosine).  k is chosen on a bounded ascending grid by mean silhouette.  Stopping rule:
     stop when `patience` consecutive k add less than `gain_epsilon` silhouette over the best so far, or when a k yields
     a cluster smaller than `min_cluster_size` (that k is rejected).  No valid k -> one subtopic.
  3. Topics: average-linkage (UPGMA, cosine distance) agglomeration over subtopic centers, cut at the largest gap
     between consecutive merge heights among cluster counts in a bounded range.  Domains: same over topic centers.
     A cut always has fewer clusters than the level below (unless that level has one node), so each level is coarser.
     Parents are the clusters that contain the child, so containment holds by construction.
  4. Nulls (per level, `null_seeds`): the level's real labels are shuffled over the chunks (same cluster sizes), group
     centers are re-derived, and each chunk's nearest-group distance and top-two margin are measured against the groups
     that do not contain it.  `dist_q` is the `null_q_dist` quantile (lower tail) of the pooled null distances; a chunk
     fits only if its distance <= dist_q.  `margin_q` is the `null_q_margin` quantile of the pooled null margins; a chunk
     whose margin is below it is ambiguous.  The margin test needs >= 3 nodes at the level (otherwise margin_q is None).
  5. Association tries subtopic, then topic, then domain.  Reasons (the failure at the first level tried): NO_FIT
     (distance above dist_q), LOW_SUPPORT (node has < min_support members), SPLIT (margin below margin_q).  A chunk
     that fails the subtopic level backs off to the parent level (topic assigned from the nearest TOPIC center; subtopic
     None); if every level fails it abstains (all keys None, level None).
  6. Terms: c-TF-IDF per node over regex tokens; terms in more than `term_max_df` of all chunks are dropped (a data
     statistic, not a list).  Labels come from the top terms.

Backends: numpy if importable (3 primitives: _prep, _sims, _msum), else pure Python.  Sims are rounded to 10 decimals
and outputs to 9, so both paths give the same keys, labels, assignments and k; centers/distances agree to <= 1e-8
(tests assert 1e-7).  `set_backend(False)` forces the fallback.  Memory for 6000 x 384: numpy path holds one float64
matrix (18 MB) plus blocked sim tiles (<= 1024 x k x 8 B) and tolist() copies of n x k (k <= 40: ~8 MB); the pure-Python
path holds boxed floats (about 74 MB) and is roughly 50-100x slower (minutes, not seconds, for a full run).
numpy is NOT in repo requirements.txt today; whether to add it is an owner decision (plan v3.2).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

try:  # optional
    import numpy as _np
except Exception:  # pragma: no cover - depends on environment
    _np = None

_USE_NP = _np is not None
LEVELS = ("subtopic", "topic", "domain")
_PREFIX = {"subtopic": "st", "topic": "tp", "domain": "dm"}
_SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_TOKEN = re.compile(r"[a-z][a-z0-9]{2,}")
REASONS = ("NO_FIT", "SPLIT", "LOW_SUPPORT")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "chunk_taxo.json"

DEFAULT_PARAMS = {
    "seeds": [11, 23, 37], "k_min": 2, "k_max": 40, "k_step": 1, "gain_epsilon": 0.005, "patience": 2,
    "min_cluster_size": 3, "kmeans_max_iter": 50,
    "topic_range": {"min_count": 2, "max_count": 20, "max_frac": 0.5},
    "domain_range": {"min_count": 2, "max_count": 8, "max_frac": 0.5},
    "null_seeds": [101, 102, 103], "null_q_dist": 0.05, "null_q_margin": 0.25, "min_support": 3,
    "identity_jaccard": 0.5, "term_count": 8, "label_terms": 3, "term_max_df": 0.5, "term_min_len": 3,
    "vote_min_share": 0.4, "vote_min_lead": 0.1, "stability_match_cos": 0.9,
    # TAXO-ROUTE-R1-1110 generous multi-membership (routing links; display metrics stay strict). rel_margin / abs_floor None =
    # derived per level: rel_margin_mult x margin_q (0.05 if margin_q is null), 1 - abs_floor_mult x dist_q. PENDING CAL.
    "multi": {"enabled": True, "rel_margin": None, "rel_margin_mult": 2.0, "abs_floor": None, "abs_floor_mult": 1.25,
              "max_links": {"subtopic": 4, "topic": 3, "domain": 3}, "soft_outliers": True}}
DEFAULT_GATE = {
    "min_silhouette_subtopic": 0.1, "min_silhouette_topic": 0.05, "min_silhouette_domain": 0.0,
    "min_stability_ari": 0.6, "min_center_match_rate": 0.7, "min_coverage_subtopic": 0.5, "min_coverage_domain": 0.8,
    "max_null_floor_spread": 0.1, "max_abstain_share": 0.3}


def set_backend(use_numpy):
    """Force numpy (True, if importable), the pure-Python fallback (False), or auto (None). Returns the active flag."""
    global _USE_NP
    _USE_NP = (_np is not None) if use_numpy is None else bool(use_numpy and _np is not None)
    return _USE_NP


def has_numpy():
    return _np is not None


# ----------------------------------------------------------------------------- config
def _chk(cond, msg):
    if not cond:
        raise ValueError("chunk_taxo config: " + msg)


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _merge(default, given, where):
    out = copy.deepcopy(default)
    _chk(isinstance(given, dict), where + " must be an object")
    for k, v in given.items():
        _chk(k in default, "unknown key %s.%s" % (where, k))
        if isinstance(default[k], dict):
            out[k] = _merge(default[k], v, where + "." + k)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _validate_params(p):
    for k in ("seeds", "null_seeds"):
        _chk(isinstance(p[k], list) and p[k] and all(isinstance(s, int) and not isinstance(s, bool) for s in p[k]),
             k + " must be a non-empty int list")
    for k in ("k_min", "k_max", "k_step", "patience", "min_cluster_size", "kmeans_max_iter", "min_support",
              "term_count", "label_terms", "term_min_len"):
        _chk(isinstance(p[k], int) and not isinstance(p[k], bool) and p[k] >= 1, k + " must be an int >= 1")
    _chk(p["k_min"] >= 2 and p["k_max"] >= p["k_min"], "need 2 <= k_min <= k_max")
    _chk(p["label_terms"] <= p["term_count"], "label_terms must be <= term_count")
    for k in ("gain_epsilon", "null_q_dist", "null_q_margin", "identity_jaccard", "term_max_df", "vote_min_share",
              "vote_min_lead", "stability_match_cos"):
        _chk(_is_num(p[k]), k + " must be a finite number")
    for k in ("null_q_dist", "null_q_margin", "identity_jaccard", "term_max_df", "vote_min_share", "vote_min_lead"):
        _chk(0 <= p[k] <= 1, k + " must be in [0,1]")
    _chk(-1 <= p["stability_match_cos"] <= 1 and p["gain_epsilon"] >= 0, "stability_match_cos/gain_epsilon range")
    m = p["multi"]
    _chk(isinstance(m["enabled"], bool) and isinstance(m["soft_outliers"], bool), "multi.enabled/soft_outliers must be bool")
    for k in ("rel_margin", "abs_floor"):
        _chk(m[k] is None or (_is_num(m[k]) and -1 <= m[k] <= 1), "multi.%s must be null or in [-1,1]" % k)
    for k in ("rel_margin_mult", "abs_floor_mult"):
        _chk(_is_num(m[k]) and 0 <= m[k] <= 10, "multi.%s must be in [0,10]" % k)
    _chk(isinstance(m["max_links"], dict) and set(m["max_links"]) == set(LEVELS) and all(
        isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 10 for v in m["max_links"].values()),
        "multi.max_links needs int 1..10 for each level")
    for k in ("topic_range", "domain_range"):
        r = p[k]
        _chk(all(isinstance(r[x], int) and r[x] >= 1 for x in ("min_count", "max_count")) and _is_num(r["max_frac"])
             and 0 < r["max_frac"] <= 1 and r["max_count"] >= r["min_count"], k + " invalid")


# Job-side sections (CHUNK-TAXO-INT-1109): validated here, consumed by vera/chunk_taxo_job.py and chunk_taxo_embed.py.
_CF_NUM = {"daily_neuron_budget": (0, 10**9), "chars_per_token": (0.5, 100), "backoff_base_s": (0, 600), "max_wait_s": (0, 3600),
           "timeout_s": (1, 600), "reserve_neurons": (0, 10**9)}
_CF_INT = {"max_attempts": (1, 10), "max_batch": (1, 1000)}
_JOB_NUM = {"interval_s": (1, 10**7), "initial_delay_s": (0, 10**7), "step_delay_s": (0, 3600), "page_size": (1, 5000),
            "tick_row_cap": (1, 100000), "statement_timeout_ms": (100, 600000), "lock_timeout_ms": (100, 600000),
            "connect_timeout_s": (1, 120), "text_chars": (50, 20000), "embed_batch": (1, 512), "max_rows": (1, 10**7),
            "max_seconds": (1, 86400), "rss_soft_mb": (1, 10**6), "rss_hard_mb": (1, 10**6),
            "embed_batch_chars": (1000, 10**6), "quarantine_max": (1, 10000), "quarantine_max_consecutive": (1, 1000),  # TAXO-ROBUST-1111
            "wake_min_interval_s": (1, 10**7)}  # WP-I D5: a safety bound on early passes, default = interval_s


def _rng(d, k, v, lo, hi, where, integer=False):
    _chk(_is_num(v) and lo <= v <= hi and (not integer or isinstance(v, int)), "%s.%s out of range" % (where, k))


def _validate_embedder(v):
    _chk(isinstance(v, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", v) is not None, "embedder must be a short lowercase name")
    return v


def _validate_cloudflare(c):
    _chk(isinstance(c, dict), "cloudflare must be an object")
    for k, v in c.items():
        if k == "model":
            _chk(isinstance(v, str) and re.fullmatch(r"(@cf/baai/)?[a-z0-9][a-z0-9._-]{0,63}", v) is not None, "cloudflare.model invalid")
        elif k in ("send_allowed", "truncate_inputs"):
            _chk(isinstance(v, bool), "cloudflare.%s must be a bool" % k)
        elif k in _CF_NUM:
            _rng(c, k, v, *_CF_NUM[k], "cloudflare")
        elif k in _CF_INT:
            _rng(c, k, v, *_CF_INT[k], "cloudflare", integer=True)
        elif k == "models":
            _chk(isinstance(v, dict) and all(isinstance(m, dict) and all(
                kk in ("max_tokens", "neurons_per_m", "dim") and (x is None or _is_num(x)) for kk, x in m.items())
                for m in v.values()), "cloudflare.models invalid")
        else:
            _chk(False, "unknown key cloudflare." + str(k))
    return copy.deepcopy(c)


def _validate_job(j):
    _chk(isinstance(j, dict), "job must be an object")
    for k, v in j.items():
        if k in ("allow_curator_origin_on_020", "malloc_trim", "finish_now"):
            _chk(isinstance(v, bool), "job.%s must be a bool" % k)
        elif k == "incremental":  # WP-I: D3 mode, ships "off"
            _chk(v in ("off", "shadow", "on"), "job.incremental must be off, shadow or on")
        elif k in ("rebuild_unassigned_share", "rebuild_interval_s"):  # WP-I: rebuild triggers; null = unset (no value ships: calibration note)
            _chk(v is None or (_is_num(v) and 0 < v <= 10**9), "job.%s must be null or a positive number" % k)
        else:
            _chk(k in _JOB_NUM, "unknown key job." + str(k))
            _rng(j, k, v, *_JOB_NUM[k], "job")
    return copy.deepcopy(j)


def load_cfg(path=None):
    """Load and validate config/chunk_taxo.json (or `path`). Returns {"enabled": bool, "params": {...}, "display_gate":
    {...}, "embedder": str ("none" if absent), "cloudflare": {...}, "job": {...}} with defaults filled in for missing keys. Raises ValueError on a bad file/value. `enabled` defaults False."""
    p = Path(path) if path else CONFIG_PATH
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError("chunk_taxo config unreadable: %s" % type(e).__name__)
    _chk(isinstance(raw, dict), "top level must be an object")
    for k in raw:
        _chk(k in ("enabled", "params", "display_gate", "embedder", "cloudflare", "job", "_note"), "unknown top-level key " + str(k))
    enabled = raw.get("enabled", False)
    _chk(isinstance(enabled, bool), "enabled must be a bool")
    params = _merge(DEFAULT_PARAMS, raw.get("params", {}), "params")
    _validate_params(params)
    gate = copy.deepcopy(DEFAULT_GATE)
    g = raw.get("display_gate", {})
    _chk(isinstance(g, dict), "display_gate must be an object")
    for k, v in g.items():
        _chk(k in DEFAULT_GATE, "unknown display_gate key " + str(k))
        _chk(v is None or _is_num(v), "display_gate.%s must be a number or null" % k)
        gate[k] = v
    return {"enabled": enabled, "params": params, "display_gate": gate,
            "embedder": _validate_embedder(raw.get("embedder", "none")),
            "cloudflare": _validate_cloudflare(raw.get("cloudflare", {})),
            "job": _validate_job(raw.get("job", {}))}


# ----------------------------------------------------------------------------- backend primitives
def _r(x, nd=9):
    return round(float(x), nd)


def _prep(vecs, owned=False):
    """Validate and L2-normalise. Returns an opaque matrix handle (float32 ndarray, or list of lists) and its dimension.
    numpy backend: ONE float32 copy (none when `owned` and the input is already float32, which is then normalised in place);
    the norms come from einsum, so there is no squared-values temporary and no float64 copy."""
    if _np is not None and (isinstance(vecs, _np.ndarray) or (_USE_NP and len(vecs) > 0)):
        if not isinstance(vecs, _np.ndarray):
            try:
                vecs, owned = _np.asarray(vecs, dtype=_np.float32), True
            except (ValueError, TypeError) as e:
                raise ValueError("ragged or non-numeric vectors") from e
        if vecs.ndim != 2 or vecs.shape[0] == 0 or vecs.shape[1] == 0:
            raise ValueError("vectors must be a non-empty 2-D array")
        A = vecs.astype(_np.float32, copy=not owned)
        nr = _np.sqrt(_np.einsum("ij,ij->i", A, A))
        if not _np.isfinite(nr).all():
            raise ValueError("non-finite vector component")
        if (nr == 0).any():
            raise ValueError("zero vector")
        A /= nr[:, None]
        return A if _USE_NP else A.tolist()
    n = len(vecs)
    if n == 0:
        raise ValueError("no vectors")
    d = len(vecs[0])
    if d == 0:
        raise ValueError("empty vector")
    rows = []
    for v in vecs:
        if len(v) != d:
            raise ValueError("ragged vectors")
        vv = [float(x) for x in v]
        if not all(math.isfinite(x) for x in vv):
            raise ValueError("non-finite vector component")
        nr = math.sqrt(sum(x * x for x in vv))
        if nr == 0:
            raise ValueError("zero vector")
        rows.append([x / nr for x in vv])
    return rows


def _nrows(X):
    return len(X)


def _row(X, i):
    return [float(x) for x in X[i]]


_BLOCK = 512  # rows per batch in every n x k computation (numpy backend)


def _ca(C):
    """Centers as a float32 (d x k) matrix for X-block @ Ca."""
    return _np.ascontiguousarray(_np.array(C, dtype=_np.float32).T)


def _sim_block(Xb, Ca):
    """Rounded dot products of one row batch with the centers, as float64 (batch x k)."""
    return _np.round((Xb @ Ca).astype(_np.float64), 10)


def _sims(X, C, block=1024):
    """n x len(C) matrix of dot products (list of lists), rounded to 10 decimals. For the numpy backend use only on small X."""
    if not C:
        return [[] for _ in range(len(X))]
    if _USE_NP:
        Ca = _ca(C)
        out = []
        for s in range(0, len(X), block):
            out.extend(_sim_block(X[s:s + block], Ca).tolist())
        return out
    return [[round(math.sumprod(x, c), 10) for c in C] for x in X]


def _col(X, c):
    """Dot products of every row with ONE center (list of n floats, rounded to 10 decimals)."""
    if _USE_NP:
        return _np.round((X @ _np.array(c, dtype=_np.float32)).astype(_np.float64), 10).tolist()
    return [s[0] for s in _sims(X, [c])]


def _assign(X, centers):
    """Nearest center per row: (labels, best rounded similarity per row). The numpy backend works in row batches."""
    if _USE_NP:
        Ca = _ca(centers)
        lab, best = [], []
        for s in range(0, len(X), _BLOCK):
            S = _sim_block(X[s:s + _BLOCK], Ca)
            a = S.argmax(axis=1)
            lab.extend(a.tolist())
            best.extend(S[_np.arange(len(a)), a].tolist())
        return lab, best
    S = _sims(X, centers)
    lab = [_argmax(r) for r in S]
    return lab, [S[i][lab[i]] for i in range(len(S))]


def _msum(X, idx):
    if _USE_NP:
        ix = _np.asarray(list(idx), dtype=_np.intp)
        acc = _np.zeros(X.shape[1], dtype=_np.float64)
        for s in range(0, len(ix), _BLOCK):
            acc += X[ix[s:s + _BLOCK]].sum(axis=0, dtype=_np.float64)
        return acc.tolist()
    d = len(X[0])
    acc = [0.0] * d
    for i in idx:
        r = X[i]
        for j in range(d):
            acc[j] += r[j]
    return acc


def _unit(v):
    nr = math.sqrt(sum(x * x for x in v))
    return [x / nr for x in v] if nr > 0 else list(v)


def _dot(a, b):
    return math.sumprod(a, b)


def _argmax(row):
    best, bi = row[0], 0
    for i in range(1, len(row)):
        if row[i] > best:
            best, bi = row[i], i
    return bi


# ----------------------------------------------------------------------------- clustering
def _kmeans(X, k, seed, max_iter):
    n = len(X)
    rng = random.Random(seed)
    chosen = [rng.randrange(n)]
    centers = [_row(X, chosen[0])]
    mind = [max(0.0, 1.0 - s) for s in _col(X, centers[0])]
    while len(centers) < k:
        total = sum(m * m for m in mind)
        if total <= 1e-18:
            pick = next(i for i in range(n) if i not in chosen)
        else:
            r, acc, pick = rng.random() * total, 0.0, n - 1
            for i in range(n):
                acc += mind[i] * mind[i]
                if acc >= r and mind[i] > 0:
                    pick = i
                    break
        chosen.append(pick)
        c = _row(X, pick)
        centers.append(c)
        mind = [min(m, max(0.0, 1.0 - s)) for m, s in zip(mind, _col(X, c))]
    labels = None
    for _ in range(max_iter):
        new, best = _assign(X, centers)
        # an empty cluster is re-seeded with the point that fits its own center worst
        sizes = Counter(new)
        for c in range(k):
            if sizes.get(c, 0) == 0:
                worst = min((i for i in range(n) if sizes[new[i]] > 1), key=lambda i: (best[i], i))
                sizes[new[worst]] -= 1
                new[worst] = c
                best[worst] = _sims(X[worst:worst + 1], [centers[c]])[0][0]
                sizes[c] = 1
        if new == labels:
            break
        labels = new
        groups = [[] for _ in range(k)]
        for i, l in enumerate(labels):
            groups[l].append(i)
        centers = [_unit(_msum(X, g)) for g in groups]
    return _canon(labels, X), sum(best)


def _kmeans_best(X, k, seeds, max_iter):
    """Best of one k-means++ run per configured seed (highest total cosine to own center; ties -> earlier seed)."""
    best, bo = None, -1e18
    for sd in seeds:
        lab, obj = _kmeans(X, k, sd, max_iter)
        if obj > bo + 1e-9:
            best, bo = lab, obj
    return best


def _canon(labels, X=None):
    """Relabel clusters in order of first appearance (input is span-sorted, so this is deterministic)."""
    m, out = {}, []
    for l in labels:
        if l not in m:
            m[l] = len(m)
        out.append(m[l])
    return out


def _silhouette(X, labels):
    """Mean cosine silhouette, exact and O(n*k*d): sum_j(1 - xi.xj) over a cluster = size - xi.S_cluster."""
    ks = sorted(set(labels))
    if len(ks) < 2:
        return 0.0
    groups = {c: [] for c in ks}
    for i, l in enumerate(labels):
        groups[l].append(i)
    sums = [_msum(X, groups[c]) for c in ks]
    pos = {c: j for j, c in enumerate(ks)}
    if _USE_NP:
        sizes = _np.array([len(groups[c]) for c in ks], dtype=_np.float64)
        own = _np.array([pos[l] for l in labels], dtype=_np.intp)
        Sa, tot = _ca(sums), 0.0
        for s in range(0, len(labels), _BLOCK):
            S = _sim_block(X[s:s + _BLOCK], Sa)
            o = own[s:s + _BLOCK]
            r = _np.arange(len(o))
            na = sizes[o]
            ok = na >= 2
            a = (na - S[r, o]) / _np.maximum(na - 1, 1)
            other = 1.0 - S / sizes[None, :]
            other[r, o] = _np.inf
            b = other.min(axis=1)
            den = _np.maximum(a, b)
            term = _np.where(den > 0, (b - a) / _np.where(den > 0, den, 1.0), 0.0)
            tot += float(term[ok].sum())
        return tot / len(labels)
    S = _sims(X, sums)
    tot = 0.0
    for i, l in enumerate(labels):
        na = len(groups[l])
        if na < 2:
            continue
        ai = pos[l]
        a = (na - S[i][ai]) / (na - 1)
        b = min(1.0 - S[i][pos[c]] / len(groups[c]) for c in ks if c != l)
        den = max(a, b)
        tot += (b - a) / den if den > 0 else 0.0
    return tot / len(labels)


def _choose_k(X, p):
    n = len(X)
    kmax = min(p["k_max"], n // p["min_cluster_size"])
    cache, best_s, best_k, nogain = {}, -2.0, 1, 0
    for k in range(p["k_min"], kmax + 1, p["k_step"]):
        lab = _kmeans_best(X, k, p["seeds"], p["kmeans_max_iter"])
        if min(Counter(lab).values()) < p["min_cluster_size"]:
            break
        s = _silhouette(X, lab)
        cache[k] = lab
        gain = s - best_s
        if s > best_s:
            best_s, best_k = s, k
        nogain = nogain + 1 if gain < p["gain_epsilon"] else 0
        if nogain >= p["patience"]:
            break
    return best_k, cache.get(best_k), (None if best_k == 1 else _r(best_s))


def _agglomerate(centers):
    """UPGMA over unit centers. Returns (merges, heights); merges = list of (a, b) active-cluster ids, cluster ids grow
    from len(centers)."""
    m = len(centers)
    D = {}
    for i in range(m):
        for j in range(i + 1, m):
            D[(i, j)] = round(1.0 - _dot(centers[i], centers[j]), 10)
    size = {i: 1 for i in range(m)}
    active = list(range(m))
    merges, heights, nxt = [], [], m
    while len(active) > 1:
        a, b = min(((x, y) for x in active for y in active if x < y), key=lambda ab: (D[ab], ab))
        h = D[(a, b)]
        for c in active:
            if c in (a, b):
                continue
            da = D[(min(a, c), max(a, c))]
            db = D[(min(b, c), max(b, c))]
            D[(c, nxt)] = round((size[a] * da + size[b] * db) / (size[a] + size[b]), 10)
        size[nxt] = size[a] + size[b]
        active = [c for c in active if c not in (a, b)] + [nxt]
        merges.append((a, b))
        heights.append(max(h, heights[-1]) if heights else h)
        nxt += 1
    return merges, heights


def _cut(centers, rng_cfg):
    """Cut the dendrogram of `centers` at the largest merge-height gap with c in the configured range.
    Returns canonical cluster index per center."""
    m = len(centers)
    if m <= 1:
        return [0] * m
    merges, h = _agglomerate(centers)
    hi = min(rng_cfg["max_count"], int(m * rng_cfg["max_frac"]), m - 1)
    lo = min(rng_cfg["min_count"], hi)
    hi = max(hi, 1)
    lo = max(lo, 1)
    cands = [c for c in range(lo, hi + 1) if c >= 2] or [1]
    best_c, best_g = cands[0], -1.0
    for c in cands:
        g = (h[m - c] - h[m - c - 1]) if c >= 2 else 0.0
        if g > best_g + 1e-12:
            best_c, best_g = c, g
    parent = {i: i for i in range(m)}  # replay m-c merges
    groups = {i: [i] for i in range(m)}
    nxt = m
    for a, b in merges[:m - best_c]:
        groups[nxt] = groups.pop(a) + groups.pop(b)
        nxt += 1
    lab = [0] * m
    for ci, (_, mem) in enumerate(sorted(groups.items(), key=lambda kv: min(kv[1]))):
        for i in mem:
            lab[i] = ci
    return lab


def _hierarchy(X, sub_labels, p):
    """Given subtopic labels return (topic_labels, domain_labels) per chunk plus parent maps."""
    ks = max(sub_labels) + 1
    groups = [[] for _ in range(ks)]
    for i, l in enumerate(sub_labels):
        groups[l].append(i)
    sub_c = [_unit(_msum(X, g)) for g in groups]
    t_of_s = _cut(sub_c, p["topic_range"])
    kt = max(t_of_s) + 1
    tg = [[] for _ in range(kt)]
    for s, t in enumerate(t_of_s):
        tg[t].extend(groups[s])
    top_c = [_unit(_msum(X, sorted(g))) for g in tg]
    d_of_t = _cut(top_c, p["domain_range"])
    return t_of_s, d_of_t


def _ari(a, b):
    n = len(a)
    if n < 2:
        return 1.0
    comb = lambda x: x * (x - 1) / 2.0
    cont = Counter(zip(a, b))
    sa, sb = Counter(a), Counter(b)
    sij = sum(comb(v) for v in cont.values())
    s1, s2 = sum(comb(v) for v in sa.values()), sum(comb(v) for v in sb.values())
    exp = s1 * s2 / comb(n)
    mx = (s1 + s2) / 2.0
    if mx == exp:
        return 1.0
    return (sij - exp) / (mx - exp)


# ----------------------------------------------------------------------------- terms
def _tokens(text, minlen):
    return [t for t in _TOKEN.findall((text or "").lower()) if len(t) >= minlen]


def _node_terms(texts, member_groups, p):
    """c-TF-IDF terms. member_groups: list of index lists (one class each). Returns list of term lists."""
    n = max(1, len(texts))
    classes_of = [[] for _ in texts]  # streaming: one Counter per text at a time, never held for all texts
    for ci, g in enumerate(member_groups):
        for i in g:
            classes_of[i].append(ci)
    classes = [Counter() for _ in member_groups]
    df = Counter()
    for i, t in enumerate(texts):
        c = Counter(_tokens(t, p["term_min_len"]))
        df.update(c.keys())
        for ci in classes_of[i]:
            classes[ci].update(c)
    banned = {t for t, c in df.items() if c / n > p["term_max_df"]}
    total = Counter()
    for c in classes:
        total.update(c)
    avg = sum(sum(c.values()) for c in classes) / max(1, len(classes))

    def top(use_ban):
        res = []
        for c in classes:
            sc = {t: f * math.log(1.0 + avg / total[t]) for t, f in c.items() if not (use_ban and t in banned)}
            res.append([t for t, _ in sorted(sc.items(), key=lambda kv: (-round(kv[1], 9), kv[0]))[:p["term_count"]]])
        return res
    a, b = top(True), None
    if any(not x for x in a):
        b = top(False)
        a = [x or y for x, y in zip(a, b)]
    return a


def _slug(parts, maxlen):
    s = "-".join(re.sub(r"[^a-z0-9]+", "", t.lower()) for t in parts if re.sub(r"[^a-z0-9]+", "", t.lower()))
    return s[:maxlen].strip("-")


def _fold(s):
    return " ".join(s.lower().split())


# ----------------------------------------------------------------------------- null
def _quantile(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return s[int(q * (len(s) - 1))]


def _level_null(X, labels, k, p):
    """Return {"dist_q", "margin_q", "spread", "per_seed": [...]} for one level."""
    n = len(labels)
    per_seed_d, all_d, all_m = [], [], []
    for sd in p["null_seeds"]:
        rng = random.Random(sd)
        lab = list(labels) if k >= 2 else [i % 2 for i in range(n)]
        rng.shuffle(lab)
        g = max(lab) + 1
        groups = [[] for _ in range(g)]
        for i, l in enumerate(lab):
            groups[l].append(i)
        keep = [j for j in range(g) if groups[j]]
        centers = [_unit(_msum(X, groups[j])) for j in keep]
        pos = {j: t for t, j in enumerate(keep)}
        ds = []
        if _USE_NP:
            g1 = len(centers) - 1
            if g1 >= 1:
                Ca = _ca(centers)
                own = _np.array([pos[l] for l in lab], dtype=_np.intp)
                for s in range(0, n, _BLOCK):
                    S = _sim_block(X[s:s + _BLOCK], Ca)
                    S[_np.arange(len(S)), own[s:s + _BLOCK]] = -_np.inf
                    if g1 >= 2:
                        part = _np.partition(S, g1 - 1, axis=1)
                        top1, top2 = part[:, -1], part[:, -2]
                        all_m.extend(_np.round(top1 - top2, 10).tolist())
                    else:
                        top1 = S.max(axis=1)
                    ds.extend(_np.round(1.0 - top1, 10).tolist())
        else:
            S = _sims(X, centers)
            for i in range(n):
                own = pos.get(lab[i])
                row = [s for t, s in enumerate(S[i]) if t != own]
                if not row:
                    continue
                row.sort(reverse=True)
                ds.append(round(1.0 - row[0], 10))
                if len(row) >= 2:
                    all_m.append(round(row[0] - row[1], 10))
        per_seed_d.append(_quantile(ds, p["null_q_dist"]))
        all_d.extend(ds)
    pv = [x for x in per_seed_d if x is not None]
    dq = _quantile(all_d, p["null_q_dist"])
    mq = _quantile(all_m, p["null_q_margin"]) if k >= 3 else None
    return {"dist_q": _r(dq) if dq is not None else 2.0, "margin_q": _r(mq) if mq is not None else None,
            "spread": _r(max(pv) - min(pv)) if pv else 0.0, "q_dist": p["null_q_dist"], "q_margin": p["null_q_margin"],
            "n_seeds": len(p["null_seeds"]), "n_null": len(all_d)}


# ----------------------------------------------------------------------------- model
class Model:
    """Result of build(). to_dict()/dumps() are canonical (sorted keys), from_dict() restores a snapshot."""

    FIELDS = ("version", "params", "nodes", "assoc", "null", "retired_keys", "span_order", "meta")

    def __init__(self, **kw):
        self.__dict__.update({f: kw.get(f) for f in self.FIELDS})

    def to_dict(self):
        return {f: self.__dict__[f] for f in self.FIELDS}

    def dumps(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_dict(cls, d):
        d = json.loads(json.dumps(d))
        d["assoc"] = {int(k): v for k, v in d.get("assoc", {}).items()}
        d["null"] = d.get("null", {})
        return cls(**{f: d.get(f) for f in cls.FIELDS})

    def node(self, key):
        return next((n for n in self.nodes if n["key"] == key), None)


def _make_version(params, span_ids):
    blob = json.dumps({"params": params, "seeds": params["seeds"], "spans": sorted(span_ids)}, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def version_for(span_ids, params):
    """The version `build` would give this span set and params, without building (WP-W no-op rule: the version hashes params,
    seeds and the sorted span ids only, never vectors)."""
    return _make_version(copy.deepcopy(params), [int(s) for s in span_ids])


def _prior_nodes(prior):
    if prior is None:
        return []
    nodes = prior.nodes if isinstance(prior, Model) else prior.get("nodes", [])
    return [n for n in nodes if n.get("key") and n.get("level") in LEVELS]


def _match_prior(new_nodes, prior_nodes, thr):
    """Greedy one-to-one member-overlap Jaccard per level. Returns {new_index: prior_key} and the matched prior keys."""
    out, used = {}, set()
    for lv in LEVELS:
        cand = []
        for i, nn in enumerate(new_nodes):
            if nn["level"] != lv:
                continue
            ms = set(nn["members"])
            for pn in prior_nodes:
                if pn["level"] != lv:
                    continue
                pm = set(pn.get("members", []))
                u = len(ms | pm)
                j = len(ms & pm) / u if u else 0.0
                if j >= thr and j > 0:
                    cand.append((-round(j, 12), i, pn["key"]))
        for negj, i, key in sorted(cand):
            if i in out or key in used:
                continue
            out[i] = key
            used.add(key)
    return out, used


_MULTI_LEVELS = ("subtopic", "topic")  # chunk links exist for these; a chunk's domains derive from its topic links


def _multi(info, sims_by_level, p, strict):
    """Generous multi-membership (TAXO-ROUTE-R1-1110). Returns [(level, key, cosine, kind)] EXCLUDING the strict link.
    Per level: every node with s >= s_best - rel_margin, s >= abs_floor and count >= min_support, best first, at most
    max_links; kind 'cosine_r<rank>'. A chunk with no strict link and nothing within the rule gets its nearest node as
    'cosine_soft'. Only kind 'cosine' (the strict link) feeds display metrics; the others are routing-only."""
    m = p.get("multi") or {}
    if not m.get("enabled"):
        return []
    out = []
    for lv in _MULTI_LEVELS:
        lvinfo = info[lv]
        if not lvinfo["keys"]:
            continue
        s = sims_by_level[lv]
        bi = _argmax(s)
        nl = lvinfo["null"] or {}
        mq = nl.get("margin_q")
        rel = m["rel_margin"] if m.get("rel_margin") is not None else (m["rel_margin_mult"] * mq if mq is not None else 0.05)
        dq = nl.get("dist_q")
        floor = m["abs_floor"] if m.get("abs_floor") is not None else (1.0 - m["abs_floor_mult"] * dq if dq is not None else -1.0)
        best = s[bi]
        cand = [(-s[j], j) for j in range(len(s)) if s[j] >= best - rel and s[j] >= floor and lvinfo["counts"][j] >= p["min_support"]]
        cand.sort()
        got = False
        for rank, (_neg, j) in enumerate(cand[:m["max_links"][lv]], 1):
            got = True
            if strict.get("level") == lv and lvinfo["keys"][j] == strict.get(lv):
                continue
            out.append((lv, lvinfo["keys"][j], round(float(s[j]), 6), "cosine_r%d" % rank))
        if not got and strict.get("level") is None and m.get("soft_outliers"):
            out.append((lv, lvinfo["keys"][bi], round(float(best), 6), "cosine_soft"))
    return out


def _decide(info, sims_by_level, p):
    """Association decision for one chunk (strict decision plus optional generous 'multi' routing links)."""
    a = _decide_strict(info, sims_by_level, p)
    mu = _multi(info, sims_by_level, p, a)
    if mu:
        a["multi"] = mu
    return a


def _decide_strict(info, sims_by_level, p):
    """Strict association decision for one chunk. info: model level data; sims_by_level: {level: [sim per node]}."""
    first_reason, first = None, None
    for lv in LEVELS:
        lvinfo = info[lv]
        if not lvinfo["keys"]:
            continue
        s = sims_by_level[lv]
        bi = _argmax(s)
        d1 = round(1.0 - s[bi], 10)
        margin = None
        if len(s) >= 2:
            second = max(x for t, x in enumerate(s) if t != bi)
            margin = round((1.0 - second) - d1, 10)
        nl = lvinfo["null"]
        reason = None
        if d1 > nl["dist_q"]:
            reason = "NO_FIT"
        elif lvinfo["counts"][bi] < p["min_support"]:
            reason = "LOW_SUPPORT"
        elif margin is not None and nl["margin_q"] is not None and margin < nl["margin_q"]:
            reason = "SPLIT"
        if first is None:
            first = (d1, margin)
            first_reason = reason
        if reason is None:
            sub = lvinfo["keys"][bi] if lv == "subtopic" else None
            top = lvinfo["keys"][bi] if lv == "topic" else (info["subtopic"]["parent"][bi] if lv == "subtopic" else None)
            if lv == "subtopic":
                dom = info["topic"]["parent"][info["subtopic"]["parent_idx"][bi]]
            elif lv == "topic":
                dom = lvinfo["parent"][bi]
            else:
                dom = lvinfo["keys"][bi]
            return {"subtopic": sub, "topic": top, "domain": dom, "level": lv, "distance": _r(d1),
                    "margin": None if margin is None else _r(margin), "reason": None if lv == "subtopic" else first_reason}
    return {"subtopic": None, "topic": None, "domain": None, "level": None, "distance": _r(first[0]) if first else None,
            "margin": None if not first or first[1] is None else _r(first[1]), "reason": first_reason or "NO_FIT"}


def _level_info(model):
    """Per-level lookup structures derived from model.nodes / model.null (rebuilt on demand, cheap)."""
    info = {lv: {"keys": [], "centers": [], "counts": [], "parent": [], "parent_idx": [], "null": model.null.get(lv)}
            for lv in LEVELS}
    idx = {lv: {} for lv in LEVELS}
    for n in model.nodes:
        lv = info[n["level"]]
        idx[n["level"]][n["key"]] = len(lv["keys"])
        lv["keys"].append(n["key"])
        lv["centers"].append(n["center"])
        lv["counts"].append(len(n["members"]))
        lv["parent"].append(n["parent_key"])
    for lv, up in (("subtopic", "topic"), ("topic", "domain")):
        info[lv]["parent_idx"] = [idx[up].get(k, 0) for k in info[lv]["parent"]]
    return info


def estimate_build_bytes(n, d, k_max=40, text_bytes=0):
    """Upper-bound estimate of the EXTRA memory (bytes) build() + quality() need on top of the caller's own n x d float32
    matrix and texts: the normalised float32 copy plus one more matrix-sized slack (a reorder copy when the input is not
    sorted, or the quality() copy), per-row bookkeeping (assoc entries, label lists), c-TF-IDF counters (about half the text
    size) and the Python-float center lists (about (2*k_max + 40) centers of d floats at 32 bytes per float).
    Checked against the real measurement by tests/test_chunk_taxo_mem_1109.py."""
    n, d, k_max, text_bytes = int(n), int(d), int(k_max), int(text_bytes)
    return 8 * n * d + 3072 * n + text_bytes // 2 + (2 * k_max + 40) * d * 32


def build(chunks, vecs, params, prior=None):
    """Derive the model. chunks: [(span_id, source_id, text)], vecs: aligned vectors. See module docstring."""
    p = copy.deepcopy(params)
    if len(chunks) != len(vecs):
        raise ValueError("chunks and vecs differ in length")
    if not chunks:
        raise ValueError("no chunks")
    order = sorted(range(len(chunks)), key=lambda i: int(chunks[i][0]))
    span_ids = [int(chunks[i][0]) for i in order]
    if len(set(span_ids)) != len(span_ids):
        raise ValueError("duplicate span_id")
    texts = [chunks[i][2] for i in order]
    if _np is not None and isinstance(vecs, _np.ndarray):
        if order == list(range(len(order))):
            X = _prep(vecs)  # already sorted: no reorder copy
        else:
            X = _prep(vecs[order], owned=True)  # the fancy-index copy is normalised in place
    else:
        X = _prep([vecs[i] for i in order])
    n = len(span_ids)

    k, sub_lab, sil = _choose_k(X, p)
    if sub_lab is None:
        sub_lab = [0] * n
    t_of_s, d_of_t = _hierarchy(X, sub_lab, p)
    top_lab = [t_of_s[l] for l in sub_lab]
    dom_lab = [d_of_t[t] for t in top_lab]
    level_labels = {"subtopic": sub_lab, "topic": top_lab, "domain": dom_lab}

    # raw nodes
    raw = []  # dicts without key/label yet
    idx_of = {}
    for lv in LEVELS:
        lab = level_labels[lv]
        groups = [[] for _ in range(max(lab) + 1)]
        for i, l in enumerate(lab):
            groups[l].append(i)
        for ci, g in enumerate(groups):
            idx_of[(lv, ci)] = len(raw)
            raw.append({"level": lv, "ci": ci, "g": g, "center": _unit(_msum(X, g)),
                        "members": [span_ids[i] for i in g]})
    for (lv, ci), ri in idx_of.items():
        if lv == "subtopic":
            raw[ri]["parent"] = idx_of[("topic", t_of_s[ci])]
        elif lv == "topic":
            raw[ri]["parent"] = idx_of[("domain", d_of_t[ci])]
        else:
            raw[ri]["parent"] = None
    terms = _node_terms(texts, [r["g"] for r in raw], p)
    for r, t in zip(raw, terms):
        r["terms"] = t

    # identity
    pnodes = _prior_nodes(prior)
    match, used_prior = _match_prior(raw, pnodes, p["identity_jaccard"])
    prior_keys = {pn["key"] for pn in pnodes}

    # keys
    taken = set(match.values())
    for i, r in enumerate(raw):
        if i in match:
            r["key"], r["reused_from"] = match[i], match[i]
            continue
        h = hashlib.sha1((r["level"] + ",".join(map(str, r["members"]))).encode()).hexdigest()[:6]
        base = "%s-%s-%s" % (_PREFIX[r["level"]], _slug(r["terms"][:p["label_terms"]], 40) or "node", h)
        base, c, key = base[:60].strip("-"), 1, None
        key = base
        while key in taken or key in prior_keys:
            c += 1
            key = "%s-%d" % (base, c)
        taken.add(key)
        r["key"], r["reused_from"] = key, None

    # labels (unique per level after folding)
    seen = {lv: set() for lv in LEVELS}
    for r in raw:
        lv, terms_ = r["level"], r["terms"]
        label = None
        for take in range(p["label_terms"], max(len(terms_), p["label_terms"]) + 1):
            cand = ", ".join(terms_[:take]) or "%s cluster" % r["level"]
            cand = cand.encode("ascii", "ignore").decode()[:120].strip() or r["level"]
            if _fold(cand) not in seen[lv]:
                label = cand
                break
        c = 1
        while label is None:
            c += 1
            suf = " #%d" % c
            cand = (", ".join(terms_[:p["label_terms"]]) or r["level"])[:120 - len(suf)] + suf
            if _fold(cand) not in seen[lv]:
                label = cand
        seen[lv].add(_fold(label))
        r["label"] = label

    nodes = []
    for r in raw:
        nodes.append({"key": r["key"], "level": r["level"],
                      "parent_key": raw[r["parent"]]["key"] if r["parent"] is not None else None,
                      "center": [_r(x) for x in r["center"]], "members": r["members"], "terms": r["terms"],
                      "label": r["label"], "reused_from": r["reused_from"]})

    null = {lv: _level_null(X, level_labels[lv], max(level_labels[lv]) + 1, p) for lv in LEVELS}
    model = Model(version=_make_version(p, span_ids), params=p, nodes=nodes, assoc={}, null=null,
                  retired_keys=sorted(prior_keys - used_prior), span_order=span_ids,
                  meta={"n": n, "k": k, "silhouette_k": sil, "levels": {lv: max(level_labels[lv]) + 1 for lv in LEVELS},
                        "numpy": bool(_USE_NP)})
    info = _level_info(model)
    if _USE_NP:
        cas = {lv: _ca(info[lv]["centers"]) for lv in LEVELS}
        for s in range(0, n, _BLOCK):  # nearest-center per row batch: no n x k temporary
            blk = {lv: _sim_block(X[s:s + _BLOCK], cas[lv]) for lv in LEVELS}
            for r in range(min(_BLOCK, n - s)):
                model.assoc[span_ids[s + r]] = _decide(info, {lv: blk[lv][r].tolist() for lv in LEVELS}, p)
    else:
        sims = {lv: _sims(X, info[lv]["centers"]) for lv in LEVELS}
        for i, sid in enumerate(span_ids):
            model.assoc[sid] = _decide(info, {lv: sims[lv][i] for lv in LEVELS}, p)
    return model


def associate(model, vec):
    """Association entry for one new vector (nearest-center with the model's null thresholds)."""
    v = _unit([float(x) for x in vec])
    if not any(v) or not all(math.isfinite(x) for x in v):
        raise ValueError("bad vector")
    info = _level_info(model)
    if len(v) != len(model.nodes[0]["center"]):
        raise ValueError("dimension mismatch")
    sims = {lv: [round(_dot(v, c), 10) for c in info[lv]["centers"]] for lv in LEVELS}
    return _decide(info, sims, model.params)


def doc_subjects(model, span_source):
    """Per source: UNION {node_key: chunk_count} over its chunks' assigned nodes (all levels carried), and VOTE: top
    topic / domain by share of the source's chunks (abstained chunks count in the denominator) with vote_min_share and
    vote_min_lead. "share" is the topic share; "domain_share" is an extra key."""
    p = model.params
    per = {}
    for sid, src in span_source.items():
        a = model.assoc.get(sid)
        if a is None:
            continue
        per.setdefault(src, []).append(a)
    out = {}
    for src in sorted(per):
        chunks = per[src]
        union = Counter()
        tc, dc = Counter(), Counter()
        for a in chunks:
            for lv in LEVELS:
                if a[lv]:
                    union[a[lv]] += 1
            if a["topic"]:
                tc[a["topic"]] += 1
            if a["domain"]:
                dc[a["domain"]] += 1

        def vote(c):
            if not c:
                return None, 0.0
            ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
            share = ranked[0][1] / len(chunks)
            lead = share - (ranked[1][1] / len(chunks) if len(ranked) > 1 else 0.0)
            ok = share >= p["vote_min_share"] and lead >= p["vote_min_lead"]
            return (ranked[0][0] if ok else None), share
        tk, ts = vote(tc)
        dk, ds = vote(dc)
        out[src] = {"union": dict(sorted(union.items())),
                    "vote": {"topic": tk, "domain": dk, "share": _r(ts), "domain_share": _r(ds)}}
    return out


def quality(model, vecs):
    """Mechanical metrics. vecs must be aligned with model.span_order (i.e. span_id order); a dict span_id->vec also works."""
    p = model.params
    if isinstance(vecs, dict):
        vecs = [vecs[s] for s in model.span_order]
    X = _prep(vecs)
    n = len(model.span_order)
    pos = {s: i for i, s in enumerate(model.span_order)}
    labels, keysets = {}, {}
    for lv in LEVELS:
        lab = [0] * n
        ks = [nd for nd in model.nodes if nd["level"] == lv]
        for ci, nd in enumerate(ks):
            for s in nd["members"]:
                lab[pos[s]] = ci
        labels[lv] = lab
    sil = {lv: _r(_silhouette(X, labels[lv])) if len(set(labels[lv])) > 1 else None for lv in LEVELS}
    # stability: re-run each other configured seed at the chosen k, compare to the build labels
    k = model.meta["k"]
    stab = {"seeds": [], "ari": {lv: None for lv in LEVELS}, "center_match_rate": None}
    others = p["seeds"][1:]
    if k >= 2 and others:
        aris = {lv: [] for lv in LEVELS}
        rates = []
        base_c = [nd["center"] for nd in model.nodes if nd["level"] == "subtopic"]
        for sd in others:
            lab = _kmeans(X, k, sd, p["kmeans_max_iter"])[0]
            t_of_s, d_of_t = _hierarchy(X, lab, p)
            alt = {"subtopic": lab, "topic": [t_of_s[l] for l in lab],
                   "domain": [d_of_t[t_of_s[l]] for l in lab]}
            for lv in LEVELS:
                aris[lv].append(_ari(labels[lv], alt[lv]))
            groups = [[] for _ in range(max(lab) + 1)]
            for i, l in enumerate(lab):
                groups[l].append(i)
            alt_c = [_unit(_msum(X, g)) for g in groups]
            free = set(range(len(alt_c)))
            hit = 0
            for bc in base_c:  # greedy best match of each base center
                if not free:
                    break
                j = max(sorted(free), key=lambda t: (round(_dot(bc, alt_c[t]), 10), -t))
                if _dot(bc, alt_c[j]) >= p["stability_match_cos"]:
                    hit += 1
                free.discard(j)
            rates.append(hit / len(base_c))
            stab["seeds"].append(sd)
        stab["ari"] = {lv: _r(sum(v) / len(v)) for lv, v in aris.items()}
        stab["center_match_rate"] = _r(sum(rates) / len(rates))
    cov = {lv: 0 for lv in LEVELS}
    mix = Counter()
    for s in model.span_order:
        a = model.assoc[s]
        for lv in LEVELS:
            if a[lv]:
                cov[lv] += 1
        mix[a["level"] or "abstain"] += 1
        if a["reason"]:
            mix["reason_" + a["reason"]] += 1
    return {"n": n, "silhouette": sil, "stability": stab,
            "null_floor_spread": {lv: model.null[lv]["spread"] for lv in LEVELS},
            "coverage": {lv: _r(cov[lv] / n) for lv in LEVELS},
            "abstention": {"abstained_share": _r(mix["abstain"] / n),
                           "backed_off_share": _r((mix["topic"] + mix["domain"]) / n),
                           "counts": {k2: v for k2, v in sorted(mix.items())}}}


def display_gate(metrics, gate_cfg):
    """(ok, reasons). A threshold set to None is skipped; a missing metric under an active threshold fails."""
    reasons = []

    def need(name, val, thr, kind):
        if thr is None:
            return
        if val is None:
            reasons.append("%s unavailable" % name)
        elif kind == "min" and val < thr:
            reasons.append("%s %.4f < %.4f" % (name, val, thr))
        elif kind == "max" and val > thr:
            reasons.append("%s %.4f > %.4f" % (name, val, thr))
    g = gate_cfg
    for lv in LEVELS:
        need("silhouette_" + lv, metrics.get("silhouette", {}).get(lv), g.get("min_silhouette_" + lv), "min")
    need("stability_ari_subtopic", metrics.get("stability", {}).get("ari", {}).get("subtopic"),
         g.get("min_stability_ari"), "min")
    need("center_match_rate", metrics.get("stability", {}).get("center_match_rate"), g.get("min_center_match_rate"), "min")
    need("coverage_subtopic", metrics.get("coverage", {}).get("subtopic"), g.get("min_coverage_subtopic"), "min")
    need("coverage_domain", metrics.get("coverage", {}).get("domain"), g.get("min_coverage_domain"), "min")
    spreads = [v for v in metrics.get("null_floor_spread", {}).values() if v is not None]
    need("null_floor_spread", max(spreads) if spreads else None, g.get("max_null_floor_spread"), "max")
    need("abstained_share", metrics.get("abstention", {}).get("abstained_share"), g.get("max_abstain_share"), "max")
    return (not reasons), reasons
