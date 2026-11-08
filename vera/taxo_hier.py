"""TAXO-HIER-1108: hierarchical matcher (domain -> topic -> subtopic) over the EXISTING taxonomy, with back-off and abstention.

Owner 2026-11-08: "We should be able to use descriptive statistics and semantic weight to guess at sub-topics, topics, and
domains"; amended: "our guess doesn't have to be "correct per human," but "functional for our purposes"". Design:
work/TAXO-HIER-1108/design.md.

Pure, deterministic, stdlib only; no network, no DB, no writes. Node data comes only from registry rows (id, label, field,
parent_topic_key, anchor_text, optional description), never from code. No keyword list: term weights are corpus IDF.

Per chunk and node: lexical channel = cosine of sublinear TF-IDF vectors (IDF over the corpus chunks); optional semantic
channel = cosine of caller-supplied embeddings (`embed`; never available in the API). Each score is turned into a percentile
of a null distribution: the best score of the chunk against k random pseudo-nodes (corpus-word bags as long as the real
nodes) for a candidate set of k nodes. A node scores as the max over itself and its descendants (the null mirrors that). Fit = mean percentile over the available channels. A chunk votes for its best node
when fit >= fit_quantile. A document is accepted at a level when top_share >= min_top_share and agree >= min_agree; levels
are tried top-down (domain, then that domain's topics, then that topic's subtopics) and the deepest accepted node wins.

CALIBRATION: UNCALIBRATED. fit_quantile 0.95, min_top_share 0.25 (= v2 min_vote_share), min_agree 0.50 are starting values
only. They are chosen by the owner-run kit work/TAXO-HIER-1108/kit/run_hier_eval.sh (grid on the development split under
the functional tests, frozen, then gated on the holdout). config/taxo_hier.json refuses mode "active" until CALIBRATED."""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

CLASSIFIER_NAME = "taxo_hier"
CLASSIFIER_VERSION = "1"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "taxo_hier.json"
STOPWORDS_PATH = Path(__file__).resolve().parent.parent / "config" / "chunk_topic_terms.json"
LEVELS = ("domain", "topic", "subtopic")
MODES = ("off", "shadow", "active")
REASONS = ("NO_TEXT", "NO_FIT", "LOW_SUPPORT", "SPLIT", "NO_CHILD_NODES")
_TOKEN = re.compile(r"[a-z0-9]+")  # identical to vera.micro_topic._TOKEN
_RANGES = {"fit_quantile": (0.5, 0.9999), "min_top_share": (0.0, 1.0), "min_agree": (0.0, 1.0)}
_WORK = {"null_replicates": (1, 20, 3), "null_chunks": (50, 20000, 2000), "min_token_len": (1, 20, 3), "seed": (0, 2**31, 1108)}
DEFAULT_PARAMS = {"fit_quantile": 0.95, "min_top_share": 0.25, "min_agree": 0.5, "null_replicates": 3, "null_chunks": 2000,
                  "min_token_len": 3, "seed": 1108, "semantic_weight": 0.5}


class ConfigError(ValueError):
    pass


def params_from(raw: dict | None) -> dict:
    """Validated parameters (defaults filled). Raises ConfigError on a bad value (fail closed)."""
    p = {**DEFAULT_PARAMS, **(raw or {})}
    for k, (lo, hi) in _RANGES.items():
        v = p[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= float(v) <= hi:
            raise ConfigError(f"taxo_hier: bad {k}")
        p[k] = float(v)
    for k, (lo, hi, _d) in _WORK.items():
        v = p[k]
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise ConfigError(f"taxo_hier: bad {k}")
    w = p["semantic_weight"]
    if isinstance(w, bool) or not isinstance(w, (int, float)) or not 0.0 <= float(w) <= 1.0:
        raise ConfigError("taxo_hier: bad semantic_weight")
    p["semantic_weight"] = float(w)
    return p


def load_config(path=None) -> dict:
    """config/taxo_hier.json -> {"mode", "params", ...}. Missing file = mode off. "active" is refused unless
    calibration_status is CALIBRATED with a non-empty calibration_note (calibration rule R1)."""
    p = Path(path or CONFIG_PATH)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"mode": "off", "params": params_from(None)}
    except (OSError, ValueError) as e:
        raise ConfigError(f"taxo_hier: cannot read {p.name}: {type(e).__name__}") from e
    mode = raw.get("mode", "off")
    if mode not in MODES:
        raise ConfigError("taxo_hier: bad mode")
    if mode == "active" and (raw.get("calibration_status") != "CALIBRATED" or not str(raw.get("calibration_note") or "").strip()):
        raise ConfigError("taxo_hier: mode active refused: calibration_status must be CALIBRATED with a calibration_note")
    return {**raw, "mode": mode, "params": params_from(raw.get("params"))}


def load_stopwords(path=None) -> frozenset:
    """The stopword list of config/chunk_topic_terms.json (function words only; data, not code). Empty on any problem."""
    try:
        lst = json.loads(Path(path or STOPWORDS_PATH).read_text(encoding="utf-8")).get("stopwords") or []
    except (OSError, ValueError, AttributeError):
        return frozenset()
    return frozenset(str(w).strip().lower() for w in lst if isinstance(w, str) and w.strip())


# ------------------------------------------------------------------------------------------------------------- taxonomy
class Taxonomy:
    """Nodes per level from registry entries. Node = {key, level, label, parent, domain, text}. Domain keys are the
    casefolded field names ("domain:<field>"); topics and subtopics keep their registry ids. Registry order is kept."""

    def __init__(self, entries):
        es = [e for e in (entries or []) if isinstance(e, dict) and e.get("id")]
        ids = {str(e["id"]) for e in es}
        self.nodes: dict = {}
        self.order: list = []
        self.children: dict = {}
        topics = [e for e in es if not e.get("parent_topic_key") or str(e["parent_topic_key"]) not in ids]
        subs = [e for e in es if e.get("parent_topic_key") and str(e["parent_topic_key"]) in ids]
        topic_ids = {str(e["id"]) for e in topics}

        def text_of(e):
            return " ".join(str(e.get(k) or "") for k in ("label", "description", "anchor_text"))
        for e in topics:
            field = str(e.get("field") or "").strip()
            if not field:
                continue
            dkey = "domain:" + field.casefold()
            if dkey not in self.nodes:
                self._add({"key": dkey, "level": "domain", "label": field, "parent": None, "domain": dkey, "text": field})
            self._add({"key": str(e["id"]), "level": "topic", "label": str(e.get("label") or e["id"]), "parent": dkey,
                       "domain": dkey, "text": text_of(e)})
        for e in subs:
            par = str(e["parent_topic_key"])
            if par not in topic_ids or par not in self.nodes:  # depth <= 2; a subtopic of a subtopic is ignored
                continue
            self._add({"key": str(e["id"]), "level": "subtopic", "label": str(e.get("label") or e["id"]), "parent": par,
                       "domain": self.nodes[par]["domain"], "text": text_of(e)})
    def _add(self, n):
        self.nodes[n["key"]] = n
        self.order.append(n["key"])
        self.children.setdefault(n["parent"], []).append(n["key"])

    def _desc(self, key):
        out, stack = [], list(self.children.get(key, ()))
        while stack:
            k = stack.pop(0)
            out.append(k)
            stack.extend(self.children.get(k, ()))
        return out

    def level_keys(self, level):
        return [k for k in self.order if self.nodes[k]["level"] == level]

    def path(self, key) -> dict:
        out = {lv: None for lv in LEVELS}
        k = key
        while k is not None:
            n = self.nodes[k]
            out[n["level"]] = k
            k = n["parent"]
        return out

    def descendants_and_self(self, key) -> set:
        return {key, *self._desc(key)}


# --------------------------------------------------------------------------------------------------- corpus statistics
class Model:
    """Fitted matcher: taxonomy + corpus IDF + null distributions per candidate set and channel.

    corpus: iterable of chunk texts (the sampled chunks of the documents to classify). embed: optional callable
    list[str] -> list[list[float]] (semantic channel); None = lexical only."""

    def __init__(self, entries, corpus, params: dict | None = None, stopwords=frozenset(), embed=None):
        self.p = params_from(params)
        self.tax = Taxonomy(entries)
        self.stop = frozenset(stopwords or ())
        self.embed = embed
        self._emb: dict = {}
        self._vec: dict = {}
        chunks = [str(c) for c in (corpus or ()) if str(c or "").strip()]
        self.n_chunks = len(chunks)
        df: Counter = Counter()
        uni: Counter = Counter()
        for c in chunks:
            toks = self.tokens(c)
            df.update(set(toks))
            uni.update(toks)
        n = max(1, self.n_chunks)
        self.idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.idf_unseen = math.log(1.0 + (n + 0.5) / 0.5)
        self._vocab = sorted(uni)
        self._cum = []
        acc = 0
        for t in self._vocab:
            acc += uni[t]
            self._cum.append(acc)
        self.node_len = {k: max(1, len(self.tokens(nd["text"]))) for k, nd in self.tax.nodes.items()}
        self.node_lex = {k: self._lex(nd["text"]) for k, nd in self.tax.nodes.items()}
        self.scoring = {k: [k, *self.tax._desc(k)] for k in self.tax.order}  # a node scores as max(itself, descendants)
        self._fit_cache: dict = {}
        self.node_sem = {}
        if self.embed is not None and self.tax.order:
            vs = self.embed([self.tax.nodes[k]["text"] for k in self.tax.order])
            self.node_sem = {k: _unit(v) for k, v in zip(self.tax.order, vs)}
        step = max(1, len(chunks) // self.p["null_chunks"]) if chunks else 1
        self._null_sample = chunks[::step][: self.p["null_chunks"]]
        if self.embed is not None:
            self.prime(self._null_sample)
        self.nulls: dict = {}
        for parent in [None] + list(self.tax.order):
            cand = self.candidates(parent)
            if cand:
                self.nulls[parent] = self._null_for(parent, cand)
        self.signature = self._signature()

    # -- text
    def tokens(self, text) -> list:
        m = self.p["min_token_len"]
        return [t for t in _TOKEN.findall(str(text or "").lower()) if len(t) >= m and not t.isdigit() and t not in self.stop]

    def _lex(self, text) -> dict:
        tf = Counter(self.tokens(text))
        v = {t: (1.0 + math.log(f)) * self.idf.get(t, self.idf_unseen) for t, f in tf.items()}
        return _unit_sparse(v)

    def lex_vec(self, text) -> dict:
        h = _h(text)
        v = self._vec.get(h)
        if v is None:
            v = self._vec[h] = self._lex(text)
        return v

    def prime(self, texts):
        """Embed and cache texts in one batch (semantic channel only; no-op without embed)."""
        if self.embed is None:
            return
        todo = []
        seen = set()
        for t in texts:
            h = _h(t)
            if h not in self._emb and h not in seen:
                seen.add(h)
                todo.append((h, str(t)))
        if todo:
            for (h, _t), v in zip(todo, self.embed([t for _h2, t in todo])):
                self._emb[h] = _unit(v)

    def sem_vec(self, text):
        if self.embed is None:
            return None
        h = _h(text)
        if h not in self._emb:
            self.prime([text])
        return self._emb.get(h)

    # -- candidate sets and null
    def candidates(self, parent) -> list:
        """Candidate node keys at the level below `parent` (None = all domains)."""
        if parent is None:
            return self.tax.level_keys("domain")
        return list(self.tax.children.get(parent, ()))

    def _pseudo(self, rng, length) -> str:
        if not self._vocab:
            return ""
        total = self._cum[-1]
        return " ".join(self._vocab[bisect.bisect_right(self._cum, rng.randrange(total))] for _ in range(length))

    def _null_for(self, parent, cand) -> dict:
        seed = int(hashlib.sha256(f"{self.p['seed']}:{parent}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        lex_vals, sem_vals = [], []
        for _r in range(self.p["null_replicates"]):
            texts = [self._pseudo(rng, self.node_len[s]) for k in cand for s in self.scoring[k]]  # mirrors the max
            pl = [self._lex(t) for t in texts]
            ps = []
            if self.embed is not None:
                ps = [_unit(v) for v in self.embed(texts)]
            for c in self._null_sample:
                cv = self.lex_vec(c)
                lex_vals.append(max(_dot_sparse(cv, v) for v in pl))
                if ps:
                    sv = self.sem_vec(c)
                    sem_vals.append(max(_dot(sv, v) for v in ps))
        return {"lex": sorted(lex_vals), "sem": sorted(sem_vals) if sem_vals else None}

    def floor(self, parent, channel="lex") -> float | None:
        """The raw score at fit_quantile of the null (reported in provenance and the kit)."""
        nl = (self.nulls.get(parent) or {}).get(channel)
        if not nl:
            return None
        i = min(len(nl) - 1, max(0, math.ceil(self.p["fit_quantile"] * len(nl)) - 1))
        return nl[i]

    # -- scoring
    def fits(self, text, parent) -> dict:
        """{node key: fit percentile in [0,1]} for the candidates below `parent`."""
        ck = (_h(text), parent)
        hit = self._fit_cache.get(ck)
        if hit is not None:
            return hit
        cand = self.candidates(parent)
        null = self.nulls.get(parent)
        if not cand or not null or not null["lex"]:
            return {}
        cv = self.lex_vec(text)
        sv = self.sem_vec(text) if null.get("sem") else None
        w = self.p["semantic_weight"]
        out = {}
        for k in cand:
            pl = _pct(null["lex"], max(_dot_sparse(cv, self.node_lex[s]) for s in self.scoring[k]))
            if sv is not None and all(s in self.node_sem for s in self.scoring[k]):
                ps = _pct(null["sem"], max(_dot(sv, self.node_sem[s]) for s in self.scoring[k]))
                out[k] = (1.0 - w) * pl + w * ps
            else:
                out[k] = pl
        if len(self._fit_cache) < 500000:  # work bound on the cache, not a decision value
            self._fit_cache[ck] = out
        return out

    def level_stats(self, chunks, parent, p: dict | None = None) -> dict:
        """Vote statistics of a document's chunks over the candidates below `parent` (p: threshold overrides)."""
        cand = self.candidates(parent)
        q = (p or self.p)["fit_quantile"]
        votes, fsum, n = Counter(), Counter(), 0
        for c in chunks:
            n += 1
            f = self.fits(c, parent)
            if not f:
                continue
            k = max(cand, key=lambda x: (round(f.get(x, 0.0), 12), -cand.index(x)))
            if f[k] >= q:
                votes[k] += 1
                fsum[k] += f[k]
        ranked = sorted(votes, key=lambda x: (-votes[x], -round(fsum[x], 9), cand.index(x)))
        tot = sum(votes.values())
        v1 = votes[ranked[0]] if ranked else 0
        v2 = votes[ranked[1]] if len(ranked) > 1 else 0
        ent = 0.0
        if tot and len(votes) > 1:
            ent = -sum((v / tot) * math.log(v / tot) for v in votes.values()) / math.log(len(votes))
        return {"n": n, "top": ranked[0] if ranked else None, "votes": dict(votes), "fit_share": _r(tot / n if n else 0.0),
                "top_share": _r(v1 / n if n else 0.0), "agree": _r(v1 / tot if tot else 0.0),
                "margin": _r((v1 - v2) / n if n else 0.0), "entropy": _r(ent),
                "top_mean_fit": _r(fsum[ranked[0]] / v1 if v1 else 0.0)}

    def accept(self, st, p: dict | None = None) -> str | None:
        """None = accepted, else the reason code."""
        p = p or self.p
        if st["top"] is None or st["fit_share"] < p["min_top_share"]:
            return "NO_FIT"
        if st["top_share"] < p["min_top_share"]:
            return "LOW_SUPPORT"
        if st["agree"] < p["min_agree"]:
            return "SPLIT"
        return None

    def classify(self, chunks, overrides: dict | None = None) -> dict:
        """{level, key, path, reason, stats}. level None = abstained (reason set). Top-down back-off. overrides: the
        threshold parameters only (fit_quantile, min_top_share, min_agree), e.g. for the kit's grid; the null is unchanged."""
        p = self.p
        if overrides:
            bad = set(overrides) - set(_RANGES)
            if bad:
                raise ConfigError(f"taxo_hier: override not allowed: {sorted(bad)}")
            p = params_from({**self.p, **overrides})
        chunks = [str(c) for c in (chunks or ()) if str(c or "").strip()]
        res = {"level": None, "key": None, "path": {lv: None for lv in LEVELS}, "reason": None, "stats": {}}
        if not chunks:
            res["reason"] = "NO_TEXT"
            return res
        if not self.tax.level_keys("domain"):
            res["reason"] = "NO_FIT"
            return res
        if self.embed is not None:
            self.prime(chunks)
        parent = None
        for lv in LEVELS:
            if not self.candidates(parent):
                res["reason"] = "NO_CHILD_NODES" if parent is not None else "NO_FIT"
                break
            st = self.level_stats(chunks, parent, p)
            res["stats"][lv] = st
            why = self.accept(st, p)
            if why is not None:
                res["reason"] = why
                break
            res["level"], res["key"] = lv, st["top"]
            parent = st["top"]
        if res["key"] is not None:
            res["path"] = self.tax.path(res["key"])
        return res

    def provenance(self) -> dict:
        return {"assigned_via": "classifier", "classifier_name": CLASSIFIER_NAME,
                "classifier_version": f"{CLASSIFIER_VERSION}:{self.signature}", "score_kind": "fit_pct"}

    def _signature(self) -> str:
        canon = json.dumps({"p": self.p, "nodes": [[k, self.tax.nodes[k]["text"]] for k in self.tax.order],
                            "n": self.n_chunks, "sem": self.embed is not None}, sort_keys=True)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def summarize(results) -> dict:
    """Aggregate counts only (shadow-mode log line and kit): documents per deepest level and per reason."""
    by_level, by_reason = Counter(), Counter()
    n = 0
    for r in results:
        n += 1
        by_level[r.get("level") or "none"] += 1
        if r.get("reason"):
            by_reason[r["reason"]] += 1
    return {"documents": n, "deepest_level": dict(by_level), "reasons": dict(by_reason),
            "any_domain": n - by_level.get("none", 0)}


# ----------------------------------------------------------------------------------------------------------- vector math
def _h(text) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _unit_sparse(v: dict) -> dict:
    s = math.sqrt(sum(x * x for x in v.values()))
    return {t: x / s for t, x in v.items()} if s > 0 else {}


def _unit(v):
    v = [float(x) for x in (v or ())]
    s = math.sqrt(sum(x * x for x in v))
    return [x / s for x in v] if s > 0 else v


def _dot_sparse(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def _dot(a, b) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _pct(sorted_null, x) -> float:
    """Share of null values strictly below x."""
    return bisect.bisect_left(sorted_null, x) / len(sorted_null) if sorted_null else 0.0


def _r(x) -> float:
    return round(float(x), 4)
