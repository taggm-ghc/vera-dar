"""TAXO-INDUCE-1107 phase 1: chunk-level topic assignment (owner 2026-11-07: "each chunk should be re-analyzed to flesh out
association with sub-topic, topics, and domains"; "ingested chunks are associated with subtopics").

Pure and deterministic; no network, no writes. Each chunk (a vera_vjay.evidence_spans text) is scored with the SAME BM25 as
vera.micro_topic.Registry (anchor texts of the ACTIVE registry rows, IDF from the anchors only, chunk tokens as the query), through
an inverted index so a whole corpus is cheap to score. The best entry is a leaf (a subtopic when one exists, else a topic); the
topic and domain follow from the registry tree (parent_topic_key, field). A chunk is assigned only at or above the BM25 acceptance
value; a document's primary key is the key with the most chunk votes (ties: higher summed score, then registry order).

Phase 1 (before migration 020): NO applied table can hold chunk links (017 and 020 are not applied; sources and evidence_spans are
not link tables), so assignments are computed in memory for the Corpus Description page and for the induction report only.
Phase 2 (after 020): the 020 backfill writes tax_chunk_subtopic / tax_chunk_topic with the provenance returned by provenance().

CALIBRATION (2026-11-07): PROVISIONAL. The acceptance value is the existing micro_topic prior (count_min_score.anchor_bm25.scholarly,
2.0, never calibrated) and min_chunk_votes 1; no owner count output exists yet (work/CORPUS-COUNTS-OWNER-1107 holds scripts only).
Calibrate from the owner-run induction report (current vs projected matched share at several values) and a spot check.

TAXO-MATCH-FIX-1108 (classifier version 2): the version-1 rule matched almost any scholarly text (IDF from the anchors only, so one
generic word such as "data" reached the acceptance value; no stop list; no lead over the runner-up; one chunk of 16 assigned the
document). Version 2 adds the match rules of config/chunk_topic_terms.json (load_rules): generic terms (stopwords + generic
scholarly words, kept as data) never score; a chunk must share at least min_distinct_terms distinct non-generic terms with the
winning anchor and lead the best entry of any OTHER subject area by min_lead_ratio; a document needs votes from at least
min_vote_share of its sampled chunks. All values PROVISIONAL (calibration note in that file). ChunkIndex(entries) without rules
keeps the version-1 scoring (pinned equal to vera.micro_topic by a test)."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from vera import micro_topic

CLASSIFIER_NAME = "chunk_bm25"
CLASSIFIER_VERSION = "2"  # TAXO-MATCH-FIX-1108: generic terms, distinct-term floor, lead ratio, vote share
RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "chunk_topic_terms.json"
_RULE_RANGES = {"min_token_len": (int, 1, 20), "min_distinct_terms": (int, 1, 20), "min_lead_ratio": (float, 1.0, 100.0),
                "min_vote_share": (float, 0.0, 1.0)}


def load_rules(path=None) -> dict:
    """Match rules from config/chunk_topic_terms.json: {generic: frozenset, min_token_len, min_distinct_terms, min_lead_ratio,
    min_vote_share}. Raises OSError (missing file) or ValueError (malformed JSON or out-of-range value) (fail closed: callers show no subject areas
    rather than fall back to the permissive version-1 rule)."""
    raw = json.loads(Path(path or RULES_PATH).read_text(encoding="utf-8"))
    out: dict = {}
    for k, (typ, lo, hi) in _RULE_RANGES.items():
        v = raw.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (typ is int and not isinstance(v, int)) or not lo <= v <= hi:
            raise ValueError(f"chunk_topic_terms: bad {k}")
        out[k] = typ(v)
    words = []
    for k in ("stopwords", "generic_scholarly_terms"):
        lst = raw.get(k)
        if not isinstance(lst, list) or not lst or not all(isinstance(w, str) and w.strip() for w in lst):
            raise ValueError(f"chunk_topic_terms: bad {k}")
        words += [w.strip().lower() for w in lst]
    out["generic"] = frozenset(words)
    return out


_K1, _B = 1.2, 0.75  # identical to vera.micro_topic (pinned by a test)


class ChunkIndex:
    """Inverted BM25 index over registry entries (dicts with id, anchor_text, field, parent_topic_key). Scores equal
    micro_topic.Registry.scores for every entry; best() breaks ties exactly like micro_topic.Registry.best."""

    def __init__(self, entries, rules: dict | None = None):
        self.rules = rules
        self.entries = [e for e in (entries or []) if isinstance(e, dict) and e.get("id")]
        self.by_id = {str(e["id"]): e for e in self.entries}
        self.domain = [str(e.get("field") or "").casefold() for e in self.entries]
        tfs = [Counter(micro_topic.tokens(e.get("anchor_text"))) for e in self.entries]
        self.anchor_terms = [frozenset(t for t in c if self.content_term(t)) for c in tfs]
        lens = [sum(c.values()) for c in tfs]
        n = len(self.entries)
        avg = (sum(lens) / n) if n else 0.0
        df: Counter = Counter()
        for c in tfs:
            df.update(c.keys())
        idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.postings: dict = {}
        for i, (tf, dl) in enumerate(zip(tfs, lens)):
            for t, f in tf.items():
                w = idf[t] * f * (_K1 + 1) / (f + _K1 * (1 - _B + _B * dl / (avg or 1.0)))
                self.postings.setdefault(t, []).append((i, w))

    def content_term(self, t: str) -> bool:
        """True for a term that may count toward a match (always True without rules: version-1 scoring)."""
        r = self.rules
        return r is None or (len(t) >= r["min_token_len"] and not t.isdigit() and t not in r["generic"])

    def query_terms(self, text) -> set:
        return {t for t in micro_topic.tokens(text) if self.content_term(t)}

    def scores(self, text) -> list:
        acc = [0.0] * len(self.entries)
        for t in self.query_terms(text):
            for i, w in self.postings.get(t, ()):
                acc[i] += w
        return acc

    def best(self, text):
        """(score, entry) of the best entry, or None when there are no entries."""
        if not self.entries:
            return None
        acc = self.scores(text)
        i = max(range(len(acc)), key=lambda j: (round(acc[j], 12), -j))
        return acc[i], self.entries[i]


def tree_of(entries) -> dict:
    return {str(e["id"]): {"parent": e.get("parent_topic_key"), "field": e.get("field")} for e in entries or [] if e.get("id")}


def root_of(tree: dict, key):
    """Top-level topic key of a key (depth <= 2, active chain only), or None for an unknown key or a broken chain."""
    seen, k = set(), key
    while k is not None and k in tree and k not in seen and len(seen) < 3:
        seen.add(k)
        p = tree[k]["parent"]
        if p is None:
            return k
        k = p
    return None


def match_detail(index: ChunkIndex, text) -> dict | None:
    """Diagnostics of one chunk (pure): best key and score, distinct non-generic terms shared with the winning anchor, and the
    best score of an entry in another subject area (0.0 when there is none). None for empty text or no registry."""
    if not str(text or "").strip() or not index.entries:
        return None
    q = index.query_terms(text)
    acc = index.scores(text)
    i = max(range(len(acc)), key=lambda j: (round(acc[j], 12), -j))
    other = [acc[j] for j in range(len(acc)) if index.domain[j] != index.domain[i]]
    return {"key": str(index.entries[i]["id"]), "score": acc[i], "shared": sorted(q & index.anchor_terms[i]),
            "runner_up_other_domain": max(other) if other else 0.0}


def assign_chunk(index: ChunkIndex, text, threshold: float):
    """(key, score) of one chunk, or None (below the acceptance value, empty text, or no registry). With rules (version 2)
    also None when fewer than min_distinct_terms non-generic terms are shared with the winning anchor, or when the score does
    not lead the best other-subject-area entry by min_lead_ratio."""
    d = match_detail(index, text)
    if d is None or d["score"] < threshold or d["score"] <= 0:
        return None
    r = index.rules
    if r is not None and (len(d["shared"]) < r["min_distinct_terms"]
                          or d["score"] < r["min_lead_ratio"] * d["runner_up_other_domain"]):
        return None
    return d["key"], round(d["score"], 4)


def hierarchy(tree: dict, key) -> dict | None:
    """{subtopic, topic, domain} of an assigned leaf key (subtopic None when the key is a top-level topic)."""
    root = root_of(tree, key)
    if root is None:
        return None
    return {"subtopic": key if key != root else None, "topic": root, "domain": tree[root].get("field")}


def rules_signature(rules: dict | None) -> str:
    """Short stable digest of the match rules (values and generic list), so a changed config yields a new classifier_version."""
    if rules is None:
        return "none"
    canon = json.dumps({k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in sorted(rules.items())}, sort_keys=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


# Migration 020 (applied 2026-11-08) link-row CHECKs, mirrored here so a backfill row is checked before any write:
# *_cname_chk, *_cver_chk, *_skind_chk, *_via_chk, *_prov_chk (classifier_* required for assigned_via backfill/classifier),
# *_cpair_chk, *_spair_chk (score and score_kind both set or both NULL), *_score_chk (finite).
_CNAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,63}$")
_CVER_RE = re.compile(r"^[ -~]{1,128}$")
_SKIND_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


def provenance(registry_version: str, threshold: float, rules: dict | None = None) -> dict:
    """Provenance carried by every assignment; phase 2 writes it into the 020 link rows (tax_chunk_subtopic / tax_chunk_topic:
    assigned_via, classifier_name, classifier_version, score_kind; score is the assign_chunk score). classifier_version =
    CLASSIFIER_VERSION:registry_version:t<threshold>[:r<rules signature>], printable ASCII, at most 128 characters."""
    reg = "".join(ch for ch in str(registry_version) if " " <= ch <= "~")
    ver = f"{CLASSIFIER_VERSION}:{reg}:t{threshold:g}" + (f":r{rules_signature(rules)}" if rules is not None else "")
    if len(ver) > 128:  # keep the rules signature and threshold; shorten the registry part
        tail = ver[len(CLASSIFIER_VERSION) + 1 + len(reg):]
        ver = f"{CLASSIFIER_VERSION}:{reg[:128 - len(CLASSIFIER_VERSION) - 1 - len(tail)]}{tail}"
    return {"assigned_via": "backfill", "classifier_name": CLASSIFIER_NAME, "classifier_version": ver, "score_kind": "bm25"}


def link_row_problems(prov: dict, score) -> list:
    """Names of the 020 CHECK constraints a link row with this provenance and score would violate ([] = passes). Pure."""
    p = []
    via, name, ver, kind = (prov.get(k) for k in ("assigned_via", "classifier_name", "classifier_version", "score_kind"))
    if via not in ("registry_sync", "curator", "classifier", "backfill"):
        p.append("via_chk")
    if name is not None and not (isinstance(name, str) and _CNAME_RE.match(name)):
        p.append("cname_chk")
    if ver is not None and not (isinstance(ver, str) and _CVER_RE.match(ver)):
        p.append("cver_chk")
    if kind is not None and not (isinstance(kind, str) and _SKIND_RE.match(kind)):
        p.append("skind_chk")
    if via not in ("registry_sync", "curator") and (name is None or ver is None):
        p.append("prov_chk")
    if (name is None) != (ver is None):
        p.append("cpair_chk")
    if (score is None) != (kind is None):
        p.append("spair_chk")
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
        p.append("score_chk")
    return p


def required_votes(n_chunks: int, min_votes: int = 1, min_vote_share: float = 0.0) -> int:
    """Votes the winning key needs: at least min_votes and at least min_vote_share of the sampled chunks (rounded up)."""
    return max(1, int(min_votes), math.ceil(float(min_vote_share) * max(0, int(n_chunks)) - 1e-9))


def document_vote(chunk_results, min_votes: int = 1, min_vote_share: float = 0.0):
    """Primary key of a document from its chunk results [(key, score) or None], or None. Most votes, then summed score,
    then the first chunk that voted for the key. Deterministic. The winner needs required_votes(len(chunk_results), ...)."""
    chunk_results = list(chunk_results or ())
    votes, total, first = Counter(), Counter(), {}
    for i, r in enumerate(chunk_results or ()):
        if r is None:
            continue
        k, s = r
        votes[k] += 1
        total[k] += s
        first.setdefault(k, i)
    if not votes:
        return None
    k = max(votes, key=lambda x: (votes[x], round(total[x], 9), -first[x]))
    return k if votes[k] >= required_votes(len(chunk_results), min_votes, min_vote_share) else None


def document_primary(index: ChunkIndex, chunks, threshold: float, min_votes: int = 1, min_vote_share: float | None = None):
    """min_vote_share None = the index rules' value (0.0 without rules)."""
    share = ((index.rules or {}).get("min_vote_share", 0.0)) if min_vote_share is None else min_vote_share
    return document_vote([assign_chunk(index, c, threshold) for c in chunks or ()], min_votes, share)
