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
Calibrate from the owner-run induction report (current vs projected matched share at several values) and a spot check."""
from __future__ import annotations

import math
from collections import Counter

from vera import micro_topic

CLASSIFIER_NAME = "chunk_bm25"
CLASSIFIER_VERSION = "1"
_K1, _B = 1.2, 0.75  # identical to vera.micro_topic (pinned by a test)


class ChunkIndex:
    """Inverted BM25 index over registry entries (dicts with id, anchor_text, field, parent_topic_key). Scores equal
    micro_topic.Registry.scores for every entry; best() breaks ties exactly like micro_topic.Registry.best."""

    def __init__(self, entries):
        self.entries = [e for e in (entries or []) if isinstance(e, dict) and e.get("id")]
        self.by_id = {str(e["id"]): e for e in self.entries}
        tfs = [Counter(micro_topic.tokens(e.get("anchor_text"))) for e in self.entries]
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

    def scores(self, text) -> list:
        acc = [0.0] * len(self.entries)
        for t in set(micro_topic.tokens(text)):
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


def assign_chunk(index: ChunkIndex, text, threshold: float):
    """(key, score) of one chunk, or None (below the acceptance value, empty text, or no registry)."""
    if not str(text or "").strip():
        return None
    b = index.best(text)
    if b is None or b[0] < threshold:
        return None
    return str(b[1]["id"]), round(b[0], 4)


def hierarchy(tree: dict, key) -> dict | None:
    """{subtopic, topic, domain} of an assigned leaf key (subtopic None when the key is a top-level topic)."""
    root = root_of(tree, key)
    if root is None:
        return None
    return {"subtopic": key if key != root else None, "topic": root, "domain": tree[root].get("field")}


def provenance(registry_version: str, threshold: float) -> dict:
    """Provenance carried by every assignment (phase 2 writes it into the 020 link rows)."""
    return {"classifier_name": CLASSIFIER_NAME, "classifier_version": f"{CLASSIFIER_VERSION}:{registry_version}:t{threshold:g}",
            "score_kind": "bm25"}


def document_vote(chunk_results, min_votes: int = 1):
    """Primary key of a document from its chunk results [(key, score) or None], or None. Most votes, then summed score,
    then the first chunk that voted for the key. Deterministic."""
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
    return k if votes[k] >= max(1, int(min_votes)) else None


def document_primary(index: ChunkIndex, chunks, threshold: float, min_votes: int = 1):
    return document_vote([assign_chunk(index, c, threshold) for c in chunks or ()], min_votes)
