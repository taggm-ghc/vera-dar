"""VERA-DAR L4: in-memory stored-document corpus with domain identification, chunking and BM25 + domain-aware fusion.

Stdlib only; no DB, network, LLM or environment reads. Config is the whole config/domain_routing.json dict (`cfg`);
nothing is hard-coded: chunk sizes, BM25 constants, branch caps and ids come from cfg["corpus"] / cfg["identification"].

Rules
  * Full documents only: answer_contract.acquisition_state(doc, path) must be "full_document_acquired", gated on
    STORED provenance fields (stored_text_kind + admission_passed). Anything else is refused, listed in .skipped,
    and never indexed.
  * Chunking rule (deterministic): a chunk starts at a non-whitespace character and covers text[start:end] with NO
    stripping (end is placed just before a whitespace character, or at len(text)). Window = chunk_chars; when the
    window does not reach the end of the text, end is moved back to the last whitespace inside the window. The next
    start is end - chunk_overlap, moved forward to the next word start. A final chunk shorter than min_chunk_chars
    is merged into its predecessor (predecessor extended to the end of text); a document with a single short chunk
    keeps it. span_id = f"{source_key}:{start}-{end}". All ids are strings; domain ids are the registry ints.
  * Domain labels are a retrieval aid only: the unrestricted BM25 branch over ALL chunks (classified or not) always
    runs; domain branches only add rank; retrieval returns PROPOSALS (status "proposed"), never admitted evidence.
  * The gate always uses corpus.contract_config_path (null -> the answer_contract default); a per-document
    contract path is never honoured.
  * restore() rebuilds documents, domains and associations from a snapshot WITHOUT identification. Pending clusters
    (candidate domains not yet promoted) are NOT restored. The return value carries
    "pending_not_restored": True. A snapshot "chunking" dict that differs from the corpus config yields status
    "chunking_mismatch" and nothing is indexed. Because load_snapshot never carries "chunking", consistency is also
    verified from the data: every snapshot chunk_assoc row of a restorable document must be a span of that document
    under the current chunker; otherwise status "chunking_mismatch" with "unmatched": n, and nothing is indexed.
  * as_sources() may return several chunks of one document (distinct span entries). The source url is the plain
    stored document url (no fragment); offsets travel in a separate "dar_span" key. grounded_ask.retrieve_all
    de-duplicates by DOI/URL, so at most ONE representative chunk (the best ranked) per stored document reaches the
    prompt; that is the explicit decision. Snippets are truncated at a word boundary to corpus.snippet_chars; full chunk text stays in retrieve()
    candidates and to_evidence_corpus spans.
"""
from __future__ import annotations

import copy
import logging
import math
import re
from array import array
from bisect import bisect_left
from collections import Counter
from collections.abc import Mapping
from typing import Callable

from vera import domain_registry, domain_routing
from vera.answer_contract import ACQ_ACQUIRED, CONFIG_PATH as CONTRACT_CONFIG_PATH, acquisition_state, full_document_kinds
from vera.domain_identify import DomainIdentifier, tokenize

LIMITATION = "stored corpus only; proposals, not admitted evidence"


def _year(doc: dict):
    y = doc.get("year")
    if isinstance(y, int) and not isinstance(y, bool):
        return y
    m = re.match(r"\s*(\d{4})", str(doc.get("published") or ""))
    return int(m.group(1)) if m else None


log = logging.getLogger(__name__)
from vera.domain_identify import SECONDARY  # noqa: E402  asserted_by suffix of secondary document labels

# NOCAP-1109 compact index. A chunk is stored as offsets into its document's text (no copied text, no per-chunk term
# map); term frequencies live in shared postings: term -> array('I') of (chunk index << _CNT_BITS | count), with a
# count of _CNT_MAX or more kept exactly in a side map. Every value a reader sees equals the former stored one.
_CNT_BITS = 8
_CNT_MAX = (1 << _CNT_BITS) - 1
_IDX_LIMIT = 1 << (32 - _CNT_BITS)
_CHUNK_KEYS = ("span_id", "source_key", "start", "end", "text", "tf", "len")


def light_stem(t: str) -> str:
    """RETR-MECH-1109: light deterministic suffix stemmer for BM25 terms ("agents"/"agent", "forecasting"/"forecast").
    Plural (-ies, -sses, -s) and -ing/-ed only. Conservative: tokens shorter than 5 characters (so "news", "lens", "gans"
    stay), digits, and words ending in -ss/-us/-is are unchanged; a stem is never shorter than 4
    characters; -ed is stripped only when what remains does not end in "it" ("united" stays). Generic: no word lists."""
    n = len(t)
    if n < 5 or not t.isalpha():
        return t
    if t.endswith("ies") and n >= 6:
        return t[:-3] + "y"
    if t.endswith("sses"):
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    if t.endswith("ing") and n - 3 >= 4:
        return t[:-3]
    if t.endswith("ed") and n - 2 >= 4 and not t[:-2].endswith("it"):
        return t[:-2]
    return t


def _stem_tokens(toks: list[str], on: bool) -> list[str]:
    return [light_stem(t) for t in toks] if on else toks


class _Chunk(Mapping):
    """Read-only chunk record with the former dict's keys. "text" is the document slice text[start:end] (the same
    string the former record copied) and "tf" is Counter(tokenize(text)) recomputed on demand (same counts, same
    first-occurrence key order). Immutable and shared between corpus versions."""
    __slots__ = ("doc", "start", "end", "len", "idx", "_icfg", "_stem")

    def __init__(self, doc: dict, start: int, end: int, n_tokens: int, idx: int, icfg: dict, stem: bool = False):
        self.doc, self.start, self.end, self.len, self.idx, self._icfg = doc, start, end, n_tokens, idx, icfg
        self._stem = stem

    @property
    def source_key(self) -> str:
        return self.doc["source_key"]

    @property
    def span_id(self) -> str:
        return f"{self.doc['source_key']}:{self.start}-{self.end}"

    @property
    def text(self) -> str:
        return self.doc["text"][self.start:self.end]

    def __getitem__(self, k):
        if k == "start":
            return self.start
        if k == "end":
            return self.end
        if k == "len":
            return self.len
        if k == "source_key":
            return self.doc["source_key"]
        if k == "span_id":
            return self.span_id
        if k == "text":
            return self.text
        if k == "tf":
            return Counter(_stem_tokens(tokenize(self.text, self._icfg), self._stem))
        raise KeyError(k)

    def __iter__(self):
        return iter(_CHUNK_KEYS)

    def __len__(self):
        return len(_CHUNK_KEYS)

    def __contains__(self, k):
        return k in _CHUNK_KEYS

    def __eq__(self, other):
        if isinstance(other, _Chunk):
            return self is other or (self.doc is other.doc and self.start == other.start and self.end == other.end)
        return Mapping.__eq__(self, other)

    __hash__ = None  # like the former dict record

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self


class _SpanIndex:
    """span_id -> chunk for one corpus version (the former dict), resolved through the per-document chunk lists
    instead of a stored string key per chunk. `mask` (optional) restricts it to a subset (the evidence-only view)."""
    __slots__ = ("_dc", "_mask")

    def __init__(self, doc_chunks: dict, mask=None):
        self._dc, self._mask = doc_chunks, mask

    def get(self, span_id, default=None):
        if not isinstance(span_id, str):
            return default
        key, sep, rng = span_id.rpartition(":")
        a, dash, b = rng.partition("-")
        if not sep or not dash or not a.isdigit() or not b.isdigit():
            return default
        lst = self._dc.get(key)
        if not lst:
            return default
        st, en = int(a), int(b)
        i = bisect_left(lst, st, key=lambda c: c.start)
        if i < len(lst) and lst[i].start == st and lst[i].end == en and str(st) == a and str(en) == b:
            c = lst[i]
            if self._mask is None or (c.idx < len(self._mask) and self._mask[c.idx]):
                return c
        return default

    def __getitem__(self, span_id):
        c = self.get(span_id)
        if c is None:
            raise KeyError(span_id)
        return c

    def __contains__(self, span_id):
        return self.get(span_id) is not None

    def __iter__(self):
        for lst in self._dc.values():
            for c in lst:
                if self._mask is None or (c.idx < len(self._mask) and self._mask[c.idx]):
                    yield c.span_id

    def __len__(self):
        return sum(1 for _ in self)

    def values(self):
        return [self[k] for k in self]


def doc_from_admitted(cand: dict, outcome: str) -> dict:
    """Ingest doc from an admit() candidate. stored_text_kind is the STORED provenance kind, never the candidate's own."""
    prov = cand.get("provenance") or {}
    return {"source_key": cand.get("canonical_url"), "db_source_id": cand.get("source_id"),
            "db_canonical_source_id": cand.get("canonical_source_id"), "url": cand.get("url") or cand.get("canonical_url"),
            "doi": cand.get("doi"), "title": cand.get("title"), "published": cand.get("published"),
            "authors": cand.get("authors"),
            "source_type": cand.get("source_type"), "retrieved_at": prov.get("retrieved_at") or cand.get("retrieved_at"),
            "text": cand.get("snippet") or "", "stored_text_kind": prov.get("stored_text_kind"),
            "admission_passed": outcome == "admitted"}


_DOI_IN_URL = re.compile(r"^https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/\S+)$", re.I)
# ARCH-FIX-1110 A: an arxiv.org abs/pdf/html URL (any version) and its 10.48550/arXiv.<id> DOI are one document.
_ARXIV_URL = re.compile(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf|html)/((?:\d{4}\.\d{4,5})|(?:[a-z\-]+(?:\.[a-z]{2})?/\d{7}))(?:v\d+)?(?:\.pdf)?$", re.I)
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+?)(?:v\d+)?$", re.I)


def doc_key_set(doi=None, url=None, isbn=None) -> set:
    """Normalised membership keys for one document (WED-ACQ): "doi:<bare lower-case DOI>" and "url:<https URL
    without fragment, trailing slash or case>". A doi.org / dx.doi.org URL contributes its DOI key instead of a URL
    key, so a record with only a doi.org URL matches a stored document that has the DOI (and the reverse)."""
    keys = set()
    d = str(doi or "").strip().lower()
    if d.startswith("doi:"):
        d = d[4:].strip()
    m = _DOI_IN_URL.match(d)
    d = m.group(1) if m else d
    ma = _ARXIV_DOI.match(d)
    if ma:
        d = "10.48550/arxiv." + ma.group(1)
    if d:
        keys.add("doi:" + d)
    u = str(url or "").strip().lower().split("#")[0].rstrip("/")
    if u:
        m = _DOI_IN_URL.match(u)
        ax = _ARXIV_URL.match(u)
        if m:
            keys.add("doi:" + _ARXIV_DOI.sub(r"10.48550/arxiv.\1", m.group(1).rstrip("/")))
        elif ax:
            keys.add("doi:10.48550/arxiv." + ax.group(1))
        else:
            keys.add("url:" + re.sub(r"^http://", "https://", u))
    # WED-ACQ2 (owner 2026-10-21, ISBN post-script): a validated ISBN-13 (hyphens removed) is a third membership key.
    # Only a value that passes the check digit counts; anything else contributes no key (never guessed).
    if isbn:
        from vera.acq_config import normalise_isbn
        n = normalise_isbn(isbn)
        if n:
            keys.add("isbn:" + n["isbn"])
    return keys


class DarCorpus:
    def __init__(self, cfg: dict | None = None, registry=None, clock: Callable[[], str] | None = None):
        self.cfg = domain_registry.load_config() if cfg is None else cfg
        self.ccfg = self.cfg.get("corpus") or {}
        self.registry = registry if registry is not None else domain_registry.load_registry(self.cfg)
        self.identifier = DomainIdentifier(self.registry, self.cfg.get("identification") or {})
        self._icfg = self.cfg.get("identification") or {}
        self._clock = clock
        self.registry.identify_question = lambda q: [d["domain_id"] for d in self.identifier.classify(q, "question")]
        self.registry.domain_terms = lambda did, n: self.identifier.profile_terms(did, n)  # content terms for external domain queries
        self.skipped: list[dict] = []
        self._rejected: set = set()  # keys the gate refused during apply_delta (re-read in an overlap window: skip quietly)
        self._docs: dict[str, dict] = {}
        self._chunks: list = []  # _Chunk records of THIS version (the evidence view: a subset)
        self._all: list = self._chunks  # every chunk of this version by index (shared by an evidence view)
        self._mask = None  # evidence view only: bytearray over chunk indexes (1 = in the view)
        self._doc_chunks: dict[str, list] = {}  # source_key -> its chunks in start order
        self._by_span = _SpanIndex(self._doc_chunks)
        self._post: dict = {}  # term -> array('I'); SHARED by every version forked from this one (append-only)
        self._post_big: dict = {}  # (term, chunk index) -> count, for counts >= _CNT_MAX (shared, append-only)
        self._df: Counter = Counter()
        self._total_len = 0
        self._reported: dict[str, set] = {}  # source_key -> corpus domain ids already reported in a plan

    # --- helpers ---------------------------------------------------------------------------------------------
    def _enabled(self) -> bool:
        return domain_registry.routing_enabled(self.cfg)

    def _chunk_spans(self, text: str) -> list[tuple[int, int]]:
        size, ov, mn = int(self.ccfg["chunk_chars"]), int(self.ccfg["chunk_overlap"]), int(self.ccfg["min_chunk_chars"])
        n, spans, pos = len(text), [], 0
        while True:
            m = re.compile(r"\S").search(text, pos)
            if not m:
                break
            start = m.start()
            end = n if start + size >= n else None
            if end is None:
                end = start + size
                ws = max((i for i in range(start + 1, end + 1) if text[i].isspace()), default=None)
                if ws is not None:
                    end = ws
            reached = end >= n
            while end > start + 1 and text[end - 1].isspace():
                end -= 1  # chunks never end in whitespace
            spans.append((start, end))
            if reached:
                break
            nxt = max(end - ov, start + 1)
            while nxt < n and not text[nxt - 1].isspace():
                nxt += 1  # snap forward to a word start
            if nxt >= end:
                nxt = end
            pos = nxt
        if len(spans) > 1 and spans[-1][1] - spans[-1][0] < mn:
            last = spans.pop()
            spans[-1] = (spans[-1][0], last[1])
        return [(s, e) for s, e in spans if text[s:e].strip()]

    def _stem_on(self) -> bool:
        return self.ccfg.get("bm25_stem") is True  # RETR-MECH-1109 (default ON in config/domain_routing.json)

    def _query_terms(self, question: str) -> list[str]:
        """BM25 query terms: generic idioms dropped (vera.search_query.drop_generic, as admission does; generic words are
        dropped by the caller, fulldoc_admission), then stemmed.
        Falls back to all terms when nothing would remain."""
        q = question
        toks = tokenize(q, self._icfg)
        if self.ccfg.get("query_drop_generic") is True:
            try:
                from vera.search_query import drop_generic
                q2 = drop_generic(q)
                t2 = tokenize(q2, self._icfg)
                if t2:
                    toks = t2
            except Exception:
                pass
        return sorted(set(_stem_tokens(toks, self._stem_on())))

    def _index_chunk(self, doc: dict, start: int, end: int):
        text = doc["text"][start:end]
        toks = _stem_tokens(tokenize(text, self._icfg), self._stem_on())
        tf = Counter(toks)
        idx = len(self._chunks)
        if idx >= _IDX_LIMIT or self._mask is not None:
            raise RuntimeError("compact index: chunk index out of range or ingest into an evidence view")
        rec = _Chunk(doc, start, end, len(toks), idx, self._icfg, self._stem_on())
        post, big = self._post, self._post_big
        for t, f in tf.items():  # postings first: a failure leaves the chunk list unchanged
            a = post.get(t)
            if a is None:
                a = post[t] = array("I")
            if f >= _CNT_MAX:
                big[(t, idx)] = f
                a.append(idx << _CNT_BITS | _CNT_MAX)
            else:
                a.append(idx << _CNT_BITS | f)
        self._chunks.append(rec)
        self._doc_chunks.setdefault(doc["source_key"], []).append(rec)
        self._df.update(tf.keys())
        self._total_len += rec.len
        return rec

    # --- versions (NOCAP-1109) -------------------------------------------------------------------------------
    def fork(self) -> "DarCorpus":
        """A new version that can take more documents while this one keeps serving unchanged. Shared, never copied:
        document records and texts, chunk records, and the postings (append-only; each version reads only chunk
        indexes below its own chunk count). Copied: the containers and the identification and registry state that
        ingest mutates. Two full copies of the index never exist."""
        new = copy.copy(self)
        new._fork_base = len(self._chunks)
        new._docs = dict(self._docs)
        new._chunks = list(self._chunks)
        new._all = new._chunks
        new._doc_chunks = dict(self._doc_chunks)
        new._by_span = _SpanIndex(new._doc_chunks)
        new._df = Counter(self._df)
        new._reported = {k: set(v) for k, v in self._reported.items()}
        new.skipped = list(self.skipped)
        new._rejected = set(self._rejected)
        memo = {id(d): d for d in self._docs.values()}
        memo[id(self.cfg)] = self.cfg
        memo[id(self._icfg)] = self._icfg
        memo[id(self._post)] = self._post
        memo[id(self._post_big)] = self._post_big
        reg = copy.deepcopy(self.registry, memo)
        memo[id(self.registry)] = reg
        ident = copy.deepcopy(self.identifier, memo)
        ident._dmemo = None
        new.registry, new.identifier = reg, ident
        reg.identify_question = lambda q: [d["domain_id"] for d in new.identifier.classify(q, "question")]
        reg.domain_terms = lambda did, n: new.identifier.profile_terms(did, n)
        return new

    def discard(self) -> None:
        """Undo a fork's postings appends (every entry at or above its base chunk index) so the shared postings again
        hold exactly the parent's entries. Called when a fork is abandoned; the parent never saw those entries."""
        base = getattr(self, "_fork_base", None)
        if base is None:
            return
        for a in list(self._post.values()):
            while a and (a[-1] >> _CNT_BITS) >= base:
                a.pop()
        for k in [k for k in list(self._post_big) if k[1] >= base]:
            self._post_big.pop(k, None)
        self._fork_base = None

    def evidence_view(self, keep) -> "DarCorpus":
        """A read-only view holding only the chunks for which keep(chunk) is true (the C1 evidence-only index): shares
        documents, registry, identifier and postings; its own chunk list, df and length totals are recomputed over the
        kept chunks exactly as the former filtered copy did."""
        ev = copy.copy(self)
        mask = bytearray(len(self._chunks))
        kept = []
        for c in self._chunks:
            if keep(c):
                mask[c.idx] = 1
                kept.append(c)
        ev._chunks = kept
        ev._all = self._chunks
        ev._mask = mask
        ev._by_span = _SpanIndex(self._doc_chunks, mask)
        df = Counter()
        n_all = len(self._chunks)
        for t, a in list(self._post.items()):
            k = 0
            for v in a:
                i = v >> _CNT_BITS
                if i < n_all and mask[i]:
                    k += 1
            if k:
                df[t] = k
        ev._df = df
        ev._total_len = sum(c.len for c in kept)
        return ev

    def _scores(self, q_terms: list[str]) -> dict:
        """chunk index -> BM25 score over this version's chunks (only chunks with a positive term frequency). Per
        chunk the terms are added in q_terms order with the same expression as before, so each score is the same
        float the former per-chunk loop produced."""
        n = len(self._chunks)
        if not n or not q_terms:
            return {}
        avg = (self._total_len / n) or 1.0
        k1, b = float(self.ccfg["bm25_k1"]), float(self.ccfg["bm25_b"])
        allc, mask, big = self._all, self._mask, self._post_big
        n_all = len(allc)
        sc: dict = {}
        for t in q_terms:
            a = self._post.get(t)
            if not a:
                continue
            df = self._df.get(t, 0)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for v in a:
                i = v >> _CNT_BITS
                if i >= n_all or (mask is not None and not mask[i]):
                    continue
                f = v & _CNT_MAX
                if f == _CNT_MAX:
                    f = big[(t, i)]
                sc[i] = sc.get(i, 0.0) + idf * f * (k1 + 1) / (f + k1 * (1 - b + b * allc[i].len / avg))
        return sc

    def _gate(self, doc: dict):
        path = self.ccfg.get("contract_config_path") or CONTRACT_CONFIG_PATH
        kind = doc.get("stored_text_kind")
        # fail-closed allow-list (GAP-DBKIND-1028): kind must be listed; the deny-list state check stays as a second gate
        return (isinstance(kind, str) and kind in full_document_kinds(path)
                and acquisition_state(doc, path) == ACQ_ACQUIRED)

    def _prep(self, doc: dict) -> dict | None:
        """Gate + duplicate handling; returns a status dict to return early, else None."""
        key = doc.get("source_key")
        if not isinstance(key, str) or not key:
            return {"status": "skipped", "reason": "no_source_key", "source_key": key}
        if key in self._docs:
            return {"status": "skipped", "reason": "duplicate", "source_key": key}
        if not self._gate(doc):
            self.skipped.append({"source_key": key, "reason": "not_full_document",
                                 "stored_text_kind": doc.get("stored_text_kind")})
            return {"status": "skipped", "reason": "not_full_document", "source_key": key}
        return None

    def _store_doc(self, doc: dict) -> dict:
        rec = {"source_key": doc["source_key"], "url": doc.get("url"), "doi": doc.get("doi"),
               "title": doc.get("title"), "authors": doc.get("authors"), "text": doc.get("text") or "", "year": _year(doc),
               "published": doc.get("published"), "source_type": doc.get("source_type"),
               "retrieved_at": doc.get("retrieved_at") or (self._clock() if self._clock else None),
               "stored_text_kind": doc.get("stored_text_kind"),
               "licence": doc.get("licence"), "licence_decision": doc.get("licence_decision"),
               "admission_passed": doc.get("admission_passed") is True,
               "db_source_id": doc.get("db_source_id"), "db_canonical_source_id": doc.get("db_canonical_source_id")}
        if doc.get("isbn"):  # WED-ACQ2: only acquired books carry one; absent keeps the record shape unchanged
            rec["isbn"] = doc["isbn"]
        self._docs[rec["source_key"]] = rec
        return rec

    def _dom_entry(self, did: int, created: set) -> dict:
        d = self.registry.get(did)
        return {"domain_id": did, "name": d.name, "name_norm": d.name_norm, "description": d.description,
                "origin": d.origin, "created": did in created}

    def _chunks_vs(self, pk: str, dids, asserted: str) -> list[dict]:
        """Re-classify one document's chunks NOW (same rule as at ingest: threshold, secondary ratio, per-chunk cap) and
        add only the labels for the given (new) domains; additive. Using the full classification keeps a boilerplate-
        similar chunk from picking up a new tiny domain just because that domain's score alone clears the threshold."""
        dids, out = set(dids), []
        cap = int(self._icfg["max_domains_per_chunk"])
        for c in self._doc_chunks.get(pk, ()):  # NOCAP-1109: the document's own chunks, same order as the full scan
            have = {d.domain_id for d in self.registry.domains_for_chunk(c["span_id"])}
            for hit in self.identifier.classify(c["text"], "chunk"):
                did = int(hit["domain_id"])
                if did in dids and did not in have and len(have) < cap and \
                        self.registry.associate_chunk(did, c["span_id"], asserted):
                    have.add(did)
                    out.append({"span_id": c["span_id"], "domain_id": did, "asserted_by": asserted})
        return out

    def _doc_from_chunks(self, pk: str, asserted: str) -> list[int]:
        """Document-level SECONDARY labels from chunk evidence (additive; per-document cap max_domains_per_doc).
        A chunk votes for its best domain, but only a domain that is NOT already one of the document's own primary
        domains. A document that joined domains is judged by each chunk's best domain overall. A document that matched
        no domain and founded its own single-document domain ("lone") is judged by each chunk's best FOREIGN domain, and
        needs at least TWO foreign domains with enough votes: one similar domain only says "resembles", two say "bridges".
        A domain qualifies with at least corpus.doc_min_chunks chunks AND doc_min_chunk_share of the document's chunks, so
        a short bridge passage in a long document stays a chunk-level fact."""
        asserted = asserted + SECONDARY  # chunk-derived document labels are secondary (never change a profile)
        cap = int(self._icfg["max_domains_per_doc"])
        own = {d.domain_id for d in self.registry.domains_for_corpus(pk)
               if not (self.registry.association_actor("corpus", d.domain_id, pk) or "").endswith(SECONDARY)}
        lone = bool(own) and all(len(self.registry.corpora_for_domain(d)) == 1 for d in own)
        count: Counter = Counter()
        n_chunks = 0
        for c in self._doc_chunks.get(pk, ()):  # NOCAP-1109: the document's own chunks, same order as the full scan
            n_chunks += 1
            top = [h for h in self.identifier.classify(c["text"], "chunk", exclude=own if lone else ())
                   if int(h["domain_id"]) not in own]
            if top:
                count[int(top[0]["domain_id"])] += 1
        need = max(int(self.ccfg["doc_min_chunks"]), math.ceil(float(self.ccfg["doc_min_chunk_share"]) * n_chunks))
        qualifying = [did for did, n in count.items() if n >= need]
        if lone and len(qualifying) < 2:
            return []
        have = {d.domain_id for d in self.registry.domains_for_corpus(pk)}
        added = []
        for did, n in sorted(count.items(), key=lambda x: (-x[1], x[0])):
            if n >= need and did not in have and len(have) < cap and self.registry.associate_corpus(did, pk, asserted):
                have.add(did)
                added.append(did)
                self.identifier.note_association(pk, did)  # D4
        return added

    # --- ingest ----------------------------------------------------------------------------------------------
    def ingest_document(self, doc: dict) -> dict:
        if not self._enabled():
            return {"status": "off"}
        early = self._prep(doc)
        if early:
            return early
        asserted = self.ccfg["asserted_by"]
        rec = self._store_doc(doc)
        key = rec["source_key"]
        res = self.identifier.add_document(key, rec["text"], asserted)
        created = {int(x) for x in res.get("created", [])}
        doc_ids = sorted({int(d["domain_id"]) for d in res.get("domains", [])})
        self._reported[key] = set(doc_ids)
        used = set(doc_ids)
        chunks_plan = []
        for s, e in self._chunk_spans(rec["text"]):
            c = self._index_chunk(rec, s, e)
            ids = []
            for d in self.identifier.classify(c["text"], "chunk"):
                did = int(d["domain_id"])
                self.registry.associate_chunk(did, c["span_id"], asserted)
                ids.append(did)
            used.update(ids)
            chunks_plan.append({"span_id": c["span_id"], "source_key": key, "start": s, "end": e,
                                "text": c["text"], "domains": ids})
        # D3: the current document's own chunks may support further document-level domains (multi-domain documents)
        for did in self._doc_from_chunks(key, asserted):
            if did not in doc_ids:
                doc_ids.append(did)
        doc_ids = sorted(doc_ids)
        self._reported[key] = set(doc_ids)
        used.update(doc_ids)
        # D3: a newly created or promoted domain is scored incrementally against EARLIER documents and chunks
        # (additive only; nothing is ever removed) and reported so dar_store.persist can record it.
        # (document-level re-classification against a new, still tiny domain is NOT done: its profile is one document,
        # so it would smear boilerplate-similar documents; secondary document labels come from chunk evidence only)
        doc_ids = sorted({d.domain_id for d in self.registry.domains_for_corpus(key)})
        self._reported[key] = set(doc_ids)
        used.update(doc_ids)
        promoted_plan = []
        promoted = [pk for pk in (res.get("promoted") or []) if pk in self._docs and pk != key]
        others = [k for k in self._docs if k != key] if created else promoted
        for pk in dict.fromkeys(promoted + others):
            known = self._reported.setdefault(pk, set())
            now = {d.domain_id for d in self.registry.domains_for_corpus(pk)}
            cha = self._chunks_vs(pk, sorted((now - known) | created), asserted)
            self._doc_from_chunks(pk, asserted)
            now = {d.domain_id for d in self.registry.domains_for_corpus(pk)}
            new = sorted(now - known)
            known |= now
            ca = [{"domain_id": d, "canonical_source_id": pk,
                   "asserted_by": self.registry.association_actor("corpus", d, pk) or asserted,
                   "name_norm": self.registry.get(d).name_norm} for d in new]
            cha = [dict(x, name_norm=self.registry.get(x["domain_id"]).name_norm,
                        start=self._by_span[x["span_id"]]["start"], end=self._by_span[x["span_id"]]["end"])
                   for x in cha]
            used.update(new)
            used.update(c["domain_id"] for c in cha)
            if ca or cha or pk in promoted:
                pr = self._docs[pk]
                promoted_plan.append({"source_key": pk, "db_source_id": pr["db_source_id"],
                                      "db_canonical_source_id": pr["db_canonical_source_id"],
                                      "corpus_assoc": ca, "chunk_assoc": cha})
        for e in self.consolidate(asserted):  # near-duplicate derived domains merge into one canonical domain
            for ca in e["corpus_assoc"]:
                used.add(ca["domain_id"])
            for ch in e["chunk_assoc"]:
                used.add(ch["domain_id"])
            if e["source_key"] == key:
                doc_ids = sorted({d.domain_id for d in self.registry.domains_for_corpus(key)})
            promoted_plan.append(e)
        return {"status": "indexed", "reason": None, "source_key": key,
                "doc": {"canonical_source_id": key, "url": rec["url"], "title": rec["title"],
                        "db_source_id": rec["db_source_id"], "db_canonical_source_id": rec["db_canonical_source_id"]},
                "domains": [self._dom_entry(d, created) for d in sorted(used)],
                # asserted_by is the REAL marker per association (":secondary" labels must not become primary on restore)
                "corpus_assoc": [{"domain_id": d, "canonical_source_id": key,
                                  "asserted_by": self.registry.association_actor("corpus", d, key) or asserted,
                                  "name_norm": self.registry.get(d).name_norm} for d in doc_ids],
                "chunks": chunks_plan, "promoted": promoted_plan}

    # --- merge with external results (comparable relevance) ----------------------------------------------------
    def _stems(self, text: str) -> set[str]:
        n = int(self.ccfg["relevance_stem_len"])  # crude prefix stemming: forecast/forecasts/forecasting share a stem
        return {t[:n] for t in tokenize(text, self._icfg)}

    def relevance(self, question: str, text: str) -> float:
        """Query-term coverage in [0, 1]: the share of the question's content stems found in `text`. ONE function for
        external (title + abstract) and stored (title + chunk) candidates, so their scores are comparable."""
        q = self._stems(question)
        return len(q & self._stems(text)) / len(q) if q else 0.0

    def floor_external(self, question: str, external: list, pool: list | None = None) -> list:
        """NONASK-1103 WP-H (FLAG-OFF: corpus.reading_list_floor.enabled must be exactly true; absent = off): drop
        external results below a topical-fit floor. An item stays only if relevance(title + snippet) >= min_relevance
        AND it contains a non-generic question stem (the WP-G1 anchor function, df over the candidate pool titles and
        snippets, anchor cut max_df_ratio). Stored items (span / in_corpus) are never touched. Flag off, a missing or
        invalid threshold, or any error returns the input list by identity."""
        fc = self.ccfg.get("reading_list_floor")
        if not isinstance(fc, dict) or fc.get("enabled") is not True:
            return external
        try:
            from vera.fulldoc_admission import topic_anchor_stems

            floor, cut = float(fc["min_relevance"]), float(fc["max_df_ratio"])
            txt = lambda r: f"{r.get('title') or ''} {r.get('snippet') or ''}"  # noqa: E731
            texts = [txt(r) for r in list(pool if pool is not None else external) if isinstance(r, dict)]
            df: dict = {}
            for t in texts:
                for st in self._stems(t):
                    df[st] = df.get(st, 0) + 1
            anchors = topic_anchor_stems(self._stems(question), df, len(texts), cut)
            keep = [r for r in external if not isinstance(r, dict) or r.get("span") or r.get("in_corpus")
                    or (self.relevance(question, txt(r)) >= floor and (not anchors or anchors & self._stems(txt(r))))]
            return keep if len(keep) != len(external) else external
        except Exception:  # noqa: BLE001  (fails open to the current behaviour)
            return external

    def rank_merge(self, question: str, external: list, stored: list, n: int) -> list:
        """Order external results and stored chunks under one policy; truncation happens later, in the caller.
        * the first head = min(len(external), max(min_fallback_slots, n - stored_merge_max)) external results are
          protected and stay first, in place (the unrestricted branch is never displaced);
        * a stored chunk below corpus.stored_min_relevance is not merged next to external results at all (an
          irrelevant chunk must not push relevant external evidence out; with NO external results stored chunks are
          returned as ranked, because nothing is displaced);
        * the remaining external results and the qualifying stored chunks are ordered by the common relevance
          (ties: external first, then original rank); a document present in both (same DOI/URL) appears once, the
          external entry kept."""
        ccfg = self.ccfg
        external = self.floor_external(question, external, list(external) + list(stored))  # WP-H, flag off = identity
        if not external:
            return list(stored)
        m = int(ccfg["stored_merge_max"])
        reserve = min(len(external), int(self.cfg.get("min_fallback_slots", 0) or 0))
        head = min(len(external), max(reserve, n - m, 0))
        floor = float(ccfg["stored_min_relevance"])

        def key(r):
            return str(r.get("doi") or "").lower() or str(r.get("url") or "").lower().rstrip("/").split("#")[0]
        have = {key(r) for r in external}
        cand = []
        for i, r in enumerate(external[head:]):
            cand.append((-self.relevance(question, f"{r.get('title') or ''} {r.get('snippet') or ''}"), 0, i, r))
        for i, r in enumerate(stored):
            if key(r) in have:
                continue
            rel = self.relevance(question, f"{r.get('title') or ''} {r.get('snippet') or ''}")
            if rel >= floor:
                cand.append((-rel, 1, i, r))
        cand.sort(key=lambda x: x[:3])
        return list(external[:head]) + [x[3] for x in cand]

    CONSOLIDATED = ":consolidated" + SECONDARY  # asserted_by suffix: counted as SECONDARY on restore (no profile change)

    def consolidate(self, asserted: str | None = None) -> list[dict]:
        """Merge near-duplicate DERIVED domains (config identification.consolidate_*): the alias stops routing and
        classification (identifier.merge_alias), its member documents and chunks gain the canonical domain as an
        additional (secondary, ':consolidated') association, and nothing is ever deleted. A deterministic function of
        the current state: restore()/apply_delta() simply call it again. Returns plan entries ("promoted"-shaped) for
        dar_store.persist; [] when disabled or nothing to merge."""
        icfg = self._icfg
        if icfg.get("consolidate_enabled") is not True:
            return []
        from vera.dar_consolidate import plan_consolidation

        asserted = (asserted or self.ccfg["asserted_by"]) + self.CONSOLIDATED
        entries: dict[str, dict] = {}
        for m in plan_consolidation(self.identifier, self.registry, icfg):
            can, ali = int(m["canonical"]), int(m["alias"])
            self.identifier.merge_alias(ali, can)
            for key in self.registry.corpora_for_domain(ali):
                new_doc = self.registry.associate_corpus(can, key, asserted)
                e = entries.setdefault(key, {"source_key": key, "db_source_id": self._docs[key]["db_source_id"],
                                             "db_canonical_source_id": self._docs[key]["db_canonical_source_id"],
                                             "corpus_assoc": [], "chunk_assoc": []})
                if new_doc:
                    e["corpus_assoc"].append({"domain_id": can, "canonical_source_id": key, "asserted_by": asserted,
                                              "name_norm": self.registry.get(can).name_norm})
                self._reported.setdefault(key, set()).add(can)
            for span_id in self.registry.chunks_for_domain(ali):
                c = self._by_span.get(span_id)
                if c is not None and self.registry.associate_chunk(can, span_id, asserted):
                    e = entries.setdefault(c["source_key"], {
                        "source_key": c["source_key"], "db_source_id": self._docs[c["source_key"]]["db_source_id"],
                        "db_canonical_source_id": self._docs[c["source_key"]]["db_canonical_source_id"],
                        "corpus_assoc": [], "chunk_assoc": []})
                    e["chunk_assoc"].append({"span_id": span_id, "domain_id": can, "asserted_by": asserted,
                                             "start": c["start"], "end": c["end"],
                                             "name_norm": self.registry.get(can).name_norm})
        return [e for e in entries.values() if e["corpus_assoc"] or e["chunk_assoc"]]

    def chunking(self) -> dict:
        return {k: int(self.ccfg[k]) for k in ("chunk_chars", "chunk_overlap", "min_chunk_chars")}

    # --- restore (no identification) --------------------------------------------------------------------------
    def restore(self, snapshot: dict) -> dict:
        if not self._enabled():
            return {"status": "off"}
        reg = self.registry
        snap_chunking = snapshot.get("chunking")
        if snap_chunking is not None and {k: int(v) for k, v in dict(snap_chunking).items()} != self.chunking():
            return {"status": "chunking_mismatch", "snapshot_chunking": dict(snap_chunking),
                    "corpus_chunking": self.chunking(), "documents": 0}
        raw_assoc = list(snapshot.get("chunk_assoc") or [])
        todo = []
        for d in snapshot.get("documents") or []:
            d = dict(d)
            d.setdefault("admission_passed", True)
            if self._prep(d):
                continue
            todo.append(d)
        if raw_assoc:
            spans_of = {d["source_key"]: set(self._chunk_spans(d.get("text") or "")) for d in todo}
            unmatched = sum(1 for a in raw_assoc if a["source_key"] in spans_of
                            and (int(a["start"]), int(a["end"])) not in spans_of[a["source_key"]])
            if unmatched:
                return {"status": "chunking_mismatch", "unmatched": unmatched, "corpus_chunking": self.chunking(),
                        "documents": 0, "pending_not_restored": True}
        for d in snapshot.get("domains") or []:
            reg.register(d["name"], d.get("description") or "", d.get("origin") or "manual")
        corpus: dict[str, list[int]] = {}
        for a in snapshot.get("corpus_assoc") or []:
            dom = reg.get(a["name_norm"])
            if dom is not None:
                if not str(a.get("asserted_by") or "").endswith(SECONDARY):  # primary members define the profile
                    corpus.setdefault(a["source_key"], []).append(dom.domain_id)
                reg.associate_corpus(dom.domain_id, a["source_key"], a.get("asserted_by") or self.ccfg["asserted_by"])
        chunk_assoc: dict[tuple, list] = {}
        for a in raw_assoc:
            chunk_assoc.setdefault((a["source_key"], int(a["start"]), int(a["end"])), []).append(a)
        restored = 0
        for d in todo:
            if d["source_key"] in self._docs:
                continue
            rec = self._store_doc(d)
            key = rec["source_key"]
            self.identifier.restore_document(key, rec["text"], sorted(set(corpus.get(key, []))))
            self._reported[key] = {x.domain_id for x in reg.domains_for_corpus(key)}
            for s, e in self._chunk_spans(rec["text"]):
                c = self._index_chunk(rec, s, e)
                for a in chunk_assoc.get((key, s, e), []):
                    dom = reg.get(a["name_norm"])
                    if dom is not None:
                        reg.associate_chunk(dom.domain_id, c["span_id"],
                                            a.get("asserted_by") or self.ccfg["asserted_by"])
            restored += 1
        self.consolidate()  # aliases are derived state: rebuilt deterministically (rows already persisted are duplicates)
        return {"status": "restored", "documents": restored, "pending_not_restored": True}

    def apply_delta(self, delta: dict) -> dict:
        """Incremental freshness: apply the rows ADDED since the last load (dar_store.DarReader.load_delta; same shape
        as a snapshot, only newer rows) to the live index without a rebuild. New documents are chunked and indexed with
        their stored associations; new associations of already-known documents (promotions, secondary labels) are
        added, and a primary one also feeds the domain profile exactly as restore() would. Anything that cannot be
        applied consistently returns {"status": "needs_full", "reason": ...} and changes NOTHING that the caller
        cannot repeat with a full rebuild (the caller then rebuilds)."""
        if not self._enabled():
            return {"status": "off"}
        reg = self.registry
        new_docs = []
        for d in delta.get("documents") or []:
            d = dict(d)
            d.setdefault("admission_passed", True)
            if d.get("source_key") in self._docs:
                old, new = self._docs[d["source_key"]].get("db_source_id"), d.get("db_source_id")
                if isinstance(old, int) and isinstance(new, int) and not isinstance(new, bool) and new > old:
                    # a NEW VERSION of a known document: its text/chunks differ, so patching in place is unsafe
                    return {"status": "needs_full", "reason": "new_version_of_known_document"}
                continue
            if d.get("source_key") in self._rejected:
                continue
            if self._prep(d):
                if isinstance(d.get("source_key"), str):
                    self._rejected.add(d["source_key"])
                continue
            new_docs.append(d)
        if self._rejected:  # associations of gate-refused documents are not indexable: drop them (idempotent re-reads)
            delta = {**delta, "corpus_assoc": [a for a in delta.get("corpus_assoc") or [] if a["source_key"] not in self._rejected],
                     "chunk_assoc": [a for a in delta.get("chunk_assoc") or [] if a["source_key"] not in self._rejected]}
        known = set(self._docs) | {d["source_key"] for d in new_docs}
        for a in list(delta.get("corpus_assoc") or []):
            if a["source_key"] not in known:
                return {"status": "needs_full", "reason": "association_for_unknown_document"}
            if reg.get(a["name_norm"]) is None and a["name_norm"] not in {x.get("name_norm") for x in delta.get("domains") or []}:
                return {"status": "needs_full", "reason": "association_for_unknown_domain"}
        spans_new = {d["source_key"]: set(self._chunk_spans(d.get("text") or "")) for d in new_docs}
        for a in list(delta.get("chunk_assoc") or []):
            k, st, en = a["source_key"], int(a["start"]), int(a["end"])
            if k in spans_new:
                if (st, en) not in spans_new[k]:
                    return {"status": "needs_full", "reason": "chunking_mismatch"}
            elif f"{k}:{st}-{en}" not in self._by_span:
                return {"status": "needs_full", "reason": "chunk_for_unknown_span"}
        for d in delta.get("domains") or []:
            reg.register(d["name"], d.get("description") or "", d.get("origin") or "manual")
        primary_new: dict[str, list[int]] = {}
        existing_primary = []
        for a in delta.get("corpus_assoc") or []:
            dom = reg.get(a["name_norm"])
            if dom is None:
                continue
            prim = not str(a.get("asserted_by") or "").endswith(SECONDARY)
            if a["source_key"] in spans_new:
                if prim:
                    primary_new.setdefault(a["source_key"], []).append(dom.domain_id)
            elif prim and not any(x.domain_id == dom.domain_id for x in reg.domains_for_corpus(a["source_key"])):
                existing_primary.append((a["source_key"], dom.domain_id))
            reg.associate_corpus(dom.domain_id, a["source_key"], a.get("asserted_by") or self.ccfg["asserted_by"])
        for key, did in existing_primary:  # a late primary association of a known document feeds its profile
            tf = self.identifier._doc_tf.get(key)
            if tf:
                self.identifier._add_to_profile(did, tf)
        by_chunk: dict[tuple, list] = {}
        for a in delta.get("chunk_assoc") or []:
            by_chunk.setdefault((a["source_key"], int(a["start"]), int(a["end"])), []).append(a)
        added = 0
        for d in new_docs:
            rec = self._store_doc(d)
            key = rec["source_key"]
            self.identifier.restore_document(key, rec["text"], sorted(set(primary_new.get(key, []))))
            self._reported[key] = {x.domain_id for x in reg.domains_for_corpus(key)}
            for s, e in self._chunk_spans(rec["text"]):
                c = self._index_chunk(rec, s, e)
                for a in by_chunk.pop((key, s, e), []):
                    dom = reg.get(a["name_norm"])
                    if dom is not None:
                        reg.associate_chunk(dom.domain_id, c["span_id"], a.get("asserted_by") or self.ccfg["asserted_by"])
            added += 1
        for (k, st, en), rows in by_chunk.items():  # chunk associations of already-indexed documents
            for a in rows:
                dom = reg.get(a["name_norm"])
                if dom is not None:
                    reg.associate_chunk(dom.domain_id, f"{k}:{st}-{en}", a.get("asserted_by") or self.ccfg["asserted_by"])
        self.consolidate()
        return {"status": "applied", "documents": added}

    # --- retrieval -------------------------------------------------------------------------------------------
    def _question_domains(self, question: str) -> list[int]:
        cap = self.ccfg.get("max_stored_domain_branches", self.cfg.get("max_domain_branches"))  # stored branches are local (no search cost)
        ids = [int(d["domain_id"]) for d in self.identifier.classify(question, "question")]
        return ids[: int(cap)] if cap is not None else ids

    def _bm25(self, q_terms: list[str], pool, limit: int, scores: dict | None = None) -> list:
        """Top `limit` chunks of `pool` by BM25 (ties: source_key, start). pool is this version's chunk list or a
        subset of it; `scores` (from _scores) may be passed to score once per question."""
        sc = self._scores(q_terms) if scores is None else scores
        if not sc:
            return []
        allc = self._all
        if pool is self._chunks:
            items = sc.items()
        else:
            ids = {c.idx for c in pool}
            items = [(i, s) for i, s in sc.items() if i in ids]
        scored = [(-s, allc[i].doc["source_key"], allc[i].start, allc[i]) for i, s in items if s > 0]
        scored.sort(key=lambda x: x[:3])
        return [x[3] for x in scored[:limit]]

    def _candidate(self, c: dict, branches: list[str]) -> dict:
        d = self._docs[c["source_key"]]
        return {"span_id": c["span_id"], "source_key": c["source_key"], "canonical_source_id": c["source_key"],
                "source_id": d["db_source_id"] or c["source_key"], "text": c["text"], "start": c["start"],
                "end": c["end"], "source_title": d["title"], "url": d["url"], "doi": d["doi"], "year": d["year"],
                "source_type": d["source_type"], "retrieved_at": d["retrieved_at"],
                "domains": sorted(x.domain_id for x in self.registry.domains_for_chunk(c["span_id"])),
                "branches": list(branches), "status": "proposed"}

    def retrieve(self, question: str, k: int | None = None, max_per_doc: int | None = None, vector_fn=None) -> dict:
        trace = {"routing": "off", "question_domains": [], "branch_counts": {}, "reserved": 0, "fallback_only": 0,
                 "limitation": LIMITATION}
        if not self._enabled():
            return {"candidates": [], "trace": trace}
        trace["routing"] = "on"
        k = int(k if k is not None else self.ccfg["result_k"])
        if k <= 0:
            return {"candidates": [], "trace": trace}
        bk = int(self.ccfg["branch_k"])
        if not self._chunks:
            return {"candidates": [], "trace": trace}
        q_terms = self._query_terms(question)
        fb = domain_routing.FALLBACK
        scores = self._scores(q_terms)  # NOCAP-1109: scored once; each branch takes its pool's top chunks
        branches = [(fb, self._bm25(q_terms, self._chunks, bk, scores))]
        qdoms = self._question_domains(question)
        trace["question_domains"] = qdoms
        trace["branch_counts"][fb] = len(branches[0][1])
        allc = self._all
        for did in qdoms:  # the domain pool restricted to scored chunks (an unscored chunk is never ranked anyway)
            pool = [allc[i] for i in scores
                    if did in {x.domain_id for x in self.registry.domains_for_chunk(allc[i].span_id)}]
            items = self._bm25(q_terms, pool, bk, scores)
            name = f"domain:{did}"
            trace["branch_counts"][name] = len(items)
            branches.append((name, items))
        groups, _ = domain_routing._rrf(branches, int(self.cfg["rrf_k"]), key_fn=lambda c: c["span_id"])
        base = sorted((g for g in groups if fb in g["branches"]), key=lambda g: g["best"])
        reserved = base[: domain_routing._reserve_count(self.cfg, len(branches[0][1]), k)] if len(branches) > 1 else []
        rid = {id(g) for g in reserved}
        rest = sorted((g for g in groups if id(g) not in rid), key=lambda g: (-g["score"], g["best"]))
        trace["reserved"] = len(reserved)
        if vector_fn is not None:
            # HYBRID-1109 (flag-gated by the caller): the word-matching order above (BM25 + domain branches, unchanged)
            # and the vector list are fused as two peers by rank-only RRF; ties keep the word-matching item first.
            try:
                vitems = list(vector_fn(self, scores) or ())
            except Exception as exc:  # noqa: BLE001  any vector failure => word-only retrieval (never lose candidates)
                log.info("hybrid vector retrieval failed, word-only (%s)", type(exc).__name__)
                vitems = []
            trace["branch_counts"]["vector"] = len(vitems)
            lex = [g["rep"] for g in reserved + rest]
            names = {g["key"]: g["branches"] for g in groups}
            fused, _ = domain_routing._rrf([("lexical", lex), ("vector", vitems)], int(self.cfg["rrf_k"]),
                                           key_fn=lambda c: c["span_id"])
            fused.sort(key=lambda g: (-g["score"], g["best"]))
            pre = [c["span_id"] for c in (getattr(vector_fn, "reserve", None) or ())]
            if pre:  # stage 2: the best passage-level matches go first (a passage-only match cannot be outvoted)
                rank = {sid: r for r, sid in enumerate(pre)}
                fused.sort(key=lambda g: rank.get(g["key"], len(rank)))
                trace["reserved_passages"] = sum(1 for g in fused if g["key"] in rank)
            for g in fused:
                g["branches"] = list(names.get(g["key"], [])) + (["vector"] if "vector" in g["branches"] else [])
            reserved, rest = [], fused
        per_doc, picked = Counter(), []
        cap = int(self.ccfg["max_chunks_per_doc"]) if max_per_doc is None else max(int(max_per_doc), 1)
        for g in reserved + rest:  # a document may fill at most max_chunks_per_doc slots, so one long document cannot
            sk = g["rep"]["source_key"]  # monopolise the candidates (cross-domain questions need several documents)
            if per_doc[sk] < cap:
                per_doc[sk] += 1
                picked.append(g)
        out = [self._candidate(g["rep"], g["branches"]) for g in picked[:k]]
        trace["fallback_only"] = sum(1 for g in groups if g["branches"] == [fb])
        return {"candidates": out, "trace": trace}

    def _snippet(self, text: str) -> str:
        n = int(self.ccfg["snippet_chars"])
        if len(text) <= n:
            return text
        cut = text[:n]
        if not text[n].isspace():
            ws = max((i for i, ch in enumerate(cut) if ch.isspace()), default=None)
            if ws:
                cut = cut[:ws]
        return cut.rstrip()

    def as_sources(self, question: str, k: int | None = None) -> tuple[list[dict], dict]:
        res = self.retrieve(question, k)
        cap = int(self.ccfg["stored_fill_max"])  # upper bound; vera.grounded_ask limits what it merges next to external results
        out, seen_docs = [], set()
        for c in res["candidates"]:
            if len(out) >= cap:
                break
            if c["source_key"] in seen_docs:  # one representative chunk per document (retrieve_all dedups by URL anyway)
                continue
            seen_docs.add(c["source_key"])
            d = self._docs[c["source_key"]]
            out.append({"title": c["source_title"], "url": c["url"], "retrieved_at": d["retrieved_at"],
                        "dar_span": {"start": c["start"], "end": c["end"], "span_id": c["span_id"]}, "published": d["published"] or (
                str(c["year"]) if c["year"] else None), "snippet": self._snippet(c["text"]), "source_type": c["source_type"],
                "doi": c["doi"], "authors": d.get("authors"), "stored_text_kind": d["stored_text_kind"],
                "provider": self.ccfg["stored_discovered_via"], "licence": d.get("licence"),
                "admission_passed": d.get("admission_passed") is True,
                "disposition": c["status"],
                "canonical_source_id": d.get("db_canonical_source_id"), "source_id": d.get("db_source_id"),
                "discovered_via": self.ccfg["stored_discovered_via"], "_domain_branches": ["stored"]})
        t = res["trace"]
        return out, {"routing": t["routing"], "stored_candidates": len(res["candidates"]), "returned": len(out),
                     "question_domains": len(t["question_domains"])}

    # --- Gate C / M4 hand-off ----------------------------------------------------------------------------------
    def to_evidence_corpus(self, question: str, candidates: list[dict]) -> dict:
        gen = self.ccfg["general_sub_question"]
        req = self.ccfg["sub_question_required"]
        pre = self.ccfg["domain_sub_question_prefix"]
        sqs = [{"id": gen["id"], "text": gen["text"], "required": req}]
        qdoms = self._question_domains(question)
        for did in qdoms:
            sqs.append({"id": f"{pre}{did}", "text": self.registry.get(did).name, "required": req})
        spans = []
        for c in candidates:
            ids = [gen["id"]] + [f"{pre}{d}" for d in qdoms if d in c.get("domains", [])]
            spans.append({"span_id": c["span_id"], "source_id": c["source_id"], "text": c["text"],
                          "source_title": c["source_title"], "year": c["year"], "source_type": c["source_type"],
                          "independence_group": c["source_key"], "sub_question_ids": ids})
        return {"question": question, "run_id": "dar", "sub_questions": sqs, "spans": spans}

    def domain_status(self, domain_id: int) -> str | None:
        """'provisional' while a DERIVED domain has fewer than identification.established_support member documents,
        'established' afterwards, 'configured' for config/manual/fixture domains, None if unknown. Computed from the
        stored associations (domain_corpus rows), so it needs no extra column and survives a restart (007 unchanged).
        A provisional domain still ranks (it is a ranking branch) and never excludes anything."""
        d = self.registry.get(domain_id)
        if d is None:
            return None
        if d.origin != "derived":
            return "configured"
        n = len(self.registry.corpora_for_domain(domain_id))
        return "established" if n >= int(self._icfg["established_support"]) else "provisional"

    def docs_with_terms(self, terms) -> set:
        """WPQ-CYCLE2-1111 (c2): source keys of the DOCUMENTS of this version holding at least one chunk with any of the
        index keys `terms` (already in index form: stemmed when stemming is on). Read-only, in memory, no I/O. Applies the
        same version and mask filter as `_scores`: postings are shared append-only across forks, so an entry at or beyond
        this version's chunk count, or masked out of an evidence view, is not counted. Counts documents, not chunks."""
        allc, mask, out = self._all, self._mask, set()
        n_all = len(allc)
        for t in terms:
            for v in self._post.get(t) or ():
                i = v >> _CNT_BITS
                if i >= n_all or (mask is not None and not mask[i]):
                    continue
                out.add(allc[i].source_key)
        return out

    def closed_form_docs(self, a: str, b: str) -> set:
        """WPQ-CYCLE2-1111 (c2): documents holding the closed form of the open compound "a b". The index stemmer does not
        fold singular and plural ("superstring" -> "superstr", "superstrings" -> "superstring"), so the index keys K are
        the closed form with and without the second word's suffix, each through the index's own stemmer. No word list."""
        a, b = str(a).lower(), str(b).lower()
        if self._stem_on():
            keys = {light_stem(a + b), light_stem(a + light_stem(b)), light_stem(a + light_stem(b) + "s")}
        else:
            keys = {a + b, a + b + "s"} | ({a + b[:-1]} if b.endswith("s") else set())
        return self.docs_with_terms(keys)

    def doc_keys(self) -> frozenset:
        """Membership keys (see doc_key_set) of every stored document, not only those retrieved for a question."""
        out: set = set()
        for d in self._docs.values():
            out |= doc_key_set(d.get("doi"), d.get("url"), d.get("isbn"))
            out |= doc_key_set(None, d.get("source_key"))
        return frozenset(out)

    def stats(self) -> dict:
        st = [self.domain_status(d.domain_id) for d in self.registry.list_domains()]
        return {"documents": len(self._docs), "chunks": len(self._chunks),
                "domains": len(self.registry.list_domains()), "skipped": len(self.skipped),
                "domains_provisional": st.count("provisional"), "domains_established": st.count("established"),
                "chunking": self.chunking()}
