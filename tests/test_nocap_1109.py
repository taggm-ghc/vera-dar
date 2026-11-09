"""NOCAP-1109 (owner decision OD-1109-NOCAP, 2026-11-09, verbatim: "the 30-document cieling is for when to stop
additional runs of searchs for new documents to ingest. there should be no cap upon stored docs and chunks to accept
from internal search results"). Offline: synthetic fixtures, fake DB-API connection, stubbed search; no network, DB or
model. Proves: (1) every stored acquired document is eligible for internal search (no document, byte or scan cap and
no R-ACQ-SERVECAP drop); (2) the ~30-document soft ceiling still stops acquisition rounds."""
from __future__ import annotations

import pytest

from tests import acq_support as A
from tests import fulltext_synth
from vera import acq_config, auto_acquire, fulldoc_corpus, fulldoc_db, query_variants6

N = 100


def _plans(n=N):
    """n stored acquired documents through the unchanged acquisition pipeline (25 per job, distinct DOIs and texts)."""
    plans = []
    for start in range(0, n, 25):
        auto_acquire.reset_for_tests()
        recs, jats, cands = {}, {}, []
        for i in range(start, min(n, start + 25)):
            doi, pmc = f"10.9999/nocap.{i:04d}", f"PMC8{i:05d}"
            recs[doi] = A.epmc_record(doi, pmcid=pmc)
            jats[pmc] = fulltext_synth.jats(f"nocap{i}", "epi", doi, seed=i + 3)
            cands.append(A.cand(doi))
        store = A.Store()
        auto_acquire.run_job(A.job(A.Q_EPI), A.cfg(max_docs_per_job=25), A.deps(cands, A.Web(recs, jats), store=store),
                             clock=A.clock_at(), env=A.ENV_ON)
        plans += store.plans
    auto_acquire.reset_for_tests()
    return plans


_CACHE: dict = {}


def plans100():
    if "p" not in _CACHE:
        _CACHE["p"] = _plans()
    return _CACHE["p"]


def test_shipped_config_has_no_serve_cap_keys():
    b = acq_config.load_config()["bounds"]
    for k in ("served_acquired_max", "served_acquired_max_bytes", "served_scan_max"):
        assert k not in b, k
    assert "OD-1109-NOCAP" in b["_served_acquired_max_note"]
    assert not hasattr(fulldoc_db, "R_ACQCAP")
    for sql in (fulldoc_db.SQL_ACQ_SOURCES, fulldoc_db.SQL_ACQ_INDEX):
        assert "LIMIT" not in sql.upper()


def test_100_stored_documents_all_searchable_no_servecap():
    plans = plans100()
    assert len(plans) == N
    src, spans = A.rows_from_plans(plans)
    c = fulldoc_db.load_acquired_rows(src, spans, fulldoc_corpus.load_config(), A.cfg())
    assert c.status == "ok" and len(c.acquired) == N
    assert not any(u.get("code") == "R-ACQ-SERVECAP" for u in c.unusable)
    served = {a["canonical_source_id"] for a in c.acquired}
    assert len(served) == N
    assert {ch["source_key"] for ch in c._dar._chunks} == served  # every document has chunks in the search index


def test_100_stored_documents_all_served_through_the_db_read_path(monkeypatch):
    """The full background read (load_acquired_db: index SELECT, by-id SELECT, span SELECT) with the shipped config."""
    plans = plans100()
    conn = A.FakeConn(plans)
    c = fulldoc_db.load_acquired_db(lambda: conn, 1, fulldoc_corpus.load_config(), A.cfg())
    assert c.status == "ok" and len(c.acquired) == N
    assert len({ch["source_key"] for ch in c._dar._chunks}) == N
    assert not any(u.get("code") == "R-ACQ-SERVECAP" for u in c.unusable)
    assert fulldoc_db.SQL_ACQ_INDEX in conn.log  # shipped config: topic-balanced order, still every row


def test_legacy_cap_keys_if_reintroduced_are_ignored():
    plans = plans100()[:40]
    src, spans = A.rows_from_plans(plans)
    c = fulldoc_db.load_acquired_rows(src, spans, fulldoc_corpus.load_config(),
                                      A.cfg(served_acquired_max=30, served_acquired_max_bytes=1, served_scan_max=5))
    assert len(c.acquired) == 40
    s, _ = fulldoc_db.fetch_acquired_rows(lambda: A.FakeConn(plans), 1, "vera-dar-auto-acquire-v1", per_topic_max=None,
                                          topic_balanced=True, micro_cfg=A.cfg()["micro_topic"])
    assert len(s) == 40


# ---------------------------------------------------------------------------------------- acquisition soft ceiling
def test_acquisition_soft_ceiling_is_kept_on_rounds():
    b = acq_config.load_config()
    assert b["bounds"]["max_docs_per_prompt"] == 31 and b["inline"]["per_prompt_ceiling_enabled"] is True
    assert acq_config.inline_fresh_docs(b) == 31


def test_acquisition_stops_new_rounds_at_the_shipped_soft_ceiling(monkeypatch):
    """With the shipped 31-document per-prompt budget and an unreachable round target, no new search round starts once
    the budget is used (stop ceiling31); fresh documents never exceed it."""
    from tests.test_acq6_rounds_1103 import ENV, H, Q, cand, cfg
    from vera import inline_acquire as ia, stream_ctx
    monkeypatch.setattr(query_variants6, "variants", lambda q: [f"variant {i}" for i in range(60)])
    h = H(monkeypatch, [[cand(f"r{r}d{j}") for j in range(4)] for r in range(60)])
    a = cfg()
    a["bounds"]["acq_round_target_new_docs"] = 1000
    h.ctx = stream_ctx.StreamCtx(300.0, clock=h.clock)
    tok = stream_ctx.CTX.set(h.ctx)
    try:
        o = ia.acquire(Q, a, env=ENV, clock=h.clock, deps_factory=lambda x, e: h.deps, persist=True)
    finally:
        stream_ctx.CTX.reset(tok)
    assert o.acquisition["stop"] == "ceiling31"
    assert len(h.evaluated) <= 31 + 4 and len(h.searches) < 60


# ------------------------------------------------------------------------- rebuild cost (bit-identical efficiency fixes)
def test_tokenize_interns_tokens():
    import sys
    from vera.domain_identify import tokenize
    icfg = fulldoc_corpus._routing_cfg()["identification"]
    a, b = tokenize("Outbreak transmission incidence", icfg), tokenize("incidence of outbreak transmission", icfg)
    assert a and all(t is sys.intern(t) for t in a + b)


def test_single_profile_memo_gives_the_unmemoised_cosine():
    from vera.dar_corpus import DarCorpus
    from vera.domain_identify import _tf, tokenize
    dar = DarCorpus(fulldoc_corpus._routing_cfg())
    for p in plans100()[:3]:
        src, spans = A.rows_from_plans([p])
        for d in fulldoc_db.load_acquired_rows(src, spans, fulldoc_corpus.load_config(), A.cfg())._docs_for_dar:
            dar.ingest_document(d)
    ident = dar.identifier
    prof = list(ident._active().values())
    assert len(prof) == 1  # the single-profile branch is the one memoised
    texts = ["outbreak transmission incidence pathogen", "the incidence of infection during an epidemic", "zzz qqq"]
    for t in texts:
        tf = _tf(tokenize(t, ident.icfg))
        want = ident._cos(tf, prof[0]) if tf else 0.0
        got = [x["score"] for x in ident.classify(t, "chunk")]
        assert (got[0] if got else 0.0) == want or (not got and want < ident.icfg["chunk_min_sim"])
        assert ident._dcos(tf, prof[0], ident._dmemo) == want if tf else True


# ------------------------------------------------- append refresh, compact index, registry indexes (NOCAP-1109 part 2)
QS = ("pathogen transmission incidence outbreak",
      "outbreak incidence surveillance cohort", "What limits predictability of forecasts?")
MARK = "vera-dar-auto-acquire-v1"


def _fp(c):
    """Everything a reader can observe of an acquired corpus (bit-level: floats via repr)."""
    import json
    dar = c._dar
    ev = c._evidence_dar()
    out = {"acq": [a["document_id"] for a in c.acquired], "unusable": c.unusable, "stamps": c.stamps,
           "pins": sorted(c.pin_candidates), "chunks": [(ch["span_id"], sorted(ch["tf"].items()), ch["len"]) for ch in dar._chunks],
           "df": sorted(dar._df.items()), "tl": dar._total_len, "ev": [len(ev._chunks), ev._total_len, sorted(ev._df.items())],
           "cov": dar.registry.coverage(), "doms": [(ch["span_id"], [d.domain_id for d in dar.registry.domains_for_chunk(ch["span_id"])])
                                                     for ch in dar._chunks],
           "r": [c.retrieve(q, 12) for q in QS], "re": [c.retrieve(q, 12, evidence_only=True) for q in QS],
           "raw": [dar.retrieve(q, 16, max_per_doc=8) for q in QS], "insp": c.inspection()}
    return json.dumps(out, sort_keys=True, default=repr)


def _db(plans):
    conn = A.FakeConn(plans)
    return lambda: conn


def _sig(plans):
    return fulldoc_db.acquired_signature(_db(plans), 1, MARK)


def test_append_refresh_equals_full_build():
    plans = plans100()
    fd, acq = fulldoc_corpus.load_config(), A.cfg()
    full = fulldoc_db.load_acquired_db(_db(plans), 1, fd, acq)
    old = fulldoc_db.load_acquired_db(_db(plans[:90]), 1, fd, acq)
    before = _fp(old)
    new = fulldoc_db.extend_acquired_db(_db(plans), 1, fd, acq, old, _sig(plans[:90]), _sig(plans))
    assert new is not None and new is not old and len(new.acquired) == N
    assert _fp(new) == _fp(full)  # append == full build, bit for bit
    assert _fp(old) == before  # the serving corpus never changed (shared postings are read below its own chunk count)
    again = fulldoc_db.extend_acquired_db(_db(plans), 1, fd, acq, new, _sig(plans), _sig(plans))
    assert again is None  # nothing newer: not an append


def test_non_append_change_is_refused_for_full_rebuild():
    plans = plans100()
    fd, acq = fulldoc_corpus.load_config(), A.cfg()
    old = fulldoc_db.load_acquired_db(_db(plans[:50]), 1, fd, acq)
    conn = A.FakeConn(plans[:60])
    conn.sources = [x for x in conn.sources if x["source_id"] != 1005]  # one old row removed, ten added
    new_sig = fulldoc_db.acquired_signature(lambda: conn, 1, MARK)
    assert new_sig == (1060, 59)
    assert fulldoc_db.extend_acquired_db(lambda: conn, 1, fd, acq, old, _sig(plans[:50]), new_sig) is None
    legacy = fulldoc_db.load_acquired_rows(*A.rows_from_plans(plans[:5]), fd, acq)  # in-memory order: not appendable
    assert fulldoc_db.extend_acquired_db(_db(plans[:6]), 1, fd, acq, legacy, _sig(plans[:5]), _sig(plans[:6])) is None


def test_failed_append_is_discarded_and_leaves_no_trace(monkeypatch):
    from vera import dar_corpus
    plans = plans100()
    fd, acq = fulldoc_corpus.load_config(), A.cfg()
    old = fulldoc_db.load_acquired_db(_db(plans[:80]), 1, fd, acq)
    before = _fp(old)
    real, n = dar_corpus.DarCorpus.ingest_document, {"i": 0}

    def boom(self, d):
        n["i"] += 1
        if n["i"] == 3:
            raise RuntimeError("injected")
        return real(self, d)
    monkeypatch.setattr(dar_corpus.DarCorpus, "ingest_document", boom)
    assert fulldoc_db.extend_acquired_db(_db(plans), 1, fd, acq, old, _sig(plans[:80]), _sig(plans)) is None
    monkeypatch.setattr(dar_corpus.DarCorpus, "ingest_document", real)
    assert _fp(old) == before
    new = fulldoc_db.extend_acquired_db(_db(plans), 1, fd, acq, old, _sig(plans[:80]), _sig(plans))
    assert _fp(new) == _fp(fulldoc_db.load_acquired_db(_db(plans), 1, fd, acq))


def test_full_rebuild_releases_the_serving_corpus_first(monkeypatch):
    plans = plans100()[:20]
    fd = fulldoc_corpus.load_config()
    fulldoc_corpus.reset_acquired_for_tests()
    fulldoc_corpus._ACQ.update(corpus=object(), sig=(None, 0))  # a non-appendable old state
    seen = []
    real = fulldoc_db.load_acquired_db

    def spy(*a, **k):
        seen.append(fulldoc_corpus._ACQ["corpus"])
        return real(*a, **k)
    monkeypatch.setattr(fulldoc_db, "load_acquired_db", spy)
    fulldoc_corpus._ACQ["building"] = True
    fulldoc_corpus._rebuild_acquired(fd, (A.cfg(), _db(plans), 1), _sig(plans))
    assert seen == [None]  # old corpus dropped before the full build: never two full copies
    assert len(fulldoc_corpus._ACQ["corpus"].acquired) == 20 and fulldoc_corpus._ACQ["building"] is False
    fulldoc_corpus.reset_acquired_for_tests()


def test_compact_chunk_records_hold_no_text_or_term_map():
    c = fulldoc_db.load_acquired_db(_db(plans100()[:3]), 1, fulldoc_corpus.load_config(), A.cfg())
    ch = c._dar._chunks[0]
    assert not hasattr(ch, "__dict__") and set(ch.__slots__) == {"doc", "start", "end", "len", "idx", "_icfg"}
    assert ch["text"] == c._dar._docs[ch["source_key"]]["text"][ch["start"]:ch["end"]]
    assert c._dar._by_span[ch["span_id"]] is ch and ch["span_id"] in c._dar._by_span
    assert c._dar._by_span.get(ch["span_id"] + "0") is None and c._dar._by_span.get("x:1-2") is None


def test_registry_indexes_equal_full_scans():
    import random
    from vera.domain_registry import InMemoryDomainRegistry
    reg = InMemoryDomainRegistry(clock=lambda: "t")
    ids = [reg.register(f"dom {i}")[0].domain_id for i in range(6)]
    rng = random.Random(3)
    for _ in range(400):
        d, a = rng.choice(ids), f"k{rng.randrange(40)}:{rng.randrange(9)}-1"
        if rng.random() < 0.8:
            reg.associate_chunk(d, a, "x")
            reg.associate_corpus(d, a.split(":")[0], "x")
        else:
            reg.disassociate_chunk(d, a)
            reg.disassociate_corpus(d, a.split(":")[0])
    spans = {s for _, s in reg._chunk} | {"absent"}
    for s in spans:
        assert reg.domains_for_chunk(s) == [reg._by_id[d] for d, x in sorted(reg._chunk) if x == s]
    for k in {c for _, c in reg._corpus} | {"absent"}:
        assert reg.domains_for_corpus(k) == [reg._by_id[d] for d, c in sorted(reg._corpus) if c == k]
    for d in ids:
        assert reg.chunks_for_domain(d) == sorted(s for x, s in reg._chunk if x == d)
        assert reg.corpora_for_domain(d) == sorted(c for x, c in reg._corpus if x == d)


def test_domain_routed_retrieve_is_not_quadratic():
    import time
    c = fulldoc_db.load_acquired_db(_db(plans100()), 1, fulldoc_corpus.load_config(), A.cfg())
    t = time.perf_counter()
    r = c._dar.retrieve(QS[0], 16, max_per_doc=8)
    assert r["trace"]["question_domains"] and r["candidates"]  # a domain branch ran
    assert time.perf_counter() - t < 1.0


# ------------------------------------------------------------------------------- verifier findings F1, F2 (2026-11-09)
def test_f1_append_abandoned_before_swap_is_discarded(monkeypatch):
    """verify/orphan.py: an append that succeeds but is abandoned before the swap (here the evidence-view pre-warm
    raises) must take its postings back out, or the retry from the same serving corpus double-counts them."""
    plans = plans100()[:40]
    fd, acq = fulldoc_corpus.load_config(), A.cfg()
    assert (fd.get("retrieval") or {}).get("evidence_only")  # _rebuild_acquired pre-warms the evidence view
    want = _fp(fulldoc_db.load_acquired_db(_db(plans), 1, fd, acq))
    old = fulldoc_db.load_acquired_db(_db(plans[:30]), 1, fd, acq)
    before = _fp(old)
    fulldoc_corpus.reset_acquired_for_tests()
    fulldoc_corpus._ACQ.update(corpus=old, sig=old._acq_sig, building=True)
    real = fulldoc_corpus.FullDocCorpus._evidence_dar

    def boom(self):
        if self is not old:
            raise RuntimeError("injected after a successful append")
        return real(self)
    monkeypatch.setattr(fulldoc_corpus.FullDocCorpus, "_evidence_dar", boom)
    with pytest.raises(RuntimeError):
        fulldoc_corpus._rebuild_acquired(fd, (acq, _db(plans), 1), _sig(plans))
    monkeypatch.setattr(fulldoc_corpus.FullDocCorpus, "_evidence_dar", real)
    assert fulldoc_corpus._ACQ["corpus"] is old and fulldoc_corpus._ACQ["building"] is False
    assert _fp(old) == before  # the serving corpus is untouched
    fulldoc_corpus._ACQ["building"] = True
    fulldoc_corpus._rebuild_acquired(fd, (acq, _db(plans), 1), _sig(plans))  # the retry
    assert _fp(fulldoc_corpus._ACQ["corpus"]) == want  # no duplicated postings: equals a full build
    fulldoc_corpus.reset_acquired_for_tests()


def test_f2_rows_committed_after_the_probe_still_append():
    """A row committed between the refresh probe and the listing no longer forces a full rebuild: the count and the
    newer rows come from one snapshot, and the corpus records the signature it was really built at."""
    plans = plans100()
    fd, acq = fulldoc_corpus.load_config(), A.cfg()
    old = fulldoc_db.load_acquired_db(_db(plans[:50]), 1, fd, acq)
    assert old._acq_sig == (1050, 50)
    probe = _sig(plans[:55])  # the probe saw 55 rows; 5 more were committed before the listing
    new = fulldoc_db.extend_acquired_db(_db(plans[:60]), 1, fd, acq, old, old._acq_sig, probe)
    assert new is not None and len(new.acquired) == 60 and new._acq_sig == (1060, 60)
    assert _fp(new) == _fp(fulldoc_db.load_acquired_db(_db(plans[:60]), 1, fd, acq))
    nxt = fulldoc_db.extend_acquired_db(_db(plans[:62]), 1, fd, acq, new, new._acq_sig, _sig(plans[:62]))
    assert nxt is not None and len(nxt.acquired) == 62 and not any(u["code"] == fulldoc_db.R_DUP for u in nxt.unusable)
