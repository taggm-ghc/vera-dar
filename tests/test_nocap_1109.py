"""NOCAP-1109 (owner decision OD-1109-NOCAP, 2026-11-09, verbatim: "the 30-document cieling is for when to stop
additional runs of searchs for new documents to ingest. there should be no cap upon stored docs and chunks to accept
from internal search results"). Offline: synthetic fixtures, fake DB-API connection, stubbed search; no network, DB or
model. Proves: (1) every stored acquired document is eligible for internal search (no document, byte or scan cap and
no R-ACQ-SERVECAP drop); (2) the ~30-document soft ceiling still stops acquisition rounds."""
from __future__ import annotations

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
