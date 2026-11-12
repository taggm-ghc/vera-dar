"""WPQ-CYCLE2-1111 (WP-G), OFFLINE: closed-form attestation of open compounds. c2 reads the in-memory BM25 postings of the
loaded corpora (real DarCorpus objects, real postings shape); c3 reads the request's own round-1 titles (real record shapes:
title, doi.org or arXiv urls). Plus the G4 hyphen rule and the 4.2 junk guard. No network, no DB. Tests G-1 to G-22 of
work/WPQ-CYCLE2-1111/PLAN.md. Everything ships OFF (max_pairs 0); these tests turn it on explicitly."""
import copy
import json
import threading
from pathlib import Path

import pytest

from tests import acq_support as A
from vera import fulldoc_corpus, fulldoc_db, query_variants_acq as qv, search_providers as sp, search_query as sq
from vera.dar_corpus import DarCorpus
from vera.fulldoc_corpus import STATUS_OK, FullDocCorpus, _routing_cfg

ROOT = Path(__file__).resolve().parent.parent
QT = json.loads((ROOT / "config" / "vera_eval_run.json").read_text())["search"]["query_terms"]
PROFILE = A.cfg()["search"]["openalex_profile"]
SS = '("super strings" OR superstrings)'
SSH = '("super-string" OR "super string" OR superstring)'
QS = ["What are super strings?", "what is super strings", "super-string theory?", "multi-agent orchestration", "M-theory",
      "What is VERA-DAR?", "state-of-the-art methods", "super market prices", "data base design", "baked goods",
      "guitar super", "albert einstein", "nano particles", "neuro science", "latest developments in agents",
      "How do agents affect forecasting?", "super string quartet", "super strings dark energy"]


def _cfg(monkeypatch, n=1, mode="corpus", min_docs=2, req=2):
    monkeypatch.setattr(sq, "open_compound_max_pairs", lambda: n)
    monkeypatch.setattr(sq, "_oc_attest_cfg", lambda: (mode, min_docs, req))


def _dar(stem=True):
    cfg = copy.deepcopy(_routing_cfg())
    cfg["corpus"]["bm25_stem"] = stem
    return DarCorpus(cfg)


def _doc(key, text):
    return {"source_key": key, "url": key, "doi": None, "title": key, "published": "2026-01-01",
            "source_type": "preprint", "retrieved_at": "2026-11-09", "text": text,
            "stored_text_kind": "full_text", "admission_passed": True, "db_source_id": None,
            "db_canonical_source_id": None}


def _dar_with(docs, stem=True):
    c = _dar(stem)
    for k, t in docs.items():
        assert c.ingest_document(_doc(k, t))["status"] == "indexed"
    return c


def _fd(docs, stem=True):
    return FullDocCorpus(STATUS_OK, [], [], {}, _dar_with(docs, stem), {}, {})


def _use(monkeypatch, *corpora):
    monkeypatch.setattr(sq, "_loaded_corpora", lambda: tuple(corpora))


FILL = "The theory describes vibrating objects and how they move through many dimensions of space. "


def _text(word):
    return (f"Notes on {word} and its properties. " + FILL) * 3


def _all(q):
    return {"ask": sp.build_openalex_search(q, QT)[0], "relaxed": sp.build_openalex_search(q, QT, relaxed=True)[0],
            "acq": sp.build_openalex_search(q, QT, profile=PROFILE)[0], "arxiv": sp.build_arxiv_query(q, {}, QT),
            "variants": qv.variants(q, terms_cfg=QT)}


def _control(monkeypatch, q):
    with monkeypatch.context() as m:
        m.setattr(sq, "open_compound_max_pairs", lambda: 0)
        return _all(q)


# --- G-1 ... G-4: c2, every query shape
def test_g1_attested_in_two_documents_every_shape(monkeypatch):
    _cfg(monkeypatch)
    _use(monkeypatch, _fd({"d1": _text("superstrings"), "d2": _text("superstring")}))
    r = _all("What are super strings?")
    assert r["ask"] == SS and r["relaxed"] == SS and r["acq"] == SS
    assert 'all:"super strings"' in r["arxiv"] and "all:superstrings" in r["arxiv"] and "all:super " not in r["arxiv"]
    assert not any(v in ("super", "strings") for v in r["variants"])


def test_g2_one_document_is_the_control(monkeypatch):
    _cfg(monkeypatch)
    _use(monkeypatch, _fd({"d1": _text("superstrings"), "d2": _text("unrelated")}))
    assert _all("What are super strings?") == _control(monkeypatch, "What are super strings?")


def test_g3_two_chunks_of_one_document_do_not_attest(monkeypatch):
    _cfg(monkeypatch)
    big = " ".join(_text("superstrings") for _ in range(6))
    c = _fd({"d1": big})
    assert len(c._dar._chunks) >= 3 and c._dar._df.get("superstring", 0) >= 2  # chunk count would pass, document count not
    _use(monkeypatch, c)
    assert _all("What are super strings?") == _control(monkeypatch, "What are super strings?")


def test_g4_corpus_not_loaded_is_the_control_for_ask_and_acquisition(monkeypatch):
    _cfg(monkeypatch)
    fulldoc_corpus.reset_for_tests()
    assert fulldoc_corpus.peek_loaded_corpora() == ()
    for q in QS:
        assert _all(q) == _control(monkeypatch, q)
    kc = sq.key_concepts("What are super strings?", pairs=True)
    assert kc == _kc_control(monkeypatch, "What are super strings?")
    assert sq.build_query_terms("What are super strings?", QT).pairs == ()


def _kc_control(monkeypatch, q):
    with monkeypatch.context() as m:
        m.setattr(sq, "open_compound_max_pairs", lambda: 0)
        return sq.key_concepts(q, pairs=True)


# --- G-5 ... G-7: hyphen rule
def test_g5_hyphen_attested_and_unattested(monkeypatch):
    _cfg(monkeypatch)
    _use(monkeypatch, _fd({"d1": _text("superstring"), "d2": _text("superstrings")}))
    ask = sp.build_openalex_search("super-string theory?", QT)[0]
    assert SSH in ask and "theory" in ask
    assert sq.build_query_terms("super-string theory?", QT).pairs == (("super", "string"),)
    _use(monkeypatch, _fd({"d1": _text("superstring")}))
    assert _all("super-string theory?") == _control(monkeypatch, "super-string theory?")


def test_g6_configured_and_short_hyphen_forms_are_byte_identical(monkeypatch):
    _cfg(monkeypatch)
    _use(monkeypatch, _fd({"d1": _text("veradar"), "d2": _text("veradar")}))
    for q in ("multi-agent orchestration", "M-theory", "What is VERA-DAR?"):
        assert _all(q) == _control(monkeypatch, q), q
    monkeypatch.setattr(sq, "_attested", lambda a, b: True)  # even if everything were attested, the candidate rules hold
    for q in ("multi-agent orchestration", "M-theory"):
        assert _all(q) == _control(monkeypatch, q), q


def test_g7_two_hyphens_unchanged(monkeypatch):
    _cfg(monkeypatch)
    monkeypatch.setattr(sq, "_attested", lambda a, b: True)
    for q in ("state-of-the-art methods", "alpha-beta-gamma theory"):
        assert _all(q) == _control(monkeypatch, q), q


# --- G-8 ... G-10: junk guard
def test_g8_cycle1_junk_is_absent(monkeypatch):
    _cfg(monkeypatch)
    _use(monkeypatch, _fd({"d1": _text("alberteinstein"), "d2": _text("unrelated")}))  # alberteinstein: exactly 1 document
    junk = ("bakedgoods", "veradar", "veracome", "guitarsuper", "alberteinstein", "latestdevelopments")
    for q in ("baked goods", "vera dar", "vera come", "guitar super", "albert einstein", "latest developments"):
        assert not any(j in str(_all(q)) for j in junk), q


def test_g9_stopword_and_generic_parts_never_join(monkeypatch):
    g = sq._guard_generic()
    assert "with" in g and "data" in g and "study" in g
    assert not sq._guard_ok(("with", "market"), g) and not sq._guard_ok(("data", "base"), g)
    assert not sq._guard_ok(("a", "bcd"), g) and not sq._guard_ok(("abcdefghijkl", "mnopqrstuvwxyz"), g)  # 6-24 letters
    assert sq._guard_ok(("super", "market"), g)
    _cfg(monkeypatch)
    monkeypatch.setattr(sq, "_attested", lambda a, b: True)
    assert _all("the market") == _control(monkeypatch, "the market")
    assert _all("study group") == _control(monkeypatch, "study group")


def test_g10_lookalikes_keep_the_phrase_with_the_closed_form(monkeypatch):
    _cfg(monkeypatch)
    _use(monkeypatch, _fd({"d1": _text("supermarket"), "d2": _text("supermarkets"), "d3": _text("database"),
                           "d4": _text("databases")}))
    assert '("super market" OR supermarket)' in sp.build_openalex_search("super market prices", QT)[0]
    # "data" is a generic scholarly term of config/chunk_topic_terms.json, so the guard (4.2 item 2) wins over attestation
    assert _all("data base design") == _control(monkeypatch, "data base design")


# --- G-11: bounded work
def test_g11_attestation_work_is_bounded(monkeypatch):
    _cfg(monkeypatch)
    calls = []
    monkeypatch.setattr(sq, "_corpus_attests", lambda a, b, n: calls.append((a, b)) or False)
    qt = sq.build_query_terms("kappa lambda omicron sigma tau upsilon", QT)
    assert qt.pairs == () and 1 <= len(calls) <= 2  # max_pairs + 1
    calls.clear()
    monkeypatch.setattr(sq, "open_compound_max_pairs", lambda: 3)
    sq.build_query_terms("kappa lambda omicron sigma tau upsilon", QT)
    assert len(calls) <= 4


# --- G-12: shipped config is OFF and byte-identical
def test_g12_shipped_config_is_off_and_every_mode_is_inert_at_zero(monkeypatch):
    assert sq.open_compound_max_pairs() == 0
    assert sq._oc_attest_cfg() == ("corpus", 2, 2)
    base = {q: _all(q) for q in QS}
    for mode in ("corpus", "request", "corpus_or_request"):
        _cfg(monkeypatch, n=0, mode=mode)
        _use(monkeypatch, _fd({"d1": _text("superstrings"), "d2": _text("superstrings")}))
        att = sq.RequestAttest()
        att.compute([{"title": "Superstrings I", "url": "https://doi.org/10.1/a"},
                     {"title": "Superstrings II", "url": "https://doi.org/10.1/b"}])
        with sq.request_attest_scope(att):
            assert {q: _all(q) for q in QS} == base


def test_g12b_invalid_attest_config_takes_defaults(tmp_path, monkeypatch, caplog):
    p = tmp_path / "query_terms.json"
    p.write_text('{"open_compound": {"max_pairs": 1, "mode": "weird", "min_docs": 0, "request_min_records": true}}')
    monkeypatch.setattr(sq, "_QT_PATH", p)
    sq._oc_attest_cfg.cache_clear()
    try:
        with caplog.at_level("WARNING"):
            assert sq._oc_attest_cfg() == ("corpus", 2, 2)
        assert sum("open_compound" in r.message for r in caplog.records) == 3
    finally:
        sq._oc_attest_cfg.cache_clear()


# --- G-13: zero DB calls from the attestation path
def test_g13_no_db_call_cold_or_warm(monkeypatch):
    _cfg(monkeypatch)
    hits = []

    def boom(*a, **k):
        hits.append(1)
        raise AssertionError("DB / loader reached from the attestation path")
    for name in ("default_connect", "cached_load", "load_db", "load_acquired_db", "extend_acquired_db", "acquired_signature"):
        monkeypatch.setattr(fulldoc_db, name, boom)
    for name in ("get_corpus", "get_acquired_corpus", "refresh_acquired"):
        monkeypatch.setattr(fulldoc_corpus, name, boom)
    fulldoc_corpus.reset_for_tests()
    assert _all("What are super strings?") == _control(monkeypatch, "What are super strings?")  # cold
    base = _fd({"d1": _text("superstrings")})
    acq = _fd({"d2": _text("superstring")})
    fulldoc_corpus._CACHE[("db", "1", "{}")] = base
    fulldoc_corpus._ACQ["corpus"] = acq
    try:
        assert _all("What are super strings?")["ask"] == SS  # warm: one document in each corpus, unioned
    finally:
        fulldoc_corpus.reset_for_tests()
    assert hits == []


# --- G-14: stem keys, versions, masks, union
def test_g14_singular_and_plural_corpora_attest_either_form(monkeypatch):
    _cfg(monkeypatch)
    sing = _fd({"d1": _text("superstring"), "d2": _text("superstring")})
    plur = _fd({"d1": _text("superstrings"), "d2": _text("superstrings")})
    for corp in (sing, plur):
        _use(monkeypatch, corp)
        assert _all("What are super strings?")["ask"] == SS
        assert _all("What is a super string?")["ask"] == '("super string" OR superstring)'


def test_g14_unstemmed_index_keys(monkeypatch):
    c = _dar_with({"d1": _text("superstrings"), "d2": _text("superstring")}, stem=False)
    assert c.closed_form_docs("super", "strings") == {"d1", "d2"}


def test_g14_fork_postings_and_evidence_mask_are_not_counted():
    c = _dar_with({"d1": _text("superstrings")})
    fork = c.fork()
    assert fork.ingest_document(_doc("d2", _text("superstrings")))["status"] == "indexed"
    assert fork.closed_form_docs("super", "strings") == {"d1", "d2"}
    assert c.closed_form_docs("super", "strings") == {"d1"}  # shared append-only postings: entries past this version skipped
    view = fork.evidence_view(lambda ch: ch["source_key"] == "d2")
    assert view.closed_form_docs("super", "strings") == {"d2"}


def test_g14_union_dedupes_by_source_key(monkeypatch):
    _cfg(monkeypatch)
    _use(monkeypatch, _fd({"same": _text("superstrings")}), _fd({"same": _text("superstrings")}))
    assert _all("What are super strings?") == _control(monkeypatch, "What are super strings?")


# --- c3 -----------------------------------------------------------------------------------------------------------
def _rec(title, n):
    return {"title": title, "url": f"https://doi.org/10.1000/ss.{n}", "doi": f"10.1000/ss.{n}", "snippet": "superstrings"}


R1 = [_rec("Superstrings: a review", 1), _rec("An introduction to superstring theory", 2), _rec("Dark energy surveys", 3)]


def _scope(items):
    att = sq.RequestAttest()
    att.compute(items)
    return att


# The 47 questions of the cycle-1 pool (work/WPQ-PDSA-1111/pool-v2.json), questions only, in pool order.
POOL_QS = [
    "What are super strings?", "superstrings", "what are superstrings", "super-string theory?", "What is Vera dar magops?",
    "from where did vera come?", "What is VERA-DAR?", "How accurate are epidemic forecasting models?",
    "What are the latest findings on lunar lava tube habitat engineering?", "Is aloe vera good for burns?",
    "Who was Vera Brittain?", "where did it come from?", "What is the capital of France?",
    "Why did Albert Einstein win the Nobel Prize for his theory of relativity?",
    "What are the latest developments in epidemic forecasting?", "Tell me about models", "What is vrea?",
    "what is vare dar?", "What are hadrons?", "What are quarks?", "What are leptons?", "What are components of RAG?",
    "What are implications of the US 2026 Midterm elections? Keep searching until you find an answer",
    "what are nano particles?", "what is neuro science", "post office strings", "data base design", "super market prices",
    "life style changes", "super string quartet", "super bowl sponsorship", "guitar super slinky strings",
    "super strings dark energy", "climate change effects", "what did Einstein say about super strings",
    "multi-agent orchestration", "midterm grades", "string theory", "M-theory", "what are quarks?", "What are photons?",
    "What are pigments?", "What are chords?", "What are sonnets?", "What are baked goods?", "What are super strings?",
    "What is a super string?"]


def test_g15_round1_is_the_control_for_every_question(monkeypatch):
    assert len(POOL_QS) == 47
    _cfg(monkeypatch, mode="request")
    base = {q: _control(monkeypatch, q) for q in POOL_QS}
    base_kc = {q: _kc_control(monkeypatch, q) for q in POOL_QS}
    with sq.request_attest_scope():  # round 1: the holder is open but empty
        assert {q: _all(q) for q in POOL_QS} == base
        assert {q: sq.key_concepts(q, pairs=True) for q in POOL_QS} == base_kc
        assert sq.request_attest_active()


def test_g16_threshold_distinct_records_and_titles(monkeypatch):
    two = _scope(R1)
    assert two.attests("super", "strings", 2) and two.attests("super", "string", 2)
    assert not _scope(R1[:1] + R1[2:]).attests("super", "strings", 2)  # one record
    same_work = [_rec("Superstrings: a review", 1), {"title": "Superstrings: a review", "url": "https://arxiv.org/abs/9",
                                                    "doi": "10.9999/other"}]
    assert not _scope(same_work).attests("super", "strings", 2)  # same normalised title
    dup_key = [_rec("Superstrings: a review", 1), _rec("Superstrings reviewed again", 1)]
    assert not _scope(dup_key).attests("super", "strings", 2)  # same DOI from two providers
    assert not _scope([{"title": "Superstrings", "url": "https://x.org/a"}]).attests("super", "strings", 2)


def test_g17_title_tokens_and_plural_rule():
    assert not _scope([_rec("Super-string theory", 1), _rec("super strings", 2)]).attests("super", "strings", 2)
    assert _scope([_rec("superstring a", 1), _rec("superstring b", 2)]).attests("super", "strings", 2)
    assert _scope([_rec("superstrings a", 1), _rec("superstrings b", 2)]).attests("super", "string", 2)
    assert not _scope([_rec("superstringent a", 1), _rec("superstringent b", 2)]).attests("super", "strings", 2)
    assert not _scope([{"url": "https://x.org/1", "snippet": "superstrings"}, {"url": "https://x.org/2"}]).attests("super", "strings", 1)


def test_g18_reranking_gets_the_pair_only_when_attested(monkeypatch):
    _cfg(monkeypatch, mode="request")
    q = "What are super strings?"
    texts = ["Superstrings and the unification of forces in physics", "Strings attached: super bowl guitar strings review"]
    ctrl = sq.relevance_tiers(_kc_control(monkeypatch, q), texts)
    with sq.request_attest_scope(_scope([_rec("x superstring", 1)])):  # one record: not attested
        assert sq.relevance_tiers(sq.key_concepts(q, pairs=True), texts) == ctrl
    with sq.request_attest_scope(_scope(R1)):
        kc = sq.key_concepts(q, pairs=True)
        assert "super-strings" in kc[0] and kc[1]["super-strings"] == 2
        assert sq.relevance_tiers(kc, texts)[0] == 0 and sq.relevance_tiers(kc, texts)[1] == 1


def test_g19_later_rounds_render_the_pair(monkeypatch):
    _cfg(monkeypatch, mode="request")
    q = "What are super strings?"
    with sq.request_attest_scope(_scope(R1)):
        assert _all(q)["ask"] == SS and _all(q)["acq"] == SS
        assert not any(v in ("super", "strings") for v in qv.variants(q, terms_cfg=QT))
    with sq.request_attest_scope(_scope([_rec("x", 1)])):
        assert _all(q) == _control(monkeypatch, q)


def test_g20_nothing_outlives_the_request(monkeypatch):
    _cfg(monkeypatch, mode="request")
    q = "What are super strings?"
    seen = {}
    with sq.request_attest_scope(_scope(R1)) as att:
        t = threading.Thread(target=lambda: seen.update(bg=_all(q)))  # a fresh context: the hand-off job, research_loop
        t.start()
        t.join()
        assert att.computed and sq.current_request_attest() is att
    assert seen["bg"] == _control(monkeypatch, q)
    assert sq.current_request_attest() is None and _all(q) == _control(monkeypatch, q)
    with sq.request_attest_scope():  # a second request: new empty holder, nothing carried over
        assert not sq.current_request_attest().computed and _all(q) == _control(monkeypatch, q)
    held = [v for v in vars(sq).values() if isinstance(v, sq.RequestAttest)]
    assert held == []


def test_g20b_compute_once_and_immutable():
    att = sq.RequestAttest()
    assert att.compute(R1) is True and att.compute([_rec("Other", 9)]) is False
    assert att.attests("super", "strings", 2)


def test_g21_titles_create_no_candidates_and_the_guard_still_applies(monkeypatch):
    _cfg(monkeypatch, mode="request")
    bake = [_rec("Bakedgoods economics", 1), _rec("Bakedgoods trade", 2)]
    with sq.request_attest_scope(_scope(bake)):
        assert _all("bake goods") == _control(monkeypatch, "bake goods")
        assert sq.build_query_terms("zeta", QT).pairs == ()  # titles cannot create a pair
    dbase = [_rec("Databases for science", 1), _rec("Database systems", 2)]
    with sq.request_attest_scope(_scope(dbase)):
        assert _all("data base design") == _control(monkeypatch, "data base design")  # generic part: guard rejects


def test_g22_trace_records_only_a_code_and_a_count(monkeypatch):
    from vera import request_trace as rt
    got = []
    monkeypatch.setattr(rt, "event", lambda *a, **k: got.append((a, k)))
    att = _scope(R1)
    assert att.attests("super", "strings", 2) and att.attests("super", "strings", 2)
    assert got == [(("acquire", "pair_attested_request"), {"counts": {"records": 2}})]
    assert "uper" not in repr(got) and "review" not in repr(got)


def test_c2_or_c3_mode_either_source_attests(monkeypatch):
    _cfg(monkeypatch, mode="corpus_or_request")
    q = "What are super strings?"
    assert _all(q) == _control(monkeypatch, q)  # neither source
    with sq.request_attest_scope(_scope(R1)):
        assert _all(q)["ask"] == SS
    _use(monkeypatch, _fd({"d1": _text("superstrings"), "d2": _text("superstrings")}))
    assert _all(q)["ask"] == SS


# --- D1 (verifier): the inline_acquire wiring, end to end with the offline fakes of tests/acq_support ----------------
def _wiring(monkeypatch, cands_for, discovered=None):
    """Runs inline_acquire.acquire (mode request, max_pairs 1) with fake providers; returns (log, outcome). `cands_for(n)` =
    the candidates the n-th search call returns. The log holds, per search call and per rank_candidates call, what the
    request scope and the provider query builder saw at that moment."""
    from vera import acq_config, auto_acquire, inline_acquire
    _cfg(monkeypatch, mode="request")
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    q = "What are super strings?"
    log = []

    def search(question, n, t):
        att = sq.current_request_attest()
        log.append(("search", None if att is None else att.computed, None if att is None or not att.computed else len(att._recs),
                    sp.build_openalex_search(q, QT)[0]))
        return list(cands_for(sum(1 for e in log if e[0] == "search") - 1))
    orig = auto_acquire.rank_candidates

    def rank(question, *a, **k):
        att = sq.current_request_attest()
        log.append(("rank", None if att is None else att.computed, None if att is None or not att.computed else len(att._recs),
                    sq.key_concepts(q, pairs=True)[0]))
        return orig(question, *a, **k)
    monkeypatch.setattr(auto_acquire, "rank_candidates", rank)
    d = auto_acquire.Deps(search=search, http_get=A.Web({}, {}), lookup=A.Lookup(), store=A.Store(),
                          corpus_keys=lambda: set(), url_check=lambda u: (True, ""), refresh=None)
    cfg = A.cfg()
    cfg.setdefault("inline", {})["use_discovered"] = True
    kw = {} if discovered is None else {"discovered": discovered}
    try:
        o = inline_acquire.acquire(q, cfg, env=A.ENV_ON, deps_factory=lambda a, e: d, **kw)
    finally:
        auto_acquire.reset_for_tests()
        inline_acquire.reset_for_tests()
    return log, o


def _c(i, title):
    return A.cand(f"10.9999/ss.{i}", title=title)


def test_d1_wiring_round1_is_control_attestation_computed_once_before_reranking(monkeypatch):
    ctrl = sp.build_openalex_search("What are super strings?", QT)[0]
    r1 = [_c(1, "Superstrings and pathogen transmission"), _c(2, "Superstring outbreak incidence models"),
          _c(3, "Superstrings and pathogen transmission")]  # #3: same normalised title as #1, other DOI: counts once (R5-D1)
    log, o = _wiring(monkeypatch, lambda n: r1)
    searches = [e for e in log if e[0] == "search"]
    ranks = [e for e in log if e[0] == "rank"]
    assert searches and ranks and o.status
    assert searches[0][1] is False and searches[0][3] == ctrl  # round 1: holder open, not computed, control query
    assert ranks[0][1] is True and ranks[0][2] == 2  # computed before re-ranking, from 2 distinct records
    assert "super-strings" in ranks[0][3]  # re-ranking sees the attested pair as one concept
    assert all(e[2] == 2 for e in log if e[1])  # never recomputed: the record count is the same in every later call
    for e in searches[1:]:  # later rounds, when the variant loop ran, get the pair
        assert e[3] != ctrl and "superstrings" in e[3]
    assert sq.current_request_attest() is None  # nothing outlives the request


def test_d1_wiring_one_record_does_not_attest(monkeypatch):
    ctrl = sp.build_openalex_search("What are super strings?", QT)[0]
    log, _o = _wiring(monkeypatch, lambda n: [_c(1, "Superstrings and pathogen transmission"), _c(2, "Unrelated outbreak models")])
    assert all(e[3] == ctrl for e in log if e[0] == "search")
    assert all("super-strings" not in e[3] for e in log if e[0] == "rank")


def test_d1_wiring_discovered_results_attest_once_and_are_not_recomputed(monkeypatch):
    ctrl = sp.build_openalex_search("What are super strings?", QT)[0]
    disc = [{"n": 1, "title": "Superstrings: a review", "url": "https://arxiv.org/abs/2401.00001", "provider": "arxiv"},
            {"n": 2, "title": "Superstring theory primer", "url": "https://arxiv.org/abs/2401.00002", "provider": "arxiv"}]
    other = [_c(7, "Unrelated dark energy"), _c(8, "Another unrelated one"), _c(9, "A third unrelated one")]
    log, _o = _wiring(monkeypatch, lambda n: other, discovered=disc)
    searches = [e for e in log if e[0] == "search"]
    assert searches
    assert searches[0][1] is True and searches[0][2] == 2 and searches[0][3] != ctrl  # computed from the discovery titles
    assert all(e[2] == 2 for e in log)  # the inline candidates (3 more records) never change it
    assert sq.current_request_attest() is None


def test_d1_wiring_no_leak_to_fresh_threads_or_pools(monkeypatch):
    import concurrent.futures as cf
    import contextvars
    _cfg(monkeypatch, mode="request")
    q = "What are super strings?"
    ctrl = sp.build_openalex_search(q, QT)[0]
    seen = {}
    pool = cf.ThreadPoolExecutor(2)
    with sq.request_attest_scope(_scope(R1)):
        inside = sp.build_openalex_search(q, QT)[0]
        seen["pool_no_copy"] = pool.submit(lambda: sp.build_openalex_search(q, QT)[0]).result()
        seen["pool_copy"] = pool.submit(contextvars.copy_context().run, lambda: sp.build_openalex_search(q, QT)[0]).result()
        bg = {}
        th = threading.Thread(target=lambda: bg.update(q=sp.build_openalex_search(q, QT)[0], h=sq.current_request_attest()))
        th.start()
        th.join()
    pool.shutdown()
    assert inside != ctrl and seen["pool_copy"] == inside  # request-context copies share the one immutable holder
    assert seen["pool_no_copy"] == ctrl and bg["q"] == ctrl and bg["h"] is None  # hand-off job: a fresh context
    assert sp.build_openalex_search(q, QT)[0] == ctrl
