"""DARINT QA regressions CX-1..CX-10 (dar-properties.md). Offline, DB-free, network-free; synthetic fixtures.
CX-4/6/8/9 (ingest, cross-domain, bounded domains, restart) need vera.dar_corpus and are in the second half."""
import copy
import hashlib
import json
import logging
import sys
import types
from pathlib import Path

import pytest

from vera import domain_registry as reg
from vera import domain_routing as dr
from vera import grounded_ask as ga

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = json.loads((ROOT / "config" / "domain_routing.json").read_text())
GCFG = {"enabled": True, "providers": ["p1"], "num_results": 5, "deadline_s": 5, "snippet_max_chars": 200,
        "resolve": {"enabled": True, "max_results_to_resolve": 4, "deadline_s": 5}}


def _enabled(**kw):
    c = copy.deepcopy(SHIPPED)
    c["enabled"] = True
    # fixture vocabulary is shared by every document of a topic: widen the shared-vocabulary band in this COPY only
    c["identification"]["salient_df_max_frac"] = 0.7
    c["identification"]["assoc_salient_min_docs"] = 2
    c.update(kw)
    return c


def _fake_search_module(monkeypatch, base, domain_items):
    """vera.search_and_fetch is replaced by a fake: the question call returns `base`, the domain-suffixed call returns domain_items."""
    mod = types.ModuleType("vera.search_and_fetch")
    mod._search_settings = lambda: {}
    mod.cached_chain = lambda s: ([types.SimpleNamespace(name="p1")], None, None)
    mod.search_question = lambda q, **kw: list(domain_items) if q.endswith("security") else list(base)
    monkeypatch.setitem(sys.modules, "vera.search_and_fetch", mod)


def _route_on(monkeypatch, cfg):
    monkeypatch.setattr(dr, "flag_enabled", lambda *a, **k: True)
    monkeypatch.setattr(dr, "get_cfg", lambda *a, **k: cfg)
    monkeypatch.setattr(dr, "get_registry", lambda *a, **k: reg.load_registry(cfg))


def _linkonly(i):
    return {"url": f"https://base.example/{i}", "title": f"base{i}", "snippet": ""}


def _withabs(i):
    return {"url": f"https://dom.example/{i}", "title": f"dom{i}", "snippet": f"domain abstract {i}"}


def test_cx1_base_link_only_items_survive_and_reach_link_resolution(monkeypatch):
    cfg = _enabled(domains=[{"name": "security"}], max_domain_branches=2, min_fallback_slots=2)
    _route_on(monkeypatch, cfg)
    _fake_search_module(monkeypatch, [_linkonly(i) for i in range(5)], [_withabs(i) for i in range(5)])
    seen = []

    def resolver(r, rcfg, deadline_at):
        seen.append(r["url"])
        return [{"title": "orig " + r["title"], "url": "https://doi.org/10.1/" + r["title"], "snippet": "resolved abstract",
                 "doi": "10.1/" + r["title"]}]

    ga.retrieve_all("AI code assistants and security risks", GCFG, resolver=resolver)
    assert seen, "base link-only items never reached link resolution (CX-1 defect 6-A)"
    assert all(u.startswith("https://base.example/") for u in seen)


def test_cx1_routing_off_shortcut_unchanged():
    results = [{"url": f"https://x.example/{i}", "title": "t", "snippet": "abs"} for i in range(5)]
    called = []
    out, _ = ga.retrieve_all("q", GCFG, search=lambda q: results + [_linkonly(9)],
                             resolver=lambda *a: called.append(1) or [])
    assert len(out) == 5 and not called  # enough direct sources: link following still skipped


def test_cx2_trace_counts_logged_and_exposed_without_sources(monkeypatch, caplog):
    cfg = _enabled(domains=[{"name": "security"}], max_domain_branches=2, min_fallback_slots=2)
    _route_on(monkeypatch, cfg)
    _fake_search_module(monkeypatch, [_linkonly(i) for i in range(5)], [_withabs(i) for i in range(5)])
    trace = {}
    with caplog.at_level(logging.INFO, logger="vera"):
        ga.retrieve_all("AI code assistants and security risks", GCFG, resolver=lambda *a: [], trace_out=trace)
    assert trace["routing"]["routing"] == "on" and trace["routing"]["reserved"] == 2
    assert trace["routing"]["branch_counts"]["unrestricted"] == 5
    blob = json.dumps(trace) + caplog.text
    assert "base.example" not in blob and "dom.example" not in blob and "domain abstract" not in blob  # counts only


def test_trace_out_optional_return_contract_unchanged():
    r = [{"url": "https://a.example/1", "title": "t", "snippet": "s"}]
    assert ga.retrieve_all("q", GCFG, search=lambda q: r)[0] == ga.retrieve_all("q", GCFG, search=lambda q: r, trace_out={})[0]


def test_cx5_router_declines_out_of_scope_question_before_retrieval():
    """Characterisation only (owner decision #72): an unrepresented field is declined by the scope router, so nothing
    from it can be ingested through /ask. Not a DAR defect; scope config is not widened."""
    from vera import scope_router as sr
    raw = json.loads((ROOT / "config" / "scope_router.json").read_text())
    cfg = sr.parse_config(raw, layer2_mode="none")
    d = sr.route("What does research show about CRISPR off-target effects?", "m", cfg=cfg)
    assert d.accepted is False


def test_cx10_single_domain_routing_import_in_retrieve_all():
    import ast
    tree = ast.parse((ROOT / "vera" / "grounded_ask.py").read_text())
    fn = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == "retrieve_all")
    imports = [n for n in ast.walk(fn) if isinstance(n, ast.ImportFrom) and any(a.name == "domain_routing" for a in n.names)]
    assert len(imports) == 1


def test_registry_docstring_states_the_new_derivation_rule():
    doc = reg.__doc__
    assert "never auto-registered as a domain" not in doc
    assert "min_support" in doc and "derived" in doc
    d, created = reg.InMemoryDomainRegistry().register("auto topic", origin="derived")
    assert created and d.origin == "derived"
    r = reg.InMemoryDomainRegistry()
    r.record_uncertain("maybe")
    assert r.list_domains() == []  # record_uncertain itself never registers


def test_shipped_routing_on_stored_branch_off_and_blocks_carry_calibration_note():
    """OD-DARINT-DAR-ON: shipped ON (note says so); OFF settable; ON + domains:[] + empty registry == baseline."""
    assert SHIPPED["enabled"] is True and SHIPPED["domains"] == []
    assert "OD-DARINT-DAR-ON" in SHIPPED["_calibration_note"]
    assert reg.routing_enabled({**SHIPPED, "enabled": False}) is False
    base = [{"url": "https://a.invalid/1"}]
    out, trace = dr.route_search("q", base, lambda q: [{"url": "https://b.invalid/2"}], reg.InMemoryDomainRegistry(),
                                 SHIPPED)
    assert out is base
    for blk in ("identification", "corpus"):
        assert "synthetic fixtures only" in SHIPPED[blk]["_calibration_note"] and "not on real data" in SHIPPED[blk]["_calibration_note"]
    # Shipped stored_branch_enabled=false per OD-1029-P4-IIA (2026-10-29), which supersedes only the flag-state
    # part of D-DARINT-019 and D-DARINT-037. The hook stays OFF.
    assert SHIPPED["corpus"]["stored_branch_enabled"] is False and SHIPPED["corpus"]["admission_hook_enabled"] is False


def test_shipped_config_stored_branch_off_get_corpus_inert():
    """C1 (OD-1029-P4-IIA): against the UNPATCHED shipped config get_corpus is inert: None, no engine built."""
    from vera import dar_runtime
    dar_runtime.reset_for_tests()
    calls = []
    try:
        assert dar_runtime.get_corpus(env={"VERA_DB_URL_RO": "postgresql://u:p@host.invalid:5432/d"},
                                      engine_factory=lambda *a, **k: calls.append(1)) is None
        assert calls == [] and dar_runtime.status() == "disabled"
    finally:
        dar_runtime.reset_for_tests()


# ---- stored-corpus integration (CX-3, CX-4, CX-6, CX-8, CX-9) ----
from vera.dar_corpus import DarCorpus  # noqa: E402

VOCAB = json.loads((ROOT / "tests" / "fixtures" / "dar" / "corpus_vocab.json").read_text())


def _body(topic, i, n=12):
    w = VOCAB[topic]
    return " ".join(" ".join(w[(i * 2 + s * 3 + j) % len(w)] for j in range(7)) + "." for s in range(n))


def _doc(key, topic, i, kind="full_text"):
    return {"source_key": key, "url": f"https://stored.example/{key}", "title": f"T {key}", "text": _body(topic, i),
            "year": 2024, "source_type": "journal", "retrieved_at": "2025-01-01", "stored_text_kind": kind,
            "admission_passed": True}


def _corpus(topics=("reef",), per=3, stored=True):
    cfg = _enabled()
    cfg["corpus"]["stored_branch_enabled"] = stored
    c = DarCorpus(cfg)
    for t in topics:
        for i in range(per):
            c.ingest_document(_doc(f"{t}{i}", t, i))
    return c


def test_cx4_stored_full_document_reaches_retrieve_all_with_external_empty():
    c = _corpus()
    q = " ".join(VOCAB["reef"][:5])
    out, _ = ga.retrieve_all(q, GCFG, search=lambda _q: [], dar_corpus=c)
    assert out and all(s["url"].startswith("https://stored.example/") for s in out)
    assert all(s["discovered_via"] == "stored_corpus" and s["stored_text_kind"] == "full_text" for s in out)
    assert len(out) <= SHIPPED["corpus"]["stored_fill_max"]  # no external results: stored chunks fill the free slots


def test_stored_branch_off_by_flag_or_missing_corpus_is_identical():
    q = " ".join(VOCAB["reef"][:5])
    ext = [{"url": "https://e.example/1", "title": "e", "snippet": "abstract"}]
    base = ga.retrieve_all(q, GCFG, search=lambda _q: ext)
    assert ga.retrieve_all(q, GCFG, search=lambda _q: ext, dar_corpus=_corpus(stored=False)) == base
    assert ga.retrieve_all(q, GCFG, search=lambda _q: ext, dar_corpus=None) == base
    off = _corpus()
    off.cfg["enabled"] = False
    assert ga.retrieve_all(q, GCFG, search=lambda _q: ext, dar_corpus=off) == base


def test_stored_candidates_never_displace_the_reserved_external_head():
    c = _corpus()
    q = " ".join(VOCAB["reef"][:5])
    ext = [{"url": f"https://e.example/{i}", "title": f"e{i}", "snippet": "abstract"} for i in range(5)]
    tr = {}
    out, _ = ga.retrieve_all(q, GCFG, search=lambda _q: ext, dar_corpus=c, trace_out=tr)
    urls = [s["url"] for s in out]
    m = sum(u.startswith("https://stored.example/") for u in urls)
    assert m >= 1 and len(urls) == 5 and urls[: 5 - m] == [f"https://e.example/{i}" for i in range(5 - m)]
    assert tr["stored"] and "stored.example" not in json.dumps(tr)


def test_abstract_only_corpus_yields_nothing():
    cfg = _enabled()
    cfg["corpus"]["stored_branch_enabled"] = True
    c = DarCorpus(cfg)
    c.ingest_document(_doc("a", "reef", 0, kind="abstract_metadata"))
    assert ga.retrieve_all(" ".join(VOCAB["reef"][:5]), GCFG, search=lambda _q: [], dar_corpus=c) == ([], [])


def test_cx3_question_domain_identified_from_content_with_empty_config_domains():
    c = _corpus(("reef", "compiler"), per=5)
    assert SHIPPED["domains"] == [] and c.registry.list_domains()
    sel = dr.identify_domains(" ".join(VOCAB["reef"][:6]), c.registry, _enabled())
    assert sel["selected"] and "content" in sel["method"]


def test_cx6_cross_domain_question_reaches_both_domains_and_unrestricted_runs():
    c = _corpus(("reef", "compiler"))
    qq = " ".join(VOCAB["reef"][:4] + VOCAB["compiler"][:4])
    res = c.retrieve(qq)
    keys = {x["source_key"][:4] for x in res["candidates"]}
    assert {"reef", "comp"} <= keys and "unrestricted" in res["trace"]["branch_counts"]


def test_cx8_twenty_same_topic_docs_give_bounded_domains():
    c = _corpus(("reef",), per=20)
    assert len(c.registry.list_domains()) <= 3  # bounded; a one-topic corpus has no contrast (0 acceptable)


def test_cx9_restart_rebuild_from_snapshot_retrieves_same_candidates():
    c = _corpus(("reef", "compiler"))
    qq = " ".join(VOCAB["compiler"][:5])
    before = [x["span_id"] for x in c.retrieve(qq)["candidates"]]
    snap = {"domains": [{"name": d.name, "name_norm": d.name_norm, "description": d.description, "origin": d.origin}
                        for d in c.registry.list_domains()],
            "documents": [{**{k: v for k, v in dd.items() if k != "admission_passed"}} for dd in
                          [_doc(f"{t}{i}", t, i) for t in ("reef", "compiler") for i in range(3)]],
            "corpus_assoc": [{"name_norm": c.registry.get(d).name_norm, "source_key": k, "asserted_by": "x"}
                             for k in (f"{t}{i}" for t in ("reef", "compiler") for i in range(3))
                             for d in [x.domain_id for x in c.registry.domains_for_corpus(k)]],
            "chunk_assoc": []}
    cfg_hash = hashlib.sha256(json.dumps(c.cfg, sort_keys=True).encode()).hexdigest()
    c2 = DarCorpus(copy.deepcopy(c.cfg))
    c2.restore(snap)
    assert hashlib.sha256(json.dumps(c2.cfg, sort_keys=True).encode()).hexdigest() == cfg_hash
    assert before and [x["span_id"] for x in c2.retrieve(qq)["candidates"]] == before


# ---- fix round 3 ----
def _snapshot_of(c, docs):
    reg = c.registry
    chunks = [(ch["source_key"], ch["start"], ch["end"], ch["span_id"]) for ch in c._chunks]
    return {"domains": [{"name": d.name, "name_norm": d.name_norm, "description": d.description, "origin": d.origin}
                        for d in reg.list_domains()],
            "documents": [{k: v for k, v in d.items() if k != "admission_passed"} for d in docs],
            "corpus_assoc": [{"name_norm": d.name_norm, "source_key": d_["source_key"], "asserted_by": "x"}
                             for d_ in docs for d in reg.domains_for_corpus(d_["source_key"])],
            "chunk_assoc": [{"name_norm": d.name_norm, "source_key": k, "start": s, "end": e, "asserted_by": "x"}
                            for k, s, e, sid in chunks for d in reg.domains_for_chunk(sid)],
            "chunking": c.chunking()}


def test_d4_restored_corpus_selects_same_question_domains_and_topk_as_live():
    import random
    gen = [f"gen{i}" for i in range(60)]
    rng = random.Random(11)
    cfg = _enabled()
    live = DarCorpus(cfg)
    docs = []
    order = [("reef", 0), ("compiler", 0), ("reef", 1), ("compiler", 1), ("orbit", 0), ("orbit", 1), ("reef", 2), ("compiler", 2)]
    for tp, j in order:
        top = [w for w in _body(tp, j, 3).replace(".", "").split()]
        words = rng.sample(gen * 2, 60) + top
        rng.shuffle(words)
        d = {"source_key": f"{tp}{j}", "url": f"https://stored.example/{tp}{j}", "title": "t", "text": " ".join(words),
             "year": 2024, "source_type": "journal", "retrieved_at": "2025-01-01", "stored_text_kind": "full_text",
             "admission_passed": True}
        docs.append(d)
        live.ingest_document(d)
    assert live.registry.list_domains()
    rest = DarCorpus(copy.deepcopy(cfg))
    assert rest.restore(_snapshot_of(live, docs)).get("status") != "chunking_mismatch"
    queries = [" ".join(VOCAB[t][:5]) for t in ("reef", "compiler", "orbit")] + [
        " ".join(VOCAB["reef"][:3] + VOCAB["orbit"][:3]), " ".join(VOCAB["compiler"][:3] + VOCAB["orbit"][:3]), "zzzz qqqq"]
    for q in queries:
        a, b = live.retrieve(q), rest.retrieve(q)
        assert a["trace"]["question_domains"] == b["trace"]["question_domains"], q
        assert [x["span_id"] for x in a["candidates"]] == [x["span_id"] for x in b["candidates"]], q
        assert [i["domain_id"] for i in live.identifier.classify(q, "question")] == \
               [i["domain_id"] for i in rest.identifier.classify(q, "question")]
        assert [round(i["score"], 9) for i in live.identifier.classify(q, "question")] == \
               [round(i["score"], 9) for i in rest.identifier.classify(q, "question")]


def test_insert_new_only_maps_unique_violations_to_exists():
    import sqlalchemy.exc as sx
    from vera import corpus_admission as ca

    class Orig(Exception):
        def __init__(self, code):
            self.pgcode = code

    class Conn:
        rolled = 0

        def begin_nested(self):
            conn = self

            class SP:
                def commit(s):
                    pass

                def rollback(s):
                    conn.rolled += 1
            return SP()

    def store_raising(code):
        class S(ca.WoCorpusStore):
            def _one(self, conn, sql, params):
                raise sx.IntegrityError("stmt", {}, Orig(code))
        return S(None, "q")
    c = Conn()
    assert store_raising("23505")._insert_new(c, "sql", {}) is None and c.rolled == 1
    import pytest as _pt
    with _pt.raises(sx.IntegrityError):
        store_raising("23514")._insert_new(Conn(), "sql", {})  # CHECK violation must surface


def test_runtime_passes_configured_placeholder_span_text():
    from vera import answer_contract, dar_runtime
    assert dar_runtime._span_text(answer_contract.CHAIN_CONFIG_PATH) == "whole abstract"
    assert dar_runtime._span_text("/nonexistent/x.json") is None


# ---- QA round 2 / owner acceptance support ----
def test_displacement_external_head_survives_and_stored_take_the_free_slots():
    c = _corpus()
    q = " ".join(VOCAB["reef"][:5])
    ext = [{"url": f"https://e.example/{i}", "title": f"e{i}", "snippet": "abstract"} for i in range(5)]
    out, _ = ga.retrieve_all(q, GCFG, search=lambda _q: ext, dar_corpus=c)
    urls = [s["url"] for s in out]
    m = SHIPPED["corpus"]["stored_merge_max"]
    assert urls[: 5 - m] == [f"https://e.example/{i}" for i in range(5 - m)]  # ranks 1..n-m always survive
    assert sum(u.startswith("https://stored.example/") for u in urls) == m


def test_public_sources_carry_stored_provenance_and_web_sources_do_not():
    c = _corpus()
    out, _ = ga.retrieve_all(" ".join(VOCAB["reef"][:5]), GCFG, search=lambda _q: [], dar_corpus=c)
    pub = ga.public_sources(out)
    assert pub and all(p["span"].startswith("chars=") and p["provider"] == "stored_corpus"
                       and p["admission_passed"] is True and p["acquisition"] == "full_document_acquired" for p in pub)
    web = ga.public_sources([{"n": 1, "title": "t", "url": "https://w.example/1", "snippet": "s"}])
    assert not ({"span", "acquisition", "admission_passed", "canonical_source_id", "source_id"} & set(web[0]))


def test_epidemiology_and_ecology_forecasting_stay_two_domains_with_overlap_allowed():
    import json as _json
    base = ROOT / "tests" / "fixtures" / "dar"
    docs = {r["doc_id"]: r["content"] for r in _json.loads((base / "demo9.json").read_text()) + _json.loads((base / "eco1.json").read_text())}
    c = DarCorpus(_enabled())
    for k in sorted(docs):
        c.ingest_document({"source_key": k, "text": docs[k], "stored_text_kind": "full_text", "admission_passed": True})
    epi = {d.domain_id for d in c.registry.domains_for_corpus("D1-1")}
    eco = {d.domain_id for d in c.registry.domains_for_corpus("E-1")}
    assert epi and eco and not (epi & eco)  # distinct primary domains although both are about forecasting
