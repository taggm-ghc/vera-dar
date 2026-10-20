"""VERA-DAR L4 tests for vera.dar_corpus. Synthetic fixtures only (years >= 2023); no DB, network or env."""
import copy
import json
import re
from pathlib import Path

import pytest

from vera import dar_corpus, domain_registry
from vera.dar_corpus import DarCorpus, doc_from_admitted
from vera.m4 import context_builder, gate_c

ROOT = Path(__file__).resolve().parent
VOCAB = json.loads((ROOT / "fixtures" / "dar" / "corpus_vocab.json").read_text())


def make_cfg(enabled=True):
    cfg = copy.deepcopy(domain_registry.load_config())
    cfg["enabled"] = enabled
    # copy only (shared config untouched): widen the salience window so the small synthetic fixture forms domains once assoc_salient_min_docs is reached
    cfg["identification"]["salient_df_max_frac"] = 0.7
    # the min-support PROMOTION path (plan "promoted" entries) needs support 2; the shipped value (1, a first document
    # founds a provisional domain) is covered by tests/test_dar_identify.py and tests/test_dar_e2e_provisional.py
    cfg["identification"]["min_support"] = 2
    return cfg


def body(topic, i, sentences=12):
    """Deterministic topical text; doc i shifts the word window so docs differ but overlap heavily."""
    w = VOCAB[topic]
    out = []
    for s in range(sentences):
        out.append(" ".join(w[(i * 2 + s * 3 + j) % len(w)] for j in range(7)) + ".")
    return " ".join(out)


def doc(key, topic, i, year=2024, kind="full_text", passed=True, text=None, **kw):
    d = {"source_key": key, "url": f"https://example.org/{key}", "doi": None, "title": f"Title {key}",
         "text": text if text is not None else body(topic, i), "year": year, "source_type": "journal",
         "retrieved_at": "2025-01-01", "stored_text_kind": kind, "admission_passed": passed}
    d.update(kw)
    return d


def build(topics=("reef", "compiler"), per=5, cfg=None):
    c = DarCorpus(cfg or make_cfg())
    plans = []
    for t in topics:
        for i in range(per):
            plans.append(c.ingest_document(doc(f"{t}{i}", t, i)))
    return c, plans


def q(topic, n=5, off=0):
    return " ".join(VOCAB[topic][off:off + n])


def test_abstract_and_metadata_refused_and_listed():
    c = DarCorpus(make_cfg())
    for kind in ("abstract_metadata", "abstract", "metadata", "snippet"):
        r = c.ingest_document(doc(f"k_{kind}", "reef", 0, kind=kind))
        assert r["status"] == "skipped" and r["reason"] == "not_full_document"
    r = c.ingest_document(doc("notadmitted", "reef", 0, passed=False))
    assert r["reason"] == "not_full_document"
    assert c.stats()["documents"] == 0 and c.stats()["chunks"] == 0
    assert {s["stored_text_kind"] for s in c.skipped} >= {"abstract_metadata", "abstract", "metadata", "snippet"}
    assert all(s["reason"] == "not_full_document" for s in c.skipped)
    assert c.retrieve(q("reef"))["candidates"] == []


def test_self_declared_full_flag_does_not_bypass_gate():
    c = DarCorpus(make_cfg())
    r = c.ingest_document(doc("x", "reef", 0, kind="abstract_metadata", is_full=True))
    assert r["status"] == "skipped"


def test_flag_off_nothing():
    c = DarCorpus(make_cfg(enabled=False))
    assert c.ingest_document(doc("a", "reef", 0)) == {"status": "off"}
    assert c.stats()["documents"] == 0
    res = c.retrieve(q("reef"))
    assert res["candidates"] == [] and res["trace"]["routing"] == "off"
    assert c.restore({"documents": []}) == {"status": "off"}


def test_duplicate_idempotent():
    c = DarCorpus(make_cfg())
    assert c.ingest_document(doc("a", "reef", 0))["status"] == "indexed"
    n = c.stats()
    r = c.ingest_document(doc("a", "reef", 0))
    assert r == {"status": "skipped", "reason": "duplicate", "source_key": "a"}
    assert c.stats() == n


def test_offsets_exact_and_span_ids():
    c = DarCorpus(make_cfg())
    d = doc("a", "reef", 0, text="  " + body("reef", 0, 30) + "   ")
    plan = c.ingest_document(d)
    assert len(plan["chunks"]) > 2
    mx = int(c.cfg["corpus"]["chunk_chars"])
    for ch in plan["chunks"]:
        assert d["text"][ch["start"]:ch["end"]] == ch["text"]
        assert ch["span_id"] == f"a:{ch['start']}-{ch['end']}"
        assert not ch["text"][0].isspace() and not ch["text"][-1].isspace()
        assert len(ch["text"]) <= mx + int(c.cfg["corpus"]["min_chunk_chars"])
    assert plan["chunks"][-1]["end"] == len(d["text"].rstrip())
    starts = [ch["start"] for ch in plan["chunks"]]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


def test_two_chunks_of_one_doc_survive_fusion():
    c = DarCorpus(make_cfg())
    c.ingest_document(doc("a", "reef", 0, text=body("reef", 0, 30)))
    res = c.retrieve(q("reef", 3), k=8)
    keys = [x["source_key"] for x in res["candidates"]]
    assert keys.count("a") >= 2
    assert len({x["span_id"] for x in res["candidates"]}) == len(res["candidates"])


def test_unrestricted_branch_includes_unclassified_chunks():
    c, _ = build()
    odd = "zebra quokka narwhal pangolin ocelot " * 20
    plan = c.ingest_document(doc("odd", "reef", 0, text=odd))
    assert plan["status"] == "indexed"
    assert all(ch["domains"] == [] for ch in plan["chunks"])
    res = c.retrieve("zebra quokka narwhal")
    assert res["candidates"] and res["candidates"][0]["source_key"] == "odd"
    assert res["candidates"][0]["domains"] == []
    assert "unrestricted" in res["candidates"][0]["branches"]


def test_cross_domain_question_two_domains():
    c, _ = build()
    res = c.retrieve(q("reef", 4) + " " + q("compiler", 4), k=8)
    assert len(res["trace"]["question_domains"]) >= 2
    srcs = {x["source_key"][:-1] for x in res["candidates"]}
    assert {"reef", "compiler"} <= srcs
    doms = {d for x in res["candidates"] for d in x["domains"]}
    assert len(doms) >= 2


def test_new_domain_without_retuning():
    c, _ = build()
    h = json.dumps(c.cfg, sort_keys=True)
    n_dom = len(c.registry.list_domains())
    plans = [c.ingest_document(doc(f"orbit{i}", "orbit", i)) for i in range(3)]
    assert json.dumps(c.cfg, sort_keys=True) == h
    assert len(c.registry.list_domains()) == n_dom + 1
    assert any(d["created"] for p in plans for d in p["domains"])
    res = c.retrieve(q("orbit", 5))
    assert res["candidates"] and all(x["source_key"].startswith("orbit") for x in res["candidates"][:3])
    new_id = max(d.domain_id for d in c.registry.list_domains())
    assert new_id in res["trace"]["question_domains"]


def test_proposals_only():
    c, _ = build()
    res = c.retrieve(q("reef"))
    assert res["candidates"]
    assert {x["status"] for x in res["candidates"]} == {"proposed"}
    for x in res["candidates"]:
        for f in ("span_id", "source_key", "canonical_source_id", "source_id", "text", "start", "end",
                  "source_title", "url", "doi", "year", "source_type", "retrieved_at", "domains", "branches"):
            assert f in x
    assert c.retrieve(q("reef"), k=2)["candidates"].__len__() <= 2


def test_empty_corpus():
    c = DarCorpus(make_cfg())
    res = c.retrieve("anything at all")
    assert res["candidates"] == [] and res["trace"]["routing"] == "on"


def test_source_id_prefers_db_id():
    c = DarCorpus(make_cfg())
    c.ingest_document(doc("a", "reef", 0, db_source_id=77, db_canonical_source_id=5))
    assert c.retrieve(q("reef"))["candidates"][0]["source_id"] == 77


def test_to_evidence_corpus_feeds_gate_c_and_context():
    c, _ = build()
    question = q("reef", 4) + " " + q("compiler", 4)
    cands = c.retrieve(question, k=8)["candidates"]
    ev = c.to_evidence_corpus(question, cands)
    sq_ids = {s["id"] for s in ev["sub_questions"]}
    cc = c.cfg["corpus"]
    assert cc["general_sub_question"]["id"] in sq_ids
    assert len(sq_ids) >= 3 and ev["run_id"] == "dar"
    assert all(s["required"] is cc["sub_question_required"] for s in ev["sub_questions"])
    assert all(s["sub_question_ids"] and s["sub_question_ids"][0] == cc["general_sub_question"]["id"]
               for s in ev["spans"])
    assert any(len(s["sub_question_ids"]) > 1 for s in ev["spans"])
    assert len({s["independence_group"] for s in ev["spans"]}) >= gate_c.MIN_INDEPENDENT_SOURCES
    out = gate_c.evaluate_adequacy(ev, question, llm=None, use_llm_missingness=False)
    assert out["coverage"][cc["general_sub_question"]["id"]]["status"] == "covered"
    assert out["decision"] in ("proceed", "search_again", "insufficient", "proceed_qualified") or out["decision"]
    ctx, idx = context_builder.build_context_with_index(ev, {}, out)
    assert "Evidence units" in ctx and "NO EVIDENCE ADMITTED" not in ctx.split("sq_general")[1].split("###")[0]
    assert idx


def test_restore_gives_identical_retrieval():
    c, plans = build()
    qs = [q("reef", 4), q("compiler", 5, 3), q("reef", 3) + " " + q("compiler", 3)]
    before = [c.retrieve(x, k=8) for x in qs]
    snap = {"domains": [], "documents": [], "corpus_assoc": [], "chunk_assoc": []}
    for d in c.registry.list_domains():
        snap["domains"].append({"name": d.name, "name_norm": d.name_norm, "description": d.description,
                                "origin": d.origin})
        for sk in c.registry.corpora_for_domain(d.domain_id):
            snap["corpus_assoc"].append({"name_norm": d.name_norm, "source_key": sk, "asserted_by": "x"})
        for sp in c.registry.chunks_for_domain(d.domain_id):
            m = re.match(r"^(.*):(\d+)-(\d+)$", sp)
            snap["chunk_assoc"].append({"name_norm": d.name_norm, "source_key": m.group(1),
                                        "start": int(m.group(2)), "end": int(m.group(3)), "asserted_by": "x"})
    for rec in c._docs.values():
        snap["documents"].append({k: rec[k] for k in ("source_key", "url", "doi", "title", "text", "year",
                                                      "published", "source_type", "retrieved_at",
                                                      "stored_text_kind", "db_source_id", "db_canonical_source_id")})
    c2 = DarCorpus(make_cfg())
    assert c2.restore(snap)["documents"] == len(snap["documents"])
    for x, b in zip(qs, before):
        a = c2.retrieve(x, k=8)
        assert [(y["span_id"], y["domains"], y["branches"]) for y in a["candidates"]] == \
               [(y["span_id"], y["domains"], y["branches"]) for y in b["candidates"]]


def test_restore_refuses_abstract_rows():
    c = DarCorpus(make_cfg())
    r = c.restore({"documents": [{"source_key": "a", "text": body("reef", 0), "stored_text_kind": "abstract_metadata"}]})
    assert r["documents"] == 0 and c.skipped


def test_as_sources_shape_and_cap():
    c, _ = build(per=5)
    srcs, counts = c.as_sources(q("reef", 4))
    cap = c.cfg["corpus"]["stored_fill_max"]
    assert 0 < len(srcs) <= cap
    for s in srcs:
        assert set(s) == {"title", "url", "published", "snippet", "source_type", "doi", "stored_text_kind",
                          "discovered_via", "_domain_branches", "retrieved_at", "dar_span", "provider", "licence",
                          "admission_passed", "canonical_source_id", "source_id", "disposition"}
        assert s["provider"] == "stored_corpus" and s["admission_passed"] is True
        assert s["discovered_via"] == c.cfg["corpus"]["stored_discovered_via"]
        assert s["_domain_branches"] == ["stored"]
    assert all(isinstance(v, (int, str)) for v in counts.values())


def _mixed_plans(c, first_kw=None):
    plans = [c.ingest_document(doc("reef0", "reef", 0, **(first_kw or {})))]
    for t in ("reef", "compiler"):
        for i in range(5):
            if (t, i) != ("reef", 0):
                plans.append(c.ingest_document(doc(f"{t}{i}", t, i)))
    return plans


def test_promotion_reported_in_plan():
    c = DarCorpus(make_cfg())
    plans = _mixed_plans(c)
    first = next(i for i, p in enumerate(plans) if p["promoted"])
    assert all(p["corpus_assoc"] == [] for p in plans[:first])
    p1 = plans[first]
    assert p1["status"] == "indexed"
    assert "reef0" in [e["source_key"] for e in p1["promoted"]]
    e = [x for x in p1["promoted"] if x["source_key"] == "reef0"][0]
    assert e["corpus_assoc"] and e["corpus_assoc"][0]["canonical_source_id"] == "reef0"
    assert {d["domain_id"] for d in p1["domains"]} >= {a["domain_id"] for a in e["corpus_assoc"]}
    for a in e["chunk_assoc"]:
        assert a["span_id"].startswith("reef0:")


def test_bounded_derived_domains_on_one_topic():
    c = DarCorpus(make_cfg())
    for i in range(20):
        c.ingest_document(doc(f"r{i}", "reef", i))
    assert len(c.registry.list_domains()) <= 3


def test_doc_from_admitted_uses_stored_provenance():
    cand = {"canonical_url": "https://e.org/a", "url": "https://e.org/a?x=1", "title": "T", "doi": "10.1/x",
            "published": "2024-03-01", "source_type": "journal", "snippet": body("reef", 0),
            "stored_text_kind": "full_text", "provenance": {"stored_text_kind": "abstract_metadata"}}
    d = doc_from_admitted(cand, "admitted")
    assert d["stored_text_kind"] == "abstract_metadata" and d["admission_passed"] is True
    assert d["source_key"] == "https://e.org/a" and d["text"] == cand["snippet"]
    c = DarCorpus(make_cfg())
    assert c.ingest_document(d)["reason"] == "not_full_document"
    assert c.skipped[0]["stored_text_kind"] == "abstract_metadata"
    cand["provenance"] = {"stored_text_kind": "full_text"}
    assert doc_from_admitted(cand, "rejected")["admission_passed"] is False
    assert doc_from_admitted(cand, "admitted")["published"] == "2024-03-01"
    c2 = DarCorpus(make_cfg())
    assert c2.ingest_document(doc_from_admitted(cand, "admitted"))["status"] == "indexed"
    assert c2.retrieve(q("reef"))["candidates"][0]["year"] == 2024


def test_identify_question_hook_and_cfg_public():
    c = DarCorpus(make_cfg())
    assert c.cfg["enabled"] is True and callable(c.registry.identify_question)


def test_gate_ignores_per_document_contract_path(tmp_path):
    permissive = tmp_path / "permissive.json"
    permissive.write_text("{}")
    c = DarCorpus(make_cfg())
    r = c.ingest_document(doc("a", "reef", 0, kind="abstract_metadata", contract_path=str(permissive)))
    assert r["status"] == "skipped" and r["reason"] == "not_full_document"
    assert c.stats()["documents"] == 0


def test_promoted_entries_carry_resolvable_ids():
    c = DarCorpus(make_cfg())
    plans = _mixed_plans(c, {"db_source_id": 11, "db_canonical_source_id": 12})
    e = [x for p in plans for x in p["promoted"] if x["source_key"] == "reef0"][0]
    assert e["db_source_id"] == 11 and e["db_canonical_source_id"] == 12
    assert e["corpus_assoc"]
    for a in e["corpus_assoc"]:
        assert a["name_norm"] and a["canonical_source_id"] == "reef0" and a["asserted_by"]
    for a in e["chunk_assoc"]:
        assert a["name_norm"] and a["span_id"] == f"reef0:{a['start']}-{a['end']}" and a["asserted_by"]


def test_doc_from_admitted_admit_shaped_with_db_ids():
    cand = {"canonical_url": "https://e.org/z", "url": "https://e.org/z", "title": "T", "published": "2024-03-01",
            "source_type": "journal", "snippet": body("reef", 0), "source_id": 41, "canonical_source_id": 42,
            "provenance": {"stored_text_kind": "full_text", "retrieved_at": "2025-02-02"}}
    d = doc_from_admitted(cand, "admitted")
    assert d["db_source_id"] == 41 and d["db_canonical_source_id"] == 42 and d["retrieved_at"] == "2025-02-02"
    c = DarCorpus(make_cfg())
    assert c.ingest_document(d)["status"] == "indexed"
    top = c.retrieve(q("reef"))["candidates"][0]
    assert top["source_id"] == 41 and top["retrieved_at"] == "2025-02-02"


def test_restore_chunking_mismatch_and_chunking_dict():
    c, _ = build(per=2)
    ch = c.stats()["chunking"]
    assert ch == c.chunking() and set(ch) == {"chunk_chars", "chunk_overlap", "min_chunk_chars"}
    bad = dict(ch, chunk_chars=ch["chunk_chars"] + 7)
    snap = {"documents": [dict(r) for r in c._docs.values()], "chunking": bad}
    c2 = DarCorpus(make_cfg())
    r = c2.restore(snap)
    assert r["status"] == "chunking_mismatch" and r["snapshot_chunking"] == bad and r["corpus_chunking"] == ch
    assert c2.stats()["documents"] == 0 and c2.stats()["chunks"] == 0
    snap["chunking"] = ch
    assert DarCorpus(make_cfg()).restore(snap)["status"] == "restored"


def test_as_sources_snippet_truncated_and_dar_span():
    c = DarCorpus(make_cfg())
    c.ingest_document(doc("a", "reef", 0, text=body("reef", 0, 30)))
    n = c.cfg["corpus"]["snippet_chars"]
    res = c.retrieve(q("reef", 3), k=8)
    srcs, _ = c.as_sources(q("reef", 3), k=8)
    assert srcs and all(len(s["snippet"]) <= n for s in srcs)
    full = {x["span_id"]: x["text"] for x in res["candidates"]}
    assert any(len(t) > n for t in full.values())
    for s in srcs:
        assert s["url"] == "https://example.org/a" and s["retrieved_at"] == "2025-01-01"
        sp = s["dar_span"]
        assert sp["span_id"] == f"a:{sp['start']}-{sp['end']}"
        text = full.get(sp["span_id"])
        assert text is not None and text.startswith(s["snippet"])
        assert s["snippet"] == text or text[len(s["snippet"])].isspace()
    # one representative chunk per document (retrieve_all dedups by URL anyway): a second document adds a second source
    c.ingest_document(doc("b", "reef", 1, text=body("reef", 1, 30)))
    srcs2, _ = c.as_sources(q("reef", 3), k=8)
    assert len({s["url"] for s in srcs2}) == len(srcs2) == 2


def _snap_of(c):
    snap = {"domains": [], "documents": [], "corpus_assoc": [], "chunk_assoc": []}
    for d in c.registry.list_domains():
        snap["domains"].append({"name": d.name, "name_norm": d.name_norm, "description": d.description,
                                "origin": d.origin})
        for sp in c.registry.chunks_for_domain(d.domain_id):
            m = re.match(r"^(.*):(\d+)-(\d+)$", sp)
            snap["chunk_assoc"].append({"name_norm": d.name_norm, "source_key": m.group(1),
                                        "start": int(m.group(2)), "end": int(m.group(3)), "asserted_by": "x"})
    for rec in c._docs.values():
        snap["documents"].append({k: rec[k] for k in ("source_key", "url", "doi", "title", "text", "year",
                                                      "published", "source_type", "retrieved_at",
                                                      "stored_text_kind", "db_source_id", "db_canonical_source_id")})
    return snap


def test_restore_detects_chunking_mismatch_from_data():
    cfg_a = make_cfg()
    c = DarCorpus(cfg_a)
    for t in ("reef", "compiler"):
        for i in range(5):
            c.ingest_document(doc(f"{t}{i}", t, i, text=body(t, i, 30)))
    snap = _snap_of(c)
    assert snap["chunk_assoc"], "fixture must produce chunk associations"
    ok = DarCorpus(copy.deepcopy(cfg_a)).restore(snap)
    assert ok["status"] == "restored" and ok["documents"] == 10 and ok["pending_not_restored"] is True
    cfg_b = copy.deepcopy(cfg_a)
    cfg_b["corpus"]["chunk_chars"] = int(cfg_b["corpus"]["chunk_chars"]) + 37
    c2 = DarCorpus(cfg_b)
    r = c2.restore(snap)
    assert r["status"] == "chunking_mismatch" and r["unmatched"] > 0 and r["documents"] == 0
    assert c2.stats()["documents"] == 0 and c2.stats()["chunks"] == 0


def test_retrieve_k_zero():
    c, _ = build()
    r = c.retrieve(q("reef"), k=0)
    assert r["candidates"] == [] and r["trace"]["reserved"] == 0
    assert c.as_sources(q("reef"), k=0)[0] == []
