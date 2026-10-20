"""DARINT: stored candidates compete with external ones under one relevance policy (DarCorpus.rank_merge), not by
insertion order. Includes the verifier DAR-6 shape. Synthetic, offline."""
import copy
import json
from pathlib import Path

from vera import grounded_ask as ga
from vera.dar_corpus import DarCorpus

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = json.loads((ROOT / "config" / "domain_routing.json").read_text())
GCFG = {"enabled": True, "providers": ["p1"], "num_results": 5, "deadline_s": 5, "snippet_max_chars": 200,
        "resolve": {"enabled": True, "max_results_to_resolve": 4, "deadline_s": 5}}
VOCAB = json.loads((ROOT / "tests" / "fixtures" / "dar" / "corpus_vocab.json").read_text())


def _body(topic, i, n=12):
    w = VOCAB[topic]
    return " ".join(" ".join(w[(i * 2 + s * 3 + j) % len(w)] for j in range(7)) + "." for s in range(n))


def _doc(key, topic, i):
    return {"source_key": key, "url": f"https://stored.example/{key}", "title": f"T {key}", "text": _body(topic, i),
            "year": 2024, "source_type": "journal", "retrieved_at": "2025-01-01", "stored_text_kind": "full_text",
            "admission_passed": True}


def _corpus(per=3, stored=True):
    cfg = copy.deepcopy(SHIPPED)
    cfg["enabled"] = True
    cfg["identification"]["salient_df_max_frac"] = 0.7
    cfg["identification"]["assoc_salient_min_docs"] = 2
    cfg["corpus"]["stored_branch_enabled"] = stored
    c = DarCorpus(cfg)
    for i in range(per):
        c.ingest_document(_doc(f"reef{i}", "reef", i))
    return c

Q = " ".join(VOCAB["reef"][:5])
N = GCFG["num_results"]


def _ext(n=5):
    return [{"url": f"https://e.example/{i}", "title": f"e{i}", "snippet": "abstract"} for i in range(n)]


def _cfg(c):
    return c.cfg, c.cfg["corpus"]["stored_merge_max"], c.cfg["min_fallback_slots"]


def _head(c):
    cfg, m, fb = _cfg(c)
    return min(5, max(fb, N - m))


def _big(extra):
    c = _corpus(per=3)
    for i in range(3, 3 + extra):
        c.ingest_document(_doc(f"reef{i}", "reef", i))
    return c


def _merge(c, ext):
    return ga._merge_stored(ext, Q, c, GCFG, {})


def test_corpus_expansion_never_removes_external_head():
    ext = _ext()
    for extra in (1, 5, 15):
        c = _big(extra)
        head = _head(c)
        out, _ = ga.retrieve_all(Q, GCFG, search=lambda _q: ext, dar_corpus=c)
        urls = [s["url"] for s in out]
        assert urls[:head] == [e["url"] for e in ext[:head]], (extra, urls)
        assert len(out) <= N


def test_stronger_stored_outranks_weaker_external():
    c = _corpus(per=3)
    ext = _ext()
    head = _head(c)
    merged = _merge(c, ext)
    first_stored = next(i for i, r in enumerate(merged) if r.get("_domain_branches") == ["stored"])
    assert first_stored >= head
    assert first_stored < merged.index(ext[4])


def test_document_in_both_appears_once():
    c = _corpus(per=3)
    stored, _ = c.as_sources(Q)
    ext = _ext() + [{"url": stored[0]["url"], "title": "dup", "snippet": "abstract"}]
    merged = _merge(c, ext)
    assert sum(r["url"] == stored[0]["url"] for r in merged) == 1
    out, _ = ga.retrieve_all(Q, GCFG, search=lambda _q: ext, dar_corpus=c)
    assert [s["url"] for s in out].count(stored[0]["url"]) <= 1


def test_no_external_stored_fill_in_own_order():
    c = _big(6)
    stored, _ = c.as_sources(Q)
    assert [r["url"] for r in _merge(c, [])] == [r["url"] for r in stored]


def test_deterministic():
    c = _big(5)
    ext = _ext()
    a = [r["url"] for r in _merge(c, ext)]
    b = [r["url"] for r in _merge(c, ext)]
    assert a == b


def test_flag_off_and_error_identity(monkeypatch):
    ext = _ext()
    assert _merge(_corpus(stored=False), ext) is ext
    assert ga._merge_stored(ext, Q, None, GCFG, {}) is ext
    c = _corpus()
    c.cfg["corpus"]["stored_min_relevance"] = "x"  # makes rank_merge raise: the branch falls back to identity
    info = {}
    assert ga._merge_stored(ext, Q, c, GCFG, info) is ext
    assert "error_class" in info["stored"]


def test_stored_provenance_fields_and_retrieved_at():
    c = _corpus(per=2)
    stored, _ = c.as_sources(Q)
    stored[0]["disposition"] = "proposed"
    out, _ = ga.retrieve_all(Q, GCFG, search=lambda _q: [], dar_corpus=type("C", (), {"cfg": c.cfg, "registry": c.registry, "as_sources": lambda self, q: (stored, {}), "rank_merge": lambda self, q, e, st, n: list(st)})())
    s = out[0]
    assert s["disposition"] == "proposed" and s["span_id"] == stored[0]["dar_span"]["span_id"]
    assert s["retrieved_at"] == stored[0]["retrieved_at"]
    pub = ga.public_sources(out)[0]
    assert pub["span_id"] == s["span_id"] and pub["disposition"] == "proposed"
    cand = ga.public_candidates(out)[0]
    assert {"span", "span_id", "disposition", "retrieved_at"} <= set(cand)
    # stored source without a stored time: never now()
    assert ga.provenance({"url": "https://x.example/a", "dar_span": {}})["retrieved_at"] is None
    # web source unchanged: now() fallback
    assert ga.provenance({"url": "https://x.example/a"})["retrieved_at"]
    web = ga.public_candidates([{"n": 1, "url": "https://w.example/1", "title": "w"}])[0]
    assert "span_id" not in web and "disposition" not in web


def _strong_ext(n=5):
    return [{"doi": f"10.9999/ext.{i}", "url": f"https://doi.org/10.9999/ext.{i}",
             "title": f"Insecticide-treated bed nets and malaria: randomized trial {i}",
             "snippet": "Randomized trials of insecticide-treated bed nets reduced malaria incidence and child mortality; "
                        "efficacy of treated nets against malaria was consistent across trials."} for i in range(1, n + 1)]


def test_dar6_irrelevant_stored_chunks_never_push_strong_external_evidence_out():
    q = "What is the efficacy of insecticide-treated bed nets against malaria in randomized trials?"
    ext = _strong_ext()
    for per in (1, 3, 8):  # corpus expansion must not matter
        out, _ = ga.retrieve_all(q, GCFG, search=lambda _q: ext, dar_corpus=_corpus(per=per))
        urls = [s["url"] for s in out]
        assert urls == [e["url"] for e in ext], (per, urls)  # all five strong external results kept, in order


def test_dar6_inverse_strongly_relevant_stored_chunk_is_included_beside_weak_external():
    weak = [{"doi": f"10.9999/ext.{10 + i}", "url": f"https://doi.org/10.9999/ext.{10 + i}", "title": f"Unrelated note {i}",
             "snippet": "General commentary on publishing."} for i in range(1, 6)]
    out, _ = ga.retrieve_all(Q, GCFG, search=lambda _q: weak, dar_corpus=_corpus(per=3))
    urls = [s["url"] for s in out]
    assert any(u.startswith("https://stored.example/") for u in urls)  # strong stored evidence is not suppressed
    stored_pos = min(i for i, u in enumerate(urls) if u.startswith("https://stored.example/"))
    assert stored_pos <= 3 and urls[:3] == [w["url"] for w in weak[:3]]  # ...while the protected head stays in place


def test_relevance_is_one_function_for_both_sides():
    c = _corpus(per=1)
    assert c.relevance(Q, "") == 0.0 and c.relevance(Q, Q) == 1.0
    assert c.relevance("coral reef bleaching", "Reefs bleach: corals") > c.relevance("coral reef bleaching", "unrelated words")


def test_domain_classes_never_exclude_unrestricted_branch_returns_outside_domain_evidence():
    """DAR-7 principle: question domains rank, they never filter. Over several questions at least one candidate lies
    OUTSIDE the question domains and arrived through the unrestricted branch only (verifier DAR-7 regression)."""
    base = ROOT / "tests" / "fixtures" / "dar"
    rows = json.loads((base / "demo9.json").read_text()) + json.loads((base / "eco1.json").read_text())
    cfg = copy.deepcopy(SHIPPED)
    cfg["enabled"] = True
    c = DarCorpus(cfg)
    for r in sorted(rows, key=lambda r: r["doc_id"]):
        c.ingest_document({"source_key": r["doc_id"], "text": r["content"], "stored_text_kind": "full_text",
                           "admission_passed": True})
    assert cfg["corpus"]["max_stored_domain_branches"] <= 2  # a wide branch cap would cover every domain
    outside_seen, branch_seen = False, False
    for r in rows:
        words = sorted({w for w in r["content"].lower().split() if w.isalpha() and len(w) > 6})
        res = c.retrieve(" ".join(words[:6]) + " " + " ".join(words[-6:]))
        qd = set(res["trace"]["question_domains"])
        out = [x for x in res["candidates"] if qd and not (set(x["domains"]) & qd)]
        branch_seen |= any(b.startswith("domain:") for x in res["candidates"] for b in x["branches"])
        outside_seen |= bool(out) and all("unrestricted" in x["branches"] for x in out)
    assert branch_seen and outside_seen
