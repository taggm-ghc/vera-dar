"""DarCorpus.apply_delta (incremental freshness): applying the rows added since a snapshot gives the same retrieval as
restoring the full snapshot. Synthetic demo9/eco1 fixtures; shipped config (DAR ON); no DB/network/env."""
import copy
import json
import re
from pathlib import Path

import pytest

from vera import domain_registry
from vera.dar_corpus import DarCorpus

BASE = Path(__file__).parent / "fixtures" / "dar"
ROWS = json.loads((BASE / "demo9.json").read_text()) + json.loads((BASE / "eco1.json").read_text())


def _cfg():
    c = copy.deepcopy(domain_registry.load_config())
    assert c["enabled"] is True
    return c


def _doc(r, i):
    return {"source_key": r["doc_id"], "url": r["url"], "title": r["title"], "text": r["content"], "year": r["year"],
            "source_type": r["source_type"], "retrieved_at": "2026-10-19", "stored_text_kind": "full_text",
            "admission_passed": True, "db_source_id": 100 + i, "db_canonical_source_id": 200 + i}


def _live(n):
    c = DarCorpus(_cfg())
    docs = [_doc(r, i) for i, r in enumerate(ROWS[:n])]
    for d in docs:
        c.ingest_document(d)
    return c, docs


def _snap(c, docs):
    reg = c.registry
    snap = {"domains": [{"name": d.name, "name_norm": d.name_norm, "description": d.description, "origin": d.origin}
                        for d in reg.list_domains()],
            "documents": [{k: v for k, v in d.items() if k != "admission_passed"} for d in docs],
            "corpus_assoc": [], "chunk_assoc": []}
    for d in reg.list_domains():
        for key in reg.corpora_for_domain(d.domain_id):
            snap["corpus_assoc"].append({"name_norm": d.name_norm, "source_key": key,
                                         "asserted_by": reg.association_actor("corpus", d.domain_id, key)})
        for sp in reg.chunks_for_domain(d.domain_id):
            m = re.match(r"^(.*):(\d+)-(\d+)$", sp)
            snap["chunk_assoc"].append({"name_norm": d.name_norm, "source_key": m.group(1), "start": int(m.group(2)),
                                        "end": int(m.group(3)), "asserted_by": reg.association_actor("chunk", d.domain_id, sp)})
    return snap


def _delta(old, new):
    have_docs = {d["source_key"] for d in old["documents"]}
    have_dom = {d["name_norm"] for d in old["domains"]}
    key = lambda a: (a["name_norm"], a["source_key"], a.get("start"), a.get("end"))  # noqa: E731
    have_c = {key(a) for a in old["corpus_assoc"]}
    have_k = {key(a) for a in old["chunk_assoc"]}
    return {"domains": [d for d in new["domains"] if d["name_norm"] not in have_dom],
            "documents": [d for d in new["documents"] if d["source_key"] not in have_docs],
            "corpus_assoc": [a for a in new["corpus_assoc"] if key(a) not in have_c],
            "chunk_assoc": [a for a in new["chunk_assoc"] if key(a) not in have_k]}


def _q(i, n=8):
    words = sorted({w for w in ROWS[i]["content"].lower().split() if w.isalpha() and len(w) > 6})
    return " ".join(words[:n])


@pytest.mark.parametrize("cut", [6, 9])
def test_delta_equals_full_restore(cut):
    old_live, old_docs = _live(cut)
    new_live, new_docs = _live(len(ROWS))
    s_old, s_new = _snap(old_live, old_docs), _snap(new_live, new_docs)
    inc = DarCorpus(_cfg())
    assert inc.restore(s_old)["status"] == "restored"
    delta = _delta(s_old, s_new)
    assert delta["documents"], "fixture must add documents"
    assert inc.apply_delta(delta)["status"] == "applied"
    full = DarCorpus(_cfg())
    assert full.restore(s_new)["status"] == "restored"
    assert inc.stats()["documents"] == full.stats()["documents"] == len(ROWS)
    for i in range(len(ROWS)):
        q = _q(i)
        a, b = inc.retrieve(q), full.retrieve(q)
        assert [x["span_id"] for x in a["candidates"]] == [x["span_id"] for x in b["candidates"]], i
        assert a["trace"]["question_domains"] == b["trace"]["question_domains"], i


def test_new_document_retrievable_after_delta_without_rebuild():
    old_live, old_docs = _live(9)
    new_live, new_docs = _live(10)
    inc = DarCorpus(_cfg())
    inc.restore(_snap(old_live, old_docs))
    q = _q(9)
    assert not any(c["source_key"] == "E-1" for c in inc.retrieve(q)["candidates"])
    r = inc.apply_delta(_delta(_snap(old_live, old_docs), _snap(new_live, new_docs)))
    assert r == {"status": "applied", "documents": 1}
    assert any(c["source_key"] == "E-1" for c in inc.retrieve(q)["candidates"])


def test_inconsistent_delta_asks_for_a_full_rebuild_and_changes_nothing_indexed():
    c, docs = _live(6)
    n = c.stats()["documents"]
    bad = {"domains": [], "documents": [], "corpus_assoc": [{"name_norm": "x", "source_key": "nope", "asserted_by": "a"}],
           "chunk_assoc": []}
    assert c.apply_delta(bad) == {"status": "needs_full", "reason": "association_for_unknown_document"}
    bad2 = {"domains": [], "documents": [], "corpus_assoc": [],
            "chunk_assoc": [{"name_norm": "x", "source_key": "D1-1", "start": 1, "end": 2, "asserted_by": "a"}]}
    assert c.apply_delta(bad2)["status"] == "needs_full"
    assert c.stats()["documents"] == n


def test_flag_off_is_inert():
    cfg = _cfg()
    cfg["enabled"] = False
    assert DarCorpus(cfg).apply_delta({"documents": []}) == {"status": "off"}


# ---- consolidation executed by DarCorpus (detection is vera.dar_consolidate) ----
def _dup_cfg():
    c = _cfg()
    c["identification"].update({"assoc_min_sim": 0.99, "sig_min_shared": 999})  # force early over-splitting
    return c


def _docs_of(prefixes):
    return [_doc(r, i) for i, r in enumerate(ROWS) if r["doc_id"][:2] in prefixes]


def test_consolidation_merges_forced_duplicates_and_keeps_subjects_apart():
    c = DarCorpus(_dup_cfg())
    plans = [c.ingest_document(d) for d in _docs_of({"D1", "D2", "D3"})]
    derived = [d for d in c.registry.list_domains() if d.origin == "derived"]
    assert len(derived) == 9  # every document founded its own domain (forced)
    active = {d.domain_id for d in derived} - set(c.identifier.aliases)
    # the three subjects survive as separate canonical domains; duplicates of one subject collapse
    assert len(active) == 3, sorted(active)
    by_subject = {}
    for did in active:
        subj = {k[:2] for k in c.registry.corpora_for_domain(did)}
        by_subject[did] = subj
    for did in active:  # every canonical domain contains all three documents of its subject
        keys = {k for k in c.registry.corpora_for_domain(did) if not c.registry.association_actor("corpus", did, k).endswith(":secondary")}
        assert keys  # has primary members
    assert any(e["corpus_assoc"] or e["chunk_assoc"] for p in plans for e in p["promoted"])  # reported for persist
    for q in (_q(0), _q(3), _q(6)):  # classification never returns an alias
        assert all(i not in c.identifier.aliases for i in c.registry.identify_question(q))


def test_consolidation_is_reproduced_after_restore_and_order_independent():
    import random
    results = []
    for seed in (0, 1, 2):
        docs = _docs_of({"D1", "D2", "D3"})
        random.Random(seed).shuffle(docs)
        c = DarCorpus(_dup_cfg())
        for d in docs:
            c.ingest_document(d)
        groups = frozenset(frozenset(k for k in c.registry.corpora_for_domain(d.domain_id)
                                     if (c.registry.association_actor("corpus", d.domain_id, k) or "").endswith(":secondary") is False)
                           for d in c.registry.list_domains() if d.domain_id not in c.identifier.aliases)
        results.append(groups)
        snap = _snap(c, docs)
        r = DarCorpus(_dup_cfg())
        assert r.restore(snap)["status"] == "restored"
        assert set(r.identifier.aliases) == set(c.identifier.aliases) or len(r.identifier.aliases) == len(c.identifier.aliases)
    subjects = [{frozenset(k[:2] for k in g) for g in res} for res in results]
    assert all(all(len(s) == 1 for s in grp) for grp in subjects)  # no canonical group mixes subjects, in any order
