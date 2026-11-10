"""RAG-ADMIT-1109: "What are the components of RAG?" must not need the word "component" in every passage.
admission.relevance.question_structural_terms (component(s), parts, elements, types, kinds): kept in the retrieval query
and in the content stems, not required by the matched-stem floor. Offline: synthetic
documents, fake embedder, no network, no DB. Hybrid-on cases use tests/test_hybrid_retrieval_1109 helpers."""
from __future__ import annotations

import copy

import pytest

from vera import fulldoc_admission as fa, hybrid_retrieval as hr
from vera.dar_corpus import DarCorpus
from vera.fulldoc_corpus import FullDocCorpus, _routing_cfg, load_config
from tests.test_hybrid_retrieval_1109 import MODEL, Loader, _reset, setup_hybrid  # noqa: F401  (_reset: autouse)

Q = "What are the components of RAG?"
NEW_TERMS = ("component", "components", "parts", "elements")  # in question_structural_terms

RAG_TEXT = [
    "Retrieval-augmented generation (RAG) pairs a retriever over a vector index with a generator. The retriever fetches "
    "passages for a query from the index and the generator writes the answer from them. ",
    "In a RAG pipeline the index stores embedded passages, the retriever ranks them for each query and the generator "
    "conditions on the top passages when it writes. ",
]
COMP_ONLY = [
    "The components of a bicycle are the frame, the wheels and the chain, and each component wears at its own rate. ",
    "Engine components are inspected at every service, and worn components are replaced before the next season. ",
]
NEITHER = [
    "Granite weathers slowly in dry climates and the resulting sand is carried by wind across the plain each spring. ",
    "Basalt flows cooled quickly at the coast and formed columns that visitors walk along on guided tours today. ",
]


def _doc(paras):
    return "".join(f"Section {i}. " + p * 2 + "\n\n" for i, p in enumerate(paras))


def _corpus(paras, title="Doc"):
    """One document, one stored partition per paragraph group; -> (corpus, {span_id: (doc, start, end)})."""
    t = _doc(paras)
    cfg = load_config()
    dar = DarCorpus(_routing_cfg())
    key = "https://example.org/d"
    res = dar.ingest_document({"source_key": key, "url": key, "doi": None, "title": title, "published": "2026-01-01",
                               "source_type": "preprint", "retrieved_at": "2026-11-09", "text": t,
                               "stored_text_kind": "full_text", "admission_passed": True, "db_source_id": None,
                               "db_canonical_source_id": None})
    assert res.get("status") == "indexed"
    acq = [{"document_id": "D1", "canonical_source_id": key, "title": title, "url": key, "doi": None,
            "published": "2026-01-01", "authors": ["Author"], "licence": "CC-BY-4.0",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/", "licence_decision": "allow",
            "fulltext_format": "PDF", "text_sha256": "x", "chars": len(t), "evidence_ranges": [[0, len(t)]],
            "excluded_ranges": []}]
    c = FullDocCorpus("ok", acq, [], {"D1": t}, dar, {key: "D1"}, cfg)
    c.source_kind = "db"
    mid = len(t) // 2
    table, lst = {2001: ("D1", 0, mid), 2002: ("D1", mid, len(t))}, []
    for sid, (_d, s, e) in table.items():
        lst.append({"span_id": sid, "start": s, "end": e})
    c.stamps = {"D1": {"corpus_added_at": None, "corpus_stamps": lst}}
    return c, table


def _prefix_cfg():
    """The shipped config without the structural list (the behaviour before RAG-ADMIT-1109)."""
    cfg = copy.deepcopy(load_config())
    del cfg["admission"]["relevance"]["question_structural_terms"]
    return cfg


def test_shipped_config_lists_the_terms():
    rel = load_config()["admission"]["relevance"]
    assert all(t in rel["question_structural_terms"] for t in NEW_TERMS)
    assert not any(t in rel["question_generic_terms"] for t in NEW_TERMS)  # not generic: they stay content stems


def test_structural_terms_stay_in_the_retrieval_query_and_the_content_stems():
    from vera.search_query import drop_generic
    rel = load_config()["admission"]["relevance"]
    assert "components" in drop_generic(Q).lower()
    assert "compo" in fa._q_stems(Q, rel, int(rel["stem_len"])) and len(fa._q_stems(Q, rel, int(rel["stem_len"]))) == 2


def test_rag_passage_without_the_word_component_is_admitted_hybrid_off():
    c, _t = _corpus(RAG_TEXT)
    before = fa.admit(Q, c, _prefix_cfg())
    assert before.admitted == ()  # the defect: "compon*" AND "rag" were both required
    after = fa.admit(Q, c, load_config())
    assert after.admitted and all("retriever" in a["text"] for a in after.admitted), after.trace
    assert not any("component" in a["text"].lower() for a in after.admitted)


def test_components_only_passage_without_rag_is_still_rejected():
    c, _t = _corpus(COMP_ONLY)
    out = fa.admit(Q, c, load_config())
    assert out.admitted == ()


def test_passage_with_neither_is_rejected():
    c, _t = _corpus(NEITHER)
    assert fa.admit(Q, c, load_config()).admitted == ()


def test_hybrid_on_similarity_rule_admits_with_one_word_overlap(monkeypatch):
    # pre-fix config: the lexical floors need 2 stems, the passage has one ("rag"); only the vector rule can admit it
    c, table = _corpus(RAG_TEXT)
    cfg = _prefix_cfg()
    setup_hybrid(monkeypatch, c, table, chunks_per_partition=4, vector_k=4)  # stem_floor_sim unset: rule off
    assert fa.admit(Q, c, cfg).admitted == ()
    hr.set_config_for_tests({"enabled": True, "chunks_per_partition": 4, "vector_k": 4, "stem_floor_sim": 0.15})
    hr.set_loader_for_tests(Loader(c, table))
    hr.ensure_index(c, MODEL, sync=True)
    on = fa.admit(Q, c, cfg)
    assert on.trace["hybrid"]["floor_by_vector"] >= 1 and on.admitted, on.trace


def test_hybrid_on_with_null_floor_relies_on_the_structural_list(monkeypatch):
    c, table = _corpus(RAG_TEXT)
    setup_hybrid(monkeypatch, c, table, chunks_per_partition=4, vector_k=4, stem_floor_sim=None)
    assert fa.admit(Q, c, _prefix_cfg()).admitted == ()
    on = fa.admit(Q, c, load_config())
    assert on.trace["hybrid"]["floor_by_vector"] == 0 and on.admitted, on.trace
    # and the similarity rule never admits a passage with no question stem at all
    c2, t2 = _corpus(NEITHER)
    setup_hybrid(monkeypatch, c2, t2, chunks_per_partition=4, vector_k=4, stem_floor_sim=-1.0)
    assert fa.admit(Q, c2, load_config()).admitted == ()


# "elements"/"parts" stay content words elsewhere: the other content word still drives admission.
QCHEM = "What are the elements of group 1?"
CHEM_ON = ["Group 1 of the periodic table holds lithium, sodium and potassium; each group 1 metal has one valence "
           "electron and reacts readily with water. ",
           "Moving down group 1 the atoms grow larger, so potassium gives up its single electron more easily than "
           "lithium does. "]  # two passages: the one-stem relaxed rule needs two passing spans in the document
CHEM_OFF = ["Group dynamics in small teams depend on trust, and a group that meets weekly settles disputes faster. "]


def test_elements_structural_but_other_content_word_still_drives_admission():
    icfg = load_config()["admission"]["relevance"]
    stems = fa._q_stems(QCHEM, icfg, int(icfg.get("stem_len", 5)))
    assert "eleme" in stems and "group" in stems  # still a content stem; only the floor stops requiring it
    c, _t = _corpus(CHEM_ON)
    before = fa.admit(QCHEM, c, _prefix_cfg())
    after = fa.admit(QCHEM, c, load_config())
    assert after.admitted, after.trace  # still answered; the passage no longer has to say "elements"
    assert {a["text"] for a in before.admitted} <= {a["text"] for a in after.admitted}  # nothing lost
    assert "lithium" in after.admitted[0]["text"]
    # a passage with no content word of the question is still rejected
    c2, _t2 = _corpus(NEITHER)
    assert fa.admit(QCHEM, c2, load_config()).admitted == ()
