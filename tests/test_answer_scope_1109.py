"""ANSWER-SCOPE-1109: a general question answered from one specific-system document (live 731863f, "What are
components of RAG?"). Stubbed writer and checker, synthetic text shaped like the case, no network, no DB."""
import json
from pathlib import Path

import pytest

from vera import answer_scope as asc
from vera import claim_check as cc
from vera import fulldoc_answer as fa
from vera import research_record as rr
from vera.ask_service import AskResult, _with_floor_note
from vera.fulldoc_evidence import EvidenceBundle

ROOT = Path(__file__).resolve().parent.parent
Q = "What are components of RAG?"
SPEC_TITLE = "VLM-guided retrieval augmented generation (RAG) for robot action prediction"
GEN_TITLES = ["Retrieval-augmented generation: retrievers, generators and indexes",
              "A guide to RAG pipelines and their components"]
SPEC_TXT = ("Our system has four components: RAG image query optimization, a VLM that iteratively refines the query "
            "image, a structured knowledge tree, and a vector database for image retrieval.")
GEN1_TXT = ("A retrieval-augmented generation system combines a retriever that finds relevant passages with a generator "
            "that writes the answer from them.")
GEN2_TXT = "RAG pipelines also need an index of embedded documents that the retriever searches."
SCOPE = {"enabled": True, "writer_hint": True, "min_title_qualifiers": 3, "prompt_sentence": "\nSCOPE RULE.",
         "specific_scope_text": "Note: specific system ({titles}) only.",
         "limited_evidence_text": "Note: {n} full document{s}, fewer than {m}."}
MARK = "INSUFFICIENT_EVIDENCE"
PP = {"enabled": True, "system_prompt": "Parts then answers. Max {max_parts}. Marker {insufficient_marker}.\n{sources}",
      "partial_statement": "PARTIAL ANSWER.", "part_heading": "Part {n} ({part}):",
      "unanswered_line": "Part {n} ({part}): no.", "partial_result_kind": "answer"}
RESP = {"enabled": True, "removed_note": "Note: off-question removed.", "partial_statement": "RESP PARTIAL."}


def cfg(scope=True, per_part=True):
    a = {"system_prompt": "legacy\n{sources}", "quote_max_chars": 200, "partial_marker": " (PARTIAL)",
         "removed_note": "Note: removed.", "unchecked_note": "Note: unchecked.", "insufficient_marker": MARK,
         "max_total_quote_chars": 700, "max_parts": 4, "per_part": dict(PP, enabled=per_part), "responsiveness": RESP}
    if scope:
        a["scope_check"] = SCOPE
    return {"answer": a, "fallback_tokens": {"insufficient": "ins", "model_insufficient": "model_ins",
                                             "corpus_unusable": "cu", "no_valid_citations": "nvc",
                                             "checker_unavailable": "cku", "claim_check_insufficient": "cci",
                                             "claim_check_disabled": "ccd", "nonresponsive": "nonresp"}}


GCFG = {"removed_citation_note": "Note: bad citations removed.",
        "claim_check": {"enabled": True, "timeout_s": 5, "max_claims": 40,
                        "checkers": [{"provider": "t", "model": "m", "api_key_env": "FD_TEST_KEY", "base_url": None}]}}


def span(i, title, text):
    return {"span_id": f"s{i}", "document_id": f"d{i}", "canonical_source_id": f"https://x.org/{i}", "title": title,
            "url": f"https://x.org/{i}", "published": "2026", "authors": [], "licence": "CC-BY-4.0", "start": 0,
            "end": len(text), "text": text, "text_sha256": "ab" * 32, "relevance": 0.9,
            "attribution": {"creator": "A", "title": title, "source": f"https://x.org/{i}", "licence": "CC-BY-4.0",
                            "licence_url": "https://creativecommons.org/licenses/by/4.0/", "changes": "Excerpt."}}


def bundle(spans):
    docs = tuple({"document_id": d, "title": "t", "licence": "CC-BY-4.0", "licence_decision": "allow"}
                 for d in sorted({s["document_id"] for s in spans}))
    return EvidenceBundle(question=Q, examined_full_documents=docs, admitted=tuple(spans), status="admitted")


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("FD_TEST_KEY", "x")


def run(writer_text, ev, scope_by_cite, c=None, seen=None):
    """scope_by_cite: {source n -> "general"|"specific"} returned by the stub checker for claims citing it."""
    def check(chk, msgs):
        if seen is not None:
            seen.append(msgs)
        p = json.loads(msgs[1]["content"])
        return json.dumps({"claims": [{"id": x["id"], "verdict": "supported", "reason": "r", "responsive": "yes",
                                       **({"scope": scope_by_cite[x["cites"][0]]} if scope_by_cite else {})}
                                      for x in p["claims"]]})
    writes = []

    def write(msgs, *a, **k):
        writes.append(msgs)
        return AskResult(writer_text, 10, 0.01)
    o = fa.answer(Q, ev, "m", None, chained={}, grounding_cfg=GCFG, cfg=c or cfg(), complete_fn=write,
                  check_call=check, reference_texts=[s["text"] for s in ev.admitted])
    return o, writes


# ------------------------------------------------------------------------------------------------- mechanical signal
def test_title_qualifiers_flag_the_specific_paper_not_the_general_ones():
    q = asc.title_qualifiers(Q, SPEC_TITLE)
    assert {"vlm", "guided", "robot", "action", "prediction"} <= set(q)
    assert not {"retrieval", "augmented", "generation", "rag"} & set(q)  # generic acronym match, no topic list
    assert asc.is_specific(Q, SPEC_TITLE, SCOPE)
    assert asc.title_qualifiers(Q, GEN_TITLES[1]) == ["guide", "pipelines"]
    assert not asc.is_specific(Q, GEN_TITLES[1], SCOPE)
    assert asc.title_qualifiers("What are components of retrieval augmented generation?", SPEC_TITLE)[:1] == ["vlm"]


def test_distinct_documents_counts_documents_not_passages():
    two_from_one = [span(1, SPEC_TITLE, SPEC_TXT), dict(span(1, SPEC_TITLE, SPEC_TXT), span_id="s1b")]
    assert asc.distinct_documents(two_from_one) == 1
    assert asc.distinct_documents([span(i, "T", "x") for i in range(6)]) == 6


# ---------------------------------------------------------------------------------------------------- (a) the floor
FCFG = {"sufficiency": {"partial_is_insufficient": True, "min_distinct_documents": 6}}


def answered(n_docs, completeness="complete"):
    ev = bundle([span(i, f"T{i}", "x") for i in range(n_docs)])
    return AskResult("a [1]", 1, 0.0, result_kind="answer", evidence=ev,
                     claim_check={"answer_completeness": completeness})


def test_one_document_answer_is_insufficient_so_research_runs():
    assert rr.sufficient(answered(1), FCFG) is False
    assert rr.triggers(rr.sufficient(answered(1), FCFG), {}) == [rr.TRIGGER_INSUFFICIENT]


def test_six_documents_answer_is_sufficient_and_floor_off_is_previous_behaviour():
    assert rr.sufficient(answered(6), FCFG) is True
    assert rr.sufficient(answered(1), {"sufficiency": {"partial_is_insufficient": True}}) is True


def test_limited_evidence_note_when_still_below_floor(monkeypatch):
    monkeypatch.setattr("vera.fulldoc_corpus.load_config", lambda *a, **k: cfg())
    out = _with_floor_note(answered(1), FCFG)
    texts = [n["text"] for n in out.notes] + [out.answer]
    assert any("Note: 1 full document, fewer than 6." in t for t in texts)
    assert _with_floor_note(answered(6), FCFG).notes == answered(6).notes
    once = _with_floor_note(out, FCFG)
    assert [n["text"] for n in once.notes].count("Note: 1 full document, fewer than 6.") <= 1


def test_shipped_floor_equals_acquisition_target():
    acq = json.loads((ROOT / "config" / "auto_acquire.json").read_text())
    floor = acq["freshness"]["sufficiency"]["min_distinct_documents"]
    assert floor == acq["bounds"]["acq_round_target_new_docs"] == 6
    sc = json.loads((ROOT / "config" / "fulldoc_answer.json").read_text())["answer"]["scope_check"]
    assert sc["enabled"] is True and "{n}" in sc["limited_evidence_text"] and "{titles}" in sc["specific_scope_text"]


# ------------------------------------------------------------------------------------------------- (b) writer prompt
def test_writer_gets_scope_rule_and_title_hint_only_when_on():
    ev = bundle([span(1, SPEC_TITLE, SPEC_TXT), span(2, GEN_TITLES[0], GEN1_TXT)])
    msgs = fa.messages_for(Q, fa.sources_from(ev), cfg())
    assert "SCOPE RULE." in msgs[0]["content"]
    p = json.loads(msgs[1]["content"])["passages"]
    assert "robot" in p[0]["title_terms_not_in_question"]
    off = fa.messages_for(Q, fa.sources_from(ev), cfg(scope=False))
    assert "SCOPE RULE." not in off[0]["content"]
    assert all("title_terms_not_in_question" not in x for x in json.loads(off[1]["content"])["passages"])


# -------------------------------------------------------------------------------------- (c) claim check + end to end
def test_checker_payload_carries_titles_and_scope_rule_only_when_on():
    claims = [{"id": 0, "text": "x [1]", "cites": [1]}]
    srcs = [{"n": 1, "title": SPEC_TITLE, "snippet": SPEC_TXT}]
    on = cc._messages(claims, srcs, Q, {"scope": SCOPE})
    assert cc.SCOPE_SYSTEM in on[0]["content"] and json.loads(on[1]["content"])["source_titles"] == {"1": SPEC_TITLE}
    off = cc._messages(claims, srcs, Q, {})
    assert cc.SCOPE_SYSTEM not in off[0]["content"] and "source_titles" not in json.loads(off[1]["content"])


def test_only_specific_doc_available_answer_states_limited_scope_and_is_partial():
    ev = bundle([span(1, SPEC_TITLE, SPEC_TXT)])
    text = "Parts:\nP1: What are components of RAG?\nAnswers:\nP1: Our system has four components: RAG image query optimization [1]."
    o, _ = run(text, ev, {1: "specific"})
    r = o.result
    assert r is not None and r.result_kind == "answer"
    assert r.claim_check["answer_completeness"] == "partial" and r.claim_check.get("scope_specific_only") is True
    notes = [n["text"] for n in r.notes] + [r.answer]
    assert any(t.startswith("Note: specific system (“VLM-guided") for t in notes)
    assert rr.sufficient(r, {"sufficiency": {"partial_is_insufficient": True}}) is False  # -> more search


def test_general_docs_available_answer_uses_them_without_scope_note():
    ev = bundle([span(1, SPEC_TITLE, SPEC_TXT), span(2, GEN_TITLES[0], GEN1_TXT), span(3, GEN_TITLES[1], GEN2_TXT)])
    text = ("Parts:\nP1: What are components of RAG?\nAnswers:\nP1: A RAG system combines a retriever that finds relevant "
            "passages with a generator that writes the answer from them [2]. RAG pipelines also need an index of embedded "
            "documents that the retriever searches [3].")
    o, writes = run(text, ev, {1: "specific", 2: "general", 3: "general"})
    r = o.result
    assert r.result_kind == "answer" and r.claim_check["answer_completeness"] == "complete"
    assert not r.claim_check.get("scope_specific_only")
    assert all("specific system" not in n["text"] for n in r.notes) and "specific system" not in r.answer
    assert "SCOPE RULE." in writes[0][0]["content"]


def test_mechanical_fallback_when_checker_gives_no_scope_verdict():
    ev = bundle([span(1, SPEC_TITLE, SPEC_TXT)])
    text = "Parts:\nP1: What are components of RAG?\nAnswers:\nP1: Our system has four components: RAG image query optimization [1]."
    o, _ = run(text, ev, None)
    assert not o.result.claim_check.get("scope_specific_only")  # fallback off by default
    c = cfg()
    c["answer"]["scope_check"] = dict(SCOPE, mechanical_fallback=True)
    o, _ = run(text, ev, None, c=c)
    assert o.result.claim_check.get("scope_specific_only") is True


def test_scope_off_is_previous_behaviour():
    ev = bundle([span(1, SPEC_TITLE, SPEC_TXT)])
    text = "Parts:\nP1: What are components of RAG?\nAnswers:\nP1: Our system has four components: RAG image query optimization [1]."
    seen = []
    o, _ = run(text, ev, None, c=cfg(scope=False), seen=seen)
    assert o.result.claim_check["answer_completeness"] == "complete"
    assert "source_titles" not in json.loads(seen[0][1]["content"])


def test_single_part_path_also_adds_scope_note():
    ev = bundle([span(1, SPEC_TITLE, SPEC_TXT)])
    o, _ = run("Our system has four components: RAG image query optimization [1].", ev, {1: "specific"},
               c=cfg(per_part=False))
    r = o.result
    assert r.claim_check["answer_completeness"] == "partial"
    assert any("specific system" in t for t in [n["text"] for n in r.notes] + [r.answer])
