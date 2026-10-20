"""PLAN-FDA D8 = A per-part answering. Stubbed model and checker, synthetic text only, no network, no DB."""
import json

import pytest

from vera import fulldoc_answer as fa
from vera.ask_service import AskResult
from vera.fulldoc_evidence import EvidenceBundle

SRC1 = "Alpha systems reduce latency by forty percent in field trials. Beta caching doubles throughput under heavy load."
SRC2 = "Gamma replication tolerates two node failures without data loss. Delta indexing halves query time on large tables."
MARK = "INSUFFICIENT_EVIDENCE"
PP = {"enabled": True, "system_prompt": "Parts then answers. Max {max_parts}. Marker {insufficient_marker}.\n{sources}",
      "partial_statement": "PARTIAL ANSWER: only some parts could be answered.",
      "part_heading": "Part {n} ({part}):",
      "unanswered_line": "Part {n} ({part}): the documents VERA-DAR holds do not support an answer to this part.",
      "partial_result_kind": "answer"}
CFG = {"answer": {"system_prompt": "legacy\n{sources}", "quote_max_chars": 200, "partial_marker": " (PARTIAL)",
                  "removed_note": "Note: removed.", "unchecked_note": "Note: unchecked.", "insufficient_marker": MARK,
                  "max_total_quote_chars": 700, "max_parts": 4, "per_part": PP},
       "fallback_tokens": {"insufficient": "ins", "model_insufficient": "model_ins", "corpus_unusable": "cu",
                           "no_valid_citations": "nvc", "checker_unavailable": "cku", "claim_check_insufficient": "cci",
                           "claim_check_disabled": "ccd"}}
GCFG = {"removed_citation_note": "Note: bad citations removed.",
        "claim_check": {"enabled": True, "timeout_s": 5, "max_claims": 40,
                        "checkers": [{"provider": "t", "model": "m", "api_key_env": "FD_TEST_KEY", "base_url": None}]}}


def span(i, text):
    attr = {"creator": f"Author {i}", "title": f"T{i}", "source": f"https://x.org/{i}", "licence": "CC-BY-4.0",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/", "changes": "Short excerpt."}
    return {"span_id": f"s{i}", "document_id": f"d{i}", "canonical_source_id": f"https://x.org/{i}", "title": f"T{i}",
            "url": f"https://x.org/{i}", "published": "2024", "authors": [], "licence": "CC-BY-4.0", "start": 0,
            "end": len(text), "text": text, "text_sha256": "ab" * 32, "relevance": 0.9, "attribution": attr}


def bundle():
    docs = ({"document_id": "d1", "canonical_source_id": "https://x.org/1", "title": "T1", "url": "https://x.org/1", "doi": None,
             "published": "2024", "licence": "CC-BY-4.0", "licence_decision": "allow", "text_sha256": "cd" * 32, "chars": 9,
             "chunks_inspected": 1, "candidate_spans": 1, "admitted_spans": 2},)
    return EvidenceBundle(question="q", examined_full_documents=docs, admitted=(span(1, SRC1), span(2, SRC2)), status="admitted")


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("FD_TEST_KEY", "x")


def reply(parts, answers):
    return ("Parts:\n" + "\n".join(f"P{i}: {t}" for i, t in enumerate(parts, 1)) + "\nAnswers:\n"
            + "\n".join(f"P{i}: {a}" for i, a in enumerate(answers, 1)))


def run(text, verdict_by_text=None, default="supported", cfg=CFG, ev=None):
    """verdict_by_text: {substring of claim -> verdict}."""
    vb = verdict_by_text or {}

    def check(chk, msgs):
        claims = json.loads(msgs[1]["content"])["claims"]
        return json.dumps({"claims": [{"id": c["id"], "verdict": next((v for k, v in vb.items() if k in c["text"]), default),
                                       "reason": "r"} for c in claims]})
    return fa.answer("What reduces latency and what tolerates failures?", ev or bundle(), "m", None, chained={},
                     grounding_cfg=GCFG, cfg=cfg, complete_fn=lambda *a, **k: AskResult(text, 10, 0.01), check_call=check,
                     reference_texts=[SRC1, SRC2])


A1, A2 = "Alpha systems cut latency a lot in trials [1].", "Gamma replication survives node failures [2]."
TWO = ["What reduces latency?", "What tolerates failures?"]


def test_full_all_parts_supported_is_answer():
    o = run(reply(TWO, [A1, A2]))
    r = o.result
    assert o.fallback is None and r.result_kind == "answer" and r.claim_check["answer_completeness"] == "complete"
    assert "Part 1 (What reduces latency?):" in r.answer and "Part 2 (What tolerates failures?):" in r.answer
    assert "do not support an answer" not in r.answer and "PARTIAL ANSWER" not in r.answer
    assert r.claim_check["parts"] == {"total": 2, "answered": [1, 2], "unanswered": []}


def test_partial_one_part_unsupported_is_marked_and_never_full():
    o = run(reply(TWO, [A1, A2]), {"Gamma": "unsupported"})
    r = o.result
    assert o.fallback is None and r.claim_check["answer_completeness"] == "partial"
    assert r.answer.startswith(PP["partial_statement"])
    assert "Alpha systems" in r.answer and "Gamma replication" not in r.answer
    assert "Part 2 (What tolerates failures?): the documents VERA-DAR holds do not support an answer to this part." in r.answer
    assert [c["part"] for c in r.citations] == [1] and r.claim_check["parts"]["unanswered"] == [2]


def test_partial_via_marker():
    o = run(reply(TWO, [A1, MARK]))
    r = o.result
    assert r.claim_check["parts"] == {"total": 2, "answered": [1], "unanswered": [2]}
    assert "do not support an answer to this part" in r.answer and r.answer.startswith(PP["partial_statement"])


def test_none_supported_falls_back_without_text():
    o = run(reply(TWO, [A1, A2]), default="unsupported")
    assert o.result is None and o.fallback == "cci"


def test_d7_partial_only_parts_not_answered_and_all_partial_falls_back():
    o = run(reply(TWO, [A1, A2]), default="partial")
    assert o.result is None and o.fallback == "cci"
    o = run(reply(TWO, [A1, A2]), {"Gamma": "partial"})  # part 2 has only a partial claim: NOT answered
    assert o.result.claim_check["parts"]["unanswered"] == [2] and "Gamma replication" not in o.result.answer
    assert o.result.claim_check["answer_completeness"] == "partial"


def test_never_full_partial_response_always_has_statement():
    for verd in ({"Gamma": "unsupported"}, {"Alpha": "unsupported"}, {"Gamma": "partial"}):
        r = run(reply(TWO, [A1, A2]), verd).result
        assert r.claim_check["answer_completeness"] == "partial" and PP["partial_statement"] in r.answer
    cfg = {**CFG, "answer": {**CFG["answer"], "per_part": {**PP, "partial_result_kind": "partial_answer"}}}
    assert run(reply(TWO, [A1, MARK]), cfg=cfg).result.result_kind == "partial_answer"
    assert run(reply(TWO, [A1, A2]), cfg=cfg).result.result_kind == "answer"  # complete is always `answer`


def test_citations_carry_span_attribution_part_and_budget_holds():
    r = run(reply(TWO, [A1, A2])).result
    assert {c["part"] for c in r.citations} == {1, 2}
    assert all(c["span_id"] and c["attribution"]["licence"] == "CC-BY-4.0" for c in r.citations)
    b = r.claim_check["verbatim_budget"]
    assert b["total"] < b["cap"] and all(len(c["quote"] or "") <= 200 for c in r.citations)


def test_single_part_question_regression():
    o = run(reply(["What reduces latency?"], [A1]))
    r = o.result
    assert r.result_kind == "answer" and "Part 1" not in r.answer and r.answer.startswith("Alpha systems cut latency")
    assert r.claim_check["answer_completeness"] == "complete"


def test_more_parts_than_max_parts_reported_not_addressed():
    parts = [f"What about topic {i}?" for i in range(1, 6)]
    o = run(reply(parts, [A1, A2, "Delta indexing halves query time [2].", "Beta caching doubles throughput [1].", A1]))
    r = o.result
    assert r.claim_check["parts"]["unanswered"] == [5] and r.claim_check["answer_completeness"] == "partial"
    assert "Part 5 (What about topic 5?): the documents VERA-DAR holds do not support" in r.answer


def test_marker_substring_inside_supported_part_does_not_drop_response():
    o = run(reply(TWO, [A1.replace("a lot", f"a lot ({MARK} is a token we mention)"), A2]))
    assert o.fallback is None and o.result.claim_check["parts"]["unanswered"] == []


def test_claim_records_for_unanswered_parts_kept_internally_not_published():
    r = run(reply(TWO, [A1, A2]), {"Gamma": "unsupported"}).result
    rec = [c for c in r.claims if c["part"] == 2]
    assert rec and rec[0]["verdict"] == "unsupported" and "Gamma replication" not in r.answer
    assert all(c["part"] in (1, 2) for c in r.claims)


def test_answered_part_keeps_partial_marker_on_partial_claims():
    two = "Alpha systems cut latency a lot in trials [1]. Beta caching is fast [1]."
    r = run(reply(TWO, [two, A2]), {"Beta": "partial"}).result
    assert "Beta caching is fast [1]. (PARTIAL)" in r.answer or "(PARTIAL)" in r.answer
    assert "Alpha systems cut latency a lot in trials [1]." in r.answer
    assert "(PARTIAL)" not in r.answer.split("Part 2")[1]


def test_unparsable_output_is_single_part_p1_and_d7_still_applies():
    o = run("Alpha systems cut latency a lot in trials [1].")
    assert o.result.claim_check["parts"] == {"total": 1, "answered": [1], "unanswered": []}
    o = run("Alpha systems cut latency a lot in trials [1].", default="partial")
    assert o.result is None and o.fallback == "cci"


def test_marker_on_all_parts_is_model_insufficient_and_no_support_is_cci():
    o = run(reply(TWO, [MARK, MARK]))
    assert o.result is None and o.fallback == "model_ins"
    assert run(MARK).fallback == "model_ins"
    o = run(reply(TWO, [A1, A2]), default="unsupported")
    assert o.fallback == "cci"


def test_switch_off_keeps_whole_draft_behaviour():
    cfg = {**CFG, "answer": {**CFG["answer"], "per_part": {**PP, "enabled": False}}}
    o = run(f"{MARK} but also something [1].", cfg=cfg)
    assert o.result is None and o.fallback == "model_ins"  # legacy substring rule
    msgs = fa.messages_for("q", fa.sources_from(bundle()), cfg)
    assert msgs[0]["content"].startswith("legacy")
    assert fa.messages_for("q", fa.sources_from(bundle()), CFG)[0]["content"].startswith("Parts then answers. Max 4.")
