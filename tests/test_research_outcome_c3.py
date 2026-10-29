import json
import re

import pytest

from vera import research_outcome as ro
from vera.research_outcome import (CandidateOutcome, PortalAttempt, reason_from_code,
                                   summarize_outcome)

NV = "No sufficiently valuable source was found across the portals searched."


def P(name, status="completed", n=0):
    return PortalAttempt(name, status, n)


def C(cid, reason):
    return CandidateOutcome(cid, {k: None for k in ro.ID_KEYS}, reason, "d")


def run(attempts, cands=(), budget=False, older=False, new=False):
    return summarize_outcome(attempts, list(cands), budget_exhausted=budget,
                             older_evidence_available=older, new_evidence_admitted=new)


def test_exhaustive_search_no_valuable_results():
    o = run([P("a"), P("b"), P("c", "skipped")])
    assert o.kind == "no_valuable_source" and o.complete
    assert NV in o.user_text
    assert o.report_detail["zero_discoveries"] is True
    assert o.report_detail["portals_skipped"] == ["c"]


@pytest.mark.parametrize("status,word", [("timeout", "timed out"), ("quota", "limit")])
def test_incomplete_coverage(status, word):
    o = run([P("a"), P("b", status)])
    assert o.kind == "incomplete_search" and not o.complete
    assert NV not in o.user_text and word in o.user_text
    assert "all portals" not in o.user_text.lower()
    assert o.report_detail["portals_completed"] == ["a"]


def test_acquisition_failure_distinguished():
    a = run([P("a", n=1)], [C("x", "no_route")])
    b = run([P("a", n=1)], [C("x", "extraction_failed")])
    assert a.kind == b.kind == "abstain"
    assert NV not in a.user_text and "full text" in a.user_text
    assert a.report_detail["candidates"][0]["terminal_reason"] == "no_route"
    assert b.report_detail["candidates"][0]["terminal_reason"] == "extraction_failed"
    assert reason_from_code("R-ACQ-NOTOA") != reason_from_code("R-ACQ-CONVERT")


@pytest.mark.parametrize("r", ["licence_rejected", "relevance_rejected", "credibility_rejected",
                               "freshness_rejected", "provenance_rejected", "injection_rejected"])
def test_admission_rejection_vs_zero(r):
    o = run([P("a", n=1)], [C("x", r)])
    assert o.kind == "no_valuable_source" and NV in o.user_text
    assert "admission criteria" in o.user_text
    assert o.report_detail["terminal_reason_counts"] == {r: 1}
    assert o.report_detail["zero_discoveries"] is False
    z = run([P("a")])
    assert "No candidates were found" in z.user_text and z.report_detail["zero_discoveries"]


def test_budget_exhaustion_is_incomplete():
    o = run([P("a"), P("b")], [C("x", "budget_exhausted")], budget=True)
    assert o.kind == "incomplete_search" and not o.complete
    assert NV not in o.user_text and "budget" in o.user_text


def test_qualified_older_only_discloses_freshness():
    o = run([P("a")], older=True)
    assert o.kind == "qualified_older_only"
    assert "freshness research found no qualifying newer evidence" in o.user_text


def test_answered_with_new():
    o = run([P("a", "timeout")], [C("x", "admitted")], new=True)
    assert o.kind == "answered_with_new" and "timed out" in o.user_text


def test_no_portals_not_complete():
    o = run([P("a", "skipped")])
    assert o.kind == "incomplete_search" and not o.complete and NV not in o.user_text


@pytest.mark.parametrize("status", ["timeout", "quota", "error"])
def test_all_portals_never_claimed_on_failure(status):
    o = run([P("a"), P("b", status)], older=False)
    assert "all portals" not in json.dumps(o.user_text).lower()
    assert not o.complete


def test_report_detail_json_and_user_text_rules():
    o = run([P("a", n=2), P("b", "error")], [C("x", "licence_rejected")], budget=True)
    json.dumps(o.report_detail)
    assert len(re.findall(r"[.!?](?:\s|$)", o.user_text)) <= 2
    assert not re.search(r"R-[A-Z]+", o.user_text)
    assert len(o.report_detail["portal_attempts"]) == 2


def test_reason_mapping_coverage():
    assert len(ro.KNOWN_CODES) == 55  # INTEG-1023: +2 ARX-A2 codes; P0DTA-DIAG: +5 (FETCH_TIMEOUT, DOC_BUDGET, RATE, DAILY, CRAWL_DELAY)
    for code in ro.KNOWN_CODES:
        assert reason_from_code(code) in ro.TERMINAL_REASONS
        assert reason_from_code(code) != "admitted"
    for bad in ("", "R-NEW-THING", None, "admitted", "R-ACQ-ADMITTED"):
        assert reason_from_code(bad) not in ("admitted",)
    assert reason_from_code("R-ACQ-INJECT") == "injection_rejected"
    assert reason_from_code("R-ABSTRACT") == "abstract_only"
    assert reason_from_code("R-ACQ-SERVECAP") == "budget_exhausted"
    assert reason_from_code("R-LIC-HOLD") == "licence_rejected"
    assert reason_from_code("R-ACQ-IRRELEVANT") == "relevance_rejected"


def test_codes_cover_source_files():
    import pathlib
    root = pathlib.Path(ro.__file__).parent
    found = set()
    for f in ("discovery_panel.py", "inline_acquire.py", "auto_acquire.py",
              "fulldoc_corpus.py", "fulldoc_db.py"):
        p = root / f
        if p.exists():
            found |= set(re.findall(r'"(R-[A-Z]+(?:-[A-Z0-9_]+)?)"', p.read_text()))
    assert found, "no source codes scanned"
    assert found <= set(ro.KNOWN_CODES), sorted(found - set(ro.KNOWN_CODES))


def test_known_code_is_already_in_corpus_not_a_rejection():  # FRESH-1023 integration tweak (verifier note 1)
    assert reason_from_code("R-ACQ-KNOWN") == "already_in_corpus"
    assert "already_in_corpus" in ro.TERMINAL_REASONS


# FIX-1028-STATUS: "newly found sources" is passage-level (new_evidence_admitted), not candidate-level.
def test_abstain_with_candidate_admitted_never_claims_new_sources():
    o = run([P("a", n=1)], [C("1", "admitted")], new=False)
    assert "newly found sources" not in o.user_text and o.kind != "answered_with_new"


def test_partial_answer_with_new_evidence_keeps_new_sources_sentence():
    o = run([P("a", n=2)], [C("1", "admitted"), C("2", "admitted")], new=True)
    assert o.kind == "answered_with_new" and o.user_text.startswith("This answer uses newly found sources")


def test_candidate_count_labels_do_not_read_as_evidence_admission():
    for lab in ro.DIAG_LABELS.values():
        assert lab not in ("used",) and "evidence" not in lab
    assert "relevance check" in ro.DIAG_LABELS["admitted"]


def test_relevance_passed_but_unused_candidates_get_accurate_sentence():
    o = run([P("a", n=1)], [C("1", "admitted")], new=False)
    assert "A document passed the initial relevance check, but none of its passages answered" in o.user_text
    o3 = run([P("a", n=2)], [C("1", "admitted"), C("2", "admitted")], new=False)
    assert "Some documents passed the initial relevance check, but none of their passages answered" in o3.user_text
    assert "admission criteria" not in o.user_text
    o2 = run([P("a", n=1)], [C("1", "relevance_rejected")], new=False)
    assert "admission criteria" in o2.user_text


def test_candidate_labels_use_shortlisted_not_selected():
    assert all("selected" not in v for v in ro.DIAG_LABELS.values())
