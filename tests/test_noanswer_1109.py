"""NOANSWER-1109: the no-answer page's evidence statements agree (live build 731863f, midterms and hadrons cases)."""
import pytest

from vera import evidence_trace_ui as etu
from vera import research_outcome as ro
from vera import ui_evidence as u

TOK = "insufficient_full_document_evidence"
NOT_ANSWER = "selected, but judged not to answer the question"


class Cap:
    def __init__(self):
        self.lines = []

    def caption(self, t):
        self.lines.append(t)

    markdown = info = caption


def _midterms():
    """Live shape (731863f): 2 stored documents reused, 0 fresh admitted, nothing saved this request (no "saved" clause);
    9 other documents fetched and not kept; 6 passages from the reused documents selected, none cited."""
    return {"evidence_status": "insufficient", "corpus_source": {"kind": "db"},
            "examined_full_documents": [{"document_id": f"d{i}", "candidate_spans": 0, "relevant_spans": 0,
                                         "admitted_spans": 0} for i in range(5)],
            "admitted_evidence": [],
            "contract": {"fallback": TOK, "search_counts": {"matched": 0}, "trace_attempt": "fetched",
                         "inline_acquisition": {
                             "reused": 2, "n_persisted": 0, "admitted": 0, "documents": [],
                             "counts": {"stored": 2, "fetched": 0},
                             "acquisition": {"status": "relevance_rejected", "target": 6},
                             "ledger": {"counted": 2, "full_text": 2, "partial_text": 0, "metadata_only": 0,
                                        "examined_unqualified": 9, "duplicates": 0, "stored": 0, "scholarly": 2,
                                        "news": 0, "official": 0, "web": 0}},
                         "trace": {"view": "public", "origin": "fetched",
                                   "passages": [{"n": i + 1, "cited": False, "reason": NOT_ANSWER} for i in range(6)]}}}


def _hadrons():
    d = _midterms()
    d["contract"]["trace"] = {"view": "public", "passages": []}
    d["contract"].pop("trace_attempt")
    d["contract"]["inline_acquisition"].update(reused=0, n_persisted=0, admitted=0, counts={"stored": 0, "fetched": 0})
    return d


def test_midterms_source_attribution_is_consistent():
    d = _midterms()
    scope, src, cnt = u.evidence_scope_line(d), u.evidence_source_line(d), u.inline_counts_line(d)
    note = u.acquisition_notice_line(d)
    assert scope == ("Evidence scope: 2 stored documents were checked for this question; 6 of their passages were "
                     "selected as possible evidence; none was used in an answer.")
    assert "no stored document relevant" not in scope and "fetched for this question" not in src
    head = u.fallback_text(TOK, n_inspected=5, n_matched=u.matched_for(d), n_stored_checked=u.stored_checked(d))
    assert head.startswith("2 stored documents were checked for this question") and "No stored full document" not in head
    assert src.startswith("Evidence source: documents already stored, checked for this question")
    assert "stored database" not in src
    assert "2 already stored" in cnt and "0 newly fetched" in cnt and "used for this answer" not in cnt
    assert note.startswith("VERA-DAR obtained 2 documents for this question (aim: at least 6); 9 other documents fetched did not have enough relevant passages to keep")
    assert "saved to VERA-DAR" not in note


def test_fresh_documents_are_counted_as_fetched_not_minus_reused():
    from vera import ask_service as a
    import inspect
    assert '"fetched": n_ad}' in inspect.getsource(a)  # admitted = fresh only; disjoint from reused


def test_hadrons_no_stored_database_source_when_nothing_was_used():
    d = _hadrons()
    assert "no passages selected" in u.evidence_scope_line(d)
    assert u.evidence_source_line(d) is None
    d["contract"]["inline_acquisition"]["ledger"].update(counted=0, full_text=0)
    assert u.acquisition_notice_line(d).startswith("VERA-DAR obtained 0 documents for this question (aim: at least 6)")
    assert u.stored_checked(d) == 0


def test_identical_passage_lines_collapse_to_one_without_naming_sources():
    cap = Cap()
    assert etu.render_trace(cap, _midterms()["contract"]["trace"], public=True)
    text = "\n".join(cap.lines)
    assert "6 passages were selected and none answered the question." in cap.lines
    assert text.count("selected, not cited") == 0 and "[1]" not in text and "http" not in text


def test_distinct_reasons_keep_their_own_lines():
    tr = {"view": "public", "passages": [{"n": 1, "cited": False, "reason": "a"}, {"n": 2, "cited": False, "reason": "b"}]}
    cap = Cap()
    etu.render_trace(cap, tr, public=True)
    assert "[1] selected, not cited: a" in cap.lines and "[2] selected, not cited: b" in cap.lines


def test_outcome_is_short_separate_sentences():
    attempts = [ro.PortalAttempt("a", "error"), ro.PortalAttempt("b", "quota"), ro.PortalAttempt("c", "paced")]
    r = ro.summarize_outcome(attempts, [], budget_exhausted=False, older_evidence_available=False,
                             new_evidence_admitted=False, source_unreachable=True)
    t = r.user_text
    assert t.startswith("The search was incomplete. ")
    assert " and VERA-DAR" not in t and "had a failure, " not in t
    for s in ("A search service had a failure.", "A search service's request limit was reached.",
              "VERA-DAR skipped a call to a source because the wait needed to respect that source's rate limit was too long.", "No conclusion about"):
        assert s in t
    assert max(len(x) for x in t.split(". ")) < 130


def test_outcome_causes_are_per_request_not_shared_state():
    bad = ro.summarize_outcome([ro.PortalAttempt("a", "error")], [], budget_exhausted=False,
                               older_evidence_available=False, new_evidence_admitted=False)
    ok = ro.summarize_outcome([ro.PortalAttempt("a", "completed", 3)], [], budget_exhausted=False,
                              older_evidence_available=False, new_evidence_admitted=False)
    bad2 = ro.summarize_outcome([ro.PortalAttempt("a", "quota")], [], budget_exhausted=False,
                                older_evidence_available=False, new_evidence_admitted=False)
    assert "failure" in bad.user_text and "failure" not in ok.user_text
    assert "request limit" in bad2.user_text and "failure" not in bad2.user_text


def test_a4_unknown_status_is_not_reported_as_a_failure():
    from vera import research_record as rr
    inline = {"status": "ok", "portals": [{"portal": "a", "status": "something_new", "results": 0},
                                          {"portal": "b", "status": "completed", "results": 4}], "candidates": []}
    rec = rr.summarize(triggers=[rr.TRIGGER_INSUFFICIENT], inline=inline, used_new=False, older_available=False,
                       fr_before=None)
    t = rec["user_text"] if "user_text" in rec else str(rec)
    assert "had a failure" not in t and "could not interpret" in t


def test_a4_no_failing_call_shows_no_failure_and_paced_is_worded_accurately():
    ok = ro.summarize_outcome([ro.PortalAttempt("a", "completed", 2)], [], budget_exhausted=False,
                              older_evidence_available=False, new_evidence_admitted=False)
    assert "failure" not in ok.user_text
    p = ro.summarize_outcome([ro.PortalAttempt("a", "paced")], [], budget_exhausted=False,
                             older_evidence_available=False, new_evidence_admitted=False).user_text
    assert "skipped a call" in p and "wait" in p and "paced its requests" not in p
    c = ro.summarize_outcome([ro.PortalAttempt("a", "own_cap")], [], budget_exhausted=False,
                             older_evidence_available=False, new_evidence_admitted=False).user_text
    assert "earlier searches" in c


# ---- answered page (live 731863f, "What are components of RAG?") ------------------------------------------------
def _answered_cites():
    at = {"creator": "A. Author", "title": "RAG survey", "source": "https://example.org/rag", "licence": "CC BY 4.0",
          "licence_url": "https://creativecommons.org/licenses/by/4.0/"}
    base = {"verdict": "supported", "quote": "retriever and generator", "quote_span": "chars=1-20",
            "canonical_source_id": "doc-3", "attribution": at}
    return [dict(base, n=3, claim_id="c1"), dict(base, n=3, claim_id="c2"),  # same passage cited by two claims
            dict(base, n=4, claim_id="c3", quote="index", quote_span="chars=30-40")]  # same document, other passage


def test_answered_citation_lines_not_repeated_one_attribution_per_document():
    cap = Cap()
    u.render_citations(cap, {"citations": _answered_cites()})
    text = "\n".join(cap.lines)
    assert text.count("[3] supported") == 1 and text.count("[4] supported") == 1
    attrs = [x for x in cap.lines if "RAG survey" in x]
    assert len(attrs) == 1, attrs


def test_answered_trace_lists_a_twice_cited_passage_once_and_gives_a_reason():
    tr = {"view": "public", "passages": [
        {"n": 3, "cited": True, "verdict": "supported", "supports": ["S."], "excerpt": None},
        {"n": 3, "cited": True, "verdict": "supported", "supports": ["S."], "excerpt": None},
        {"n": 2, "cited": False, "reason": "the answer did not use this passage"}]}
    cap = Cap()
    etu.render_trace(cap, tr, public=True)
    assert sum(1 for x in cap.lines if x.startswith("[3] supported")) == 1
    assert "no reason recorded" not in "\n".join(cap.lines)
    from vera import evidence_trace
    assert "no reason recorded" not in evidence_trace.load_labels()["not_cited"]["unused"]


def test_reused_document_is_not_newly_found_even_when_the_held_snapshot_lags():
    from vera import ask_service as a
    adm = [{"document_id": "d9", "canonical_source_id": "https://doi.org/10.1234/abc"}]
    pub = {"candidates": [{"outcome": "reused", "doi": "10.1234/abc"}]}
    assert a._new_doc_ids(adm, held_keys=()) == {"d9"}  # the stale snapshot alone says new
    assert a._new_doc_ids(adm, held_keys=(), reused_keys=a._reused_keys(pub)) == set()
    assert a._new_doc_ids(adm, held_keys=(), none_fresh=a._nothing_fresh({"admitted": 0})) == set()  # nothing fresh admitted
    assert a._nothing_fresh({"admitted": 2}) is False and a._nothing_fresh({}) is False
    other = [{"document_id": "d8", "canonical_source_id": "https://doi.org/10.1234/zzz"}]
    assert a._new_doc_ids(other, held_keys=(), reused_keys=a._reused_keys(pub)) == {"d8"}


def test_answered_page_counts_and_freshness_agree_for_a_reused_document():
    d = {"citations": [], "contract": {"inline_acquisition": {"counts": {"stored": 1, "fetched": 0}}}}
    assert u.inline_counts_line(d) == "Sources used for this answer: 1 stored, 0 newly fetched."
    r = ro.summarize_outcome([ro.PortalAttempt("a", "completed", 2)], [], budget_exhausted=False,
                             older_evidence_available=True, new_evidence_admitted=False)
    assert "newly found" not in r.user_text


def test_coverage_causes_are_separate_sentences():
    r = ro.summarize_outcome([ro.PortalAttempt("a", "error"), ro.PortalAttempt("b", "quota")], [],
                             budget_exhausted=False, older_evidence_available=False, new_evidence_admitted=True)
    assert r.user_text.startswith("This answer uses newly found sources. Coverage was incomplete. A search service had a failure. ")
    assert " because " not in r.user_text


def test_wording_singular_scope_and_no_other_when_none_read():
    d = _midterms()
    d["contract"]["inline_acquisition"].update(reused=1, counts={"stored": 1, "fetched": 0})
    scope = u.evidence_scope_line(d)
    assert "1 stored document was checked" in scope and "6 of its passages were" in scope and "their passages" not in scope
    d = _midterms()
    led = d["contract"]["inline_acquisition"]["ledger"]
    led.update(counted=0, full_text=0, scholarly=0)
    note = u.acquisition_notice_line(d)
    assert note.startswith("VERA-DAR obtained 0 documents for this question (aim: at least 6); 9 documents fetched did not have") and "other" not in note
