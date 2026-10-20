from tests.m45_support import FakeLLM, corpus  # noqa: F401
from vera.pipeline_llm import LLMError
from vera.m5.claim_verification import verify_claims
from vera.m5.revision import revise_response, revise_response_detailed


def resp(*claims, text="Answer text."):
    return {"response_text": text, "claims": [
        {"claim_text": t, "evidence_span_ids": ids} for t, ids in claims]}


def llm_verdicts(*vs):
    return FakeLLM(strict={"verdicts": [
        {"index": i, "verdict": v, "issue": "x"} for i, v in enumerate(vs)]})


def test_supported_claim_passes(corpus):
    r = verify_claims(resp(("Developers completed the task 55.8% faster in a controlled experiment", ["s1"])),
                      corpus, llm=llm_verdicts("supported"))
    assert r["verification_status"] == "verified" and not r["unsupported_claims"]


def test_hallucinated_number_caught_deterministically(corpus):
    r = verify_claims(resp(("Developers completed the task 80% faster in a controlled experiment", ["s1"])),
                      corpus, llm=llm_verdicts("supported"))  # even if the LLM is fooled
    assert r["unsupported_claims"][0]["status"] == "unsupported"
    assert r["unsupported_claims"][0]["recommended_action"] == "remove"
    assert r["needs_revision"]


def test_missing_or_fabricated_citation_caught(corpus):
    r = verify_claims(resp(("Something about quality", []), ("Something about security", ["nope"])),
                      corpus, llm=llm_verdicts("supported", "supported"))
    assert [u["status"] for u in r["unsupported_claims"]] == ["unsupported", "unsupported"]


def test_overgeneralisation_flagged_as_overreach(corpus):
    r = verify_claims(resp(("AI assistants always make every developer faster", ["s1"])),
                      corpus, llm=llm_verdicts("partial"))
    assert r["unsupported_claims"][0]["status"] == "overreach"
    assert r["unsupported_claims"][0]["recommended_action"] == "tone_down"


def test_outdated_only_support_flagged_weak(corpus):
    corpus["spans"][0]["year"] = 2020
    r = verify_claims(resp(("Developers completed the task faster in a controlled experiment", ["s1"])),
                      corpus, llm=llm_verdicts("supported"))
    assert r["unsupported_claims"][0]["status"] == "weak"


def test_selective_use_of_contested_evidence_flagged(corpus):
    rel = {"edges": [{"span1_id": "s2", "span2_id": "s1", "relation_type": "refutes", "confidence": 0.9}]}
    claim = ("Developers completed the task 55.8% faster in a controlled experiment", ["s1"])
    flagged = verify_claims(resp(claim), corpus, relations=rel, llm=llm_verdicts("supported"))
    assert flagged["unsupported_claims"][0]["status"] == "contested"
    ok = verify_claims(resp(claim, text="However, another trial found the opposite."), corpus,
                       relations=rel, llm=llm_verdicts("supported"))
    assert not ok["unsupported_claims"]


def test_llm_failure_is_never_reported_as_verified(corpus):
    def boom(s, u):
        raise LLMError("down")
    r = verify_claims(resp(("Developers completed the task 55.8% faster in a controlled experiment", ["s1"])),
                      corpus, llm=boom)
    assert r["verification_status"] == "unverified"
    assert r["verified_response"]["claims"][0]["verification_status"] == "unverified"


def test_input_response_not_mutated(corpus):
    original = resp(("x y z", ["s1"]))
    verify_claims(original, corpus, llm=llm_verdicts("unsupported"))
    assert "verification_status" not in original["claims"][0]


def test_revision_single_pass_returns_text(corpus):
    flagged = [{"claim_index": 0, "claim_text": "bad", "issues": ["i"], "recommended_action": "remove"}]
    llm = FakeLLM(revise={"revised_text": "Clean answer.", "actions": [{"claim_index": 0, "action": "removed"}]})
    assert revise_response(resp(("bad", ["s1"])), flagged, llm=llm) == "Clean answer."
    assert len(llm.calls) == 1
    assert revise_response_detailed(resp(("bad", ["s1"])), flagged, llm=llm)["actions"][0]["action"] == "removed"
