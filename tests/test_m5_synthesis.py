from tests.m45_support import FakeLLM, corpus  # noqa: F401
from vera.m4.context_builder import build_reasoning_context
from vera.m5.synthesis import parse_label_map, synthesize_response


def ctx(corpus):
    return build_reasoning_context(corpus, {"edges": []})


def test_claims_trace_to_span_ids(corpus):
    llm = FakeLLM(research_answer={
        "response_text": "Faster in lab [E1]. Slower for experts [E2].",
        "claims": [{"claim_text": "Faster in lab", "evidence_labels": ["E1"]},
                   {"claim_text": "Slower for experts", "evidence_labels": ["[E2]"]}]})
    r = synthesize_response("q", ctx(corpus), llm=llm)
    assert r["evidence_citations"] == [{"claim_text": "Faster in lab", "span_ids": ["s1"]},
                                       {"claim_text": "Slower for experts", "span_ids": ["s2"]}]
    assert r["invalid_citations"] == [] and r["tokens"] == 10


def test_invented_labels_are_rejected(corpus):
    llm = FakeLLM(research_answer={"response_text": "x", "claims": [
        {"claim_text": "c", "evidence_labels": ["E1", "E42"]}, {"claim_text": " ", "evidence_labels": []}]})
    r = synthesize_response("q", ctx(corpus), llm=llm)
    assert r["claims"][0]["evidence_span_ids"] == ["s1"] and r["invalid_citations"] == ["E42"]
    assert len(r["claims"]) == 1


def test_label_map_parses_context(corpus):
    assert parse_label_map(ctx(corpus))["E3"] == "s3"


def test_prompt_marks_context_untrusted_and_grounds_only(corpus):
    llm = FakeLLM(research_answer={"response_text": "", "claims": []})
    synthesize_response("q", ctx(corpus), llm=llm)
    sys_prompt = llm.calls[0][0]
    assert "ONLY the reasoning context" in sys_prompt and "untrusted" in sys_prompt
