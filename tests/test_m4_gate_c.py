from tests.m45_support import FakeLLM, corpus, good_corpus, span  # noqa: F401
from vera.m4.gate_c import evaluate_adequacy


def test_high_coverage_is_adequate(corpus):
    g = evaluate_adequacy(corpus, "q")
    assert g["decision"] == "adequate" and not g["qualified"] and g["coverage_ratio"] == 1.0
    assert g["missing_evidence"] == []


def test_gap_triggers_search_again_with_named_need(corpus):
    corpus["spans"] = [s for s in corpus["spans"] if "sq_quality" not in s["sub_question_ids"]]
    g = evaluate_adequacy(corpus, "q")
    assert g["decision"] == "search_again"
    assert g["missing_evidence"][0]["sub_question_id"] == "sq_quality"
    assert g["missing_evidence"][0]["reason"] == "missing"


def test_redundant_sources_are_not_independent(corpus):
    for s in corpus["spans"]:
        if "sq_field" in s["sub_question_ids"]:
            s["independence_group"] = "same_origin"
    g = evaluate_adequacy(corpus, "q")
    assert g["coverage"]["sq_field"]["status"] == "thin"
    assert g["decision"] == "search_again"


def test_outdated_evidence_flagged(corpus):
    for s in corpus["spans"]:
        if "sq_quality" in s["sub_question_ids"]:
            s["year"] = 2021
    assert evaluate_adequacy(corpus, "q")["coverage"]["sq_quality"]["status"] == "outdated"


def test_cap_reached_with_partial_evidence_is_qualified_adequate(corpus):
    corpus["spans"] = [s for s in corpus["spans"] if "sq_quality" not in s["sub_question_ids"]]
    g = evaluate_adequacy(corpus, "q", searches_done=2, max_searches=2)
    assert g["decision"] == "adequate" and g["qualified"] and g["residual_gaps"]


def test_cap_reached_with_little_evidence_is_insufficient():
    c = {"spans": [span("a", "x", ["sq_field"])]}
    assert evaluate_adequacy(c, "q", searches_done=2)["decision"] == "insufficient"
    assert evaluate_adequacy({"spans": []}, "q", searches_done=0)["decision"] == "search_again"
    assert evaluate_adequacy({"spans": []}, "q", searches_done=2)["decision"] == "insufficient"


def test_llm_missingness_triggers_search_but_not_past_cap(corpus):
    llm = FakeLLM(audit={"gaps": [{"sub_question_id": "sq_field", "need": "enterprise RCT", "evidence_type": "rct"}]})
    assert evaluate_adequacy(corpus, "q", llm=llm)["decision"] == "search_again"
    g = evaluate_adequacy(corpus, "q", searches_done=2, llm=llm)
    assert g["decision"] == "adequate" and g["residual_gaps"]
