import pytest

from tests.m3_helpers import FakeLLM, domains
from vera.m3 import appraisal_rubric as ar

SPANS = [{"text": "t1", "start_index": 10, "end_index": 12, "relevance_score": 0.5, "evidence_type": "fact"},
         {"text": "t2", "start_index": 40, "end_index": 42, "relevance_score": 0.9, "evidence_type": "methodology"}]


def engine(**kw):
    return ar.RubricEngine(FakeLLM(**{"m3.appraise": domains(**kw)}))


def test_weights_sum_to_one_and_cover_domains():
    assert abs(sum(ar.WEIGHTS.values()) - 1) < 1e-9 and set(ar.WEIGHTS) == set(ar.DOMAINS)


def test_overall_weighted_mean():
    a = engine(bias=5, inc=1, ind=5, imp=5, pub=5, app=5).appraise({}, "q", SPANS)
    assert a.overall_quality == round(5 * 0.9 + 1 * 0.1, 2) and a.known_weight == 1.0


def test_unknown_excluded_not_zero():
    a = engine(bias=None, inc=4, ind=4, imp=4, pub=4, app=4).appraise({}, "q", SPANS)
    assert a.score("risk_of_bias") is None and a.overall_quality == 4.0 and a.known_weight == 0.7


def test_no_spans_means_all_unknown_and_no_llm_call():
    e = engine()
    a = e.appraise({}, "q", [])
    assert a.overall_quality is None and a.known_weight == 0 and e.llm.prompts == []


def test_score_without_rationale_becomes_unknown():
    data = domains()
    data["domains"]["imprecision"]["rationale"] = " "
    ds, _ = ar.parse_domains(data, 2)
    assert ds["imprecision"].score is None and "untraceable" in ds["imprecision"].rationale


@pytest.mark.parametrize("bad", [0, 6, "x", True, float("nan"), None])
def test_invalid_scores_are_unknown(bad):
    data = domains()
    data["domains"]["applicability"]["score"] = bad
    ds, _ = ar.parse_domains(data, 2)
    assert ds["applicability"].score is None


def test_missing_domain_and_garbage_output():
    ds, flags = ar.parse_domains({"domains": {"risk_of_bias": {"score": 3, "rationale": "r"}}}, 1)
    assert ds["risk_of_bias"].score == 3 and ds["indirectness"].score is None
    ds2, _ = ar.parse_domains("garbage", 0)
    assert all(d.score is None for d in ds2.values())


def test_citations_remapped_to_source_offsets_and_bounded():
    data = domains()
    data["domains"]["risk_of_bias"]["cited_spans"] = [0, 99, "x"]  # index 0 = highest relevance span (start 40)
    a = ar.RubricEngine(FakeLLM(**{"m3.appraise": data})).appraise({}, "q", SPANS)
    assert a.domains["risk_of_bias"].cited_spans == [40]


def test_out_of_window_year_caps_indirectness():
    a = engine(ind=5).appraise({"published_date": "2019-05-01"}, "q", SPANS)
    assert a.score("indirectness") == 2 and a.notes and "2019" in a.domains["indirectness"].rationale
    b = engine(ind=5).appraise({"published_date": "2025-05-01"}, "q", SPANS)
    assert b.score("indirectness") == 5 and not b.notes


def test_rationale_text_traceable_and_prompt_has_context():
    e = engine()
    a = e.appraise({"title": "T"}, "my question", SPANS, other_evidence=[{"source": "S2", "findings": ["x"]}])
    assert "risk_of_bias=4" in a.rationale_text()
    user = e.llm.prompts[0][1]
    assert "my question" in user and "<excerpts>" in user and "S2" in user
    assert a.to_dict()["rubric_version"] == ar.RUBRIC_VERSION
