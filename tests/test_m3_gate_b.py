import pytest

from tests.m3_helpers import FakeLLM, domains
from vera.m3 import gate_b
from vera.m3.appraisal_rubric import RubricEngine

SPAN = [{"text": "t", "start_index": 0, "end_index": 1, "relevance_score": 1, "evidence_type": "fact"}]


def gate(spans=SPAN, **kw):
    r = RubricEngine(FakeLLM(**{"m3.appraise": domains(**kw)}))
    return gate_b.evaluate_source({"source_id": "s1"}, "q", r, spans=spans)


def test_admit():
    g = gate()
    assert g["decision"] == "admit" and g["overall_quality"] == 4.0
    assert g["rationale"].startswith("Gate B ADMIT") and g["policy"]["version"] == gate_b.POLICY_VERSION


def test_threshold_boundaries():
    # exactly 3 everywhere -> overall 3.0 -> admit (>=)
    assert gate(bias=3, inc=3, ind=3, imp=3, pub=3, app=3)["decision"] == "admit"
    # 2.9-ish overall with critical domains >=3 -> qualify
    g = gate(bias=3, inc=2, ind=3, imp=2, pub=2, app=3)
    assert g["overall_quality"] == 2.65 and g["decision"] == "qualify"
    # just under reject threshold
    assert gate(bias=2, inc=2, ind=3, imp=2, pub=2, app=2)["decision"] == "reject"  # 2.2


def test_qualify_on_limited_applicability_despite_high_overall():
    g = gate(bias=5, inc=5, ind=5, imp=5, pub=5, app=2)
    assert g["overall_quality"] >= 3 and g["decision"] == "qualify"
    assert any("applicability" in r for r in g["reasons"])


def test_unknown_critical_domain_qualifies_not_admits():
    g = gate(bias=None)
    assert g["decision"] == "qualify" and any("risk_of_bias unknown" in r for r in g["reasons"])


def test_low_known_weight_qualifies():
    g = gate(bias=4, inc=None, ind=4, imp=None, pub=None, app=4)  # known weight 0.65
    assert g["decision"] == "qualify" and any("assessable" in r for r in g["reasons"])


def test_hard_rejects_even_with_high_overall():
    assert gate(bias=1, inc=5, ind=5, imp=5, pub=5, app=5)["decision"] == "reject"
    assert gate(bias=5, inc=5, ind=1, imp=5, pub=5, app=5)["decision"] == "reject"


def test_inconsistency_alone_never_rejects():
    g = gate(inc=1)
    assert g["decision"] == "admit"


def test_no_spans_rejects_without_llm():
    r = RubricEngine(FakeLLM(**{"m3.appraise": domains()}))
    g = gate_b.evaluate_source({"source_id": "s"}, "q", r, spans=[])
    assert g["decision"] == "reject" and "no evidence spans" in g["reasons"][0] and r.llm.prompts == []


def test_all_unknown_rejects():
    g = gate(bias=None, inc=None, ind=None, imp=None, pub=None, app=None)
    assert g["decision"] == "reject"


def test_custom_policy_and_spans_from_source():
    r = RubricEngine(FakeLLM(**{"m3.appraise": domains(bias=3, inc=3, ind=3, imp=3, pub=3, app=3)}))
    strict = gate_b.GateBPolicy(admit_min=3.5)
    g = gate_b.evaluate_source({"source_id": "s", "spans": SPAN}, "q", r, policy=strict)
    assert g["decision"] == "qualify"
