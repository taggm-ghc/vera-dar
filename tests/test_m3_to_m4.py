"""S2 tests: M3->M4 adapter + tagger. Fakes only: no DB, no network, no LLM/provider."""
import json
from pathlib import Path

import pytest

import vera.m3_to_m4 as m
from vera.m3_to_m4 import AdapterError, adapt_m3_to_m4, coverage_table, tag_spans
from vera.m4.common import MAX_UNITS
from vera.m4.gate_c import evaluate_adequacy
from vera.pipeline_llm import UNTRUSTED_NOTE, LLMError, LLMResult

FREEZE = Path(__file__).resolve().parent.parent / "config" / "vera_eval_freeze.json"
REQS = json.loads(FREEZE.read_text())["requirements"]  # read only
IDS = [r["id"] for r in REQS]


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real LLM path reached in a test")
    monkeypatch.setattr("vera.pipeline_llm.complete_json", boom)


def res(data):
    return LLMResult(data=data, model="fake", tokens=1, cost_usd=0.0)


class FakeTagger:
    """Tags label S<i> in each prompt using `plan(i_global) -> list[str]`; records prompts."""
    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def __call__(self, system, user):
        self.calls.append((system, user))
        labels = [ln.split(":")[0] for ln in user.split("<evidence>")[1].splitlines() if ln.startswith("S")]
        return res({"tags": [{"span": lab, "req_ids": self.plan(lab, user)} for lab in labels]})


def sp(sid, rel=0.5, text=None):
    return {"span_id": sid, "text": f"text {sid}" if text is None else text, "relevance_score": rel,
            "evidence_type": "rct", "start_index": 0, "end_index": 5}


def corpus(*sources):
    return {"counts": {"admit": len(sources)}, "failures": [], "sources": list(sources)}


def src(sid, spans, **kw):
    return {"source_id": sid, "title": f"T{sid}", "overall_quality": 4.0, "spans": spans, **kw}


def test_flatten_order_and_source_fields():
    c = corpus(src(1, [sp(10, 0.2), sp(11, 0.9)], year=2024, source_type="rct"), src(2, [sp(20, 0.5)]))
    orig = json.dumps(c, sort_keys=True)
    out = adapt_m3_to_m4(c, REQS, llm=FakeTagger(lambda lab, u: ["req_rct"]))
    assert [s["span_id"] for s in out["spans"]] == [11, 20, 10]
    s11 = out["spans"][0]
    assert (s11["source_id"], s11["source_title"], s11["year"], s11["source_type"]) == (1, "T1", 2024, "rct")
    assert json.dumps(c, sort_keys=True) == orig  # input not mutated


def test_sub_questions_are_the_frozen_requirements():
    out = adapt_m3_to_m4(corpus(src(1, [sp(1)])), REQS, llm=FakeTagger(lambda l, u: ["req_rct"]))
    assert [q["id"] for q in out["sub_questions"]] == IDS and len(IDS) == 8
    assert all(q["required"] is True for q in out["sub_questions"])
    assert out["tagging"]["coverage"].keys() == set(IDS)


def test_unknown_ids_dropped_and_counted():
    t = FakeTagger(lambda l, u: ["req_rct", "req_BOGUS", "sq_controlled"])
    out = adapt_m3_to_m4(corpus(src(1, [sp(1)])), REQS, llm=t)
    assert out["spans"][0]["sub_question_ids"] == ["req_rct"]
    assert out["tagging"]["dropped_ids"] == ["req_BOGUS", "sq_controlled"]


def test_only_unknown_ids_means_untagged_and_raises():
    with pytest.raises(AdapterError, match="no tags"):
        adapt_m3_to_m4(corpus(src(1, [sp(1)])), REQS, llm=FakeTagger(lambda l, u: ["nope"]))


def test_untagged_spans_reported_not_assigned():
    t = FakeTagger(lambda lab, u: ["req_rct"] if lab == "S1" else [])
    out = adapt_m3_to_m4(corpus(src(1, [sp(1, 0.9), sp(2, 0.1)])), REQS, llm=t)
    assert out["spans"][1]["sub_question_ids"] == []
    assert out["tagging"]["untagged_spans"] == [2]


def test_excess_requirement_ids_per_span_capped_and_counted():
    t = FakeTagger(lambda l, u: IDS[:6])
    out = adapt_m3_to_m4(corpus(src(1, [sp(1)])), REQS, llm=t)
    assert out["spans"][0]["sub_question_ids"] == IDS[: m.TAG_MAX_REQ_PER_SPAN]
    assert len(out["tagging"]["dropped_ids"]) == 6 - m.TAG_MAX_REQ_PER_SPAN


def test_empty_spans_raise():
    with pytest.raises(AdapterError, match="0 spans"):
        adapt_m3_to_m4(corpus(src(1, [])), REQS, llm=FakeTagger(lambda l, u: []))
    with pytest.raises(AdapterError, match="0 spans"):
        adapt_m3_to_m4({"sources": []}, REQS, llm=FakeTagger(lambda l, u: []))


def test_none_span_id_raises_and_no_llm_call():
    t = FakeTagger(lambda l, u: ["req_rct"])
    with pytest.raises(AdapterError, match="span_id"):
        adapt_m3_to_m4(corpus(src(1, [sp(None)])), REQS, llm=t)
    assert t.calls == []


def test_duplicate_span_id_and_empty_text_raise():
    with pytest.raises(AdapterError, match="duplicate"):
        adapt_m3_to_m4(corpus(src(1, [sp(1), sp(1)])), REQS, llm=FakeTagger(lambda l, u: []))
    with pytest.raises(AdapterError, match="empty"):
        adapt_m3_to_m4(corpus(src(1, [sp(1, text="")])), REQS, llm=FakeTagger(lambda l, u: []))


@pytest.mark.parametrize("payload", [{"tags": "x"}, {"nope": 1}, {"tags": [1]}, {"tags": [{"span": "S1"}]},
                                     {"tags": [{"span": "S1", "req_ids": "req_rct"}]}])
def test_malformed_llm_output_raises_verbosely(payload):
    with pytest.raises(AdapterError, match="malformed"):
        adapt_m3_to_m4(corpus(src(1, [sp(1)])), REQS, llm=lambda s, u: res(payload))


def test_llm_error_propagates_with_batch_context():
    def bad(s, u):
        raise LLMError("timeout")
    with pytest.raises(LLMError, match=r"tagging batch 1 \(1 spans\) failed: timeout"):
        adapt_m3_to_m4(corpus(src(1, [sp(1)])), REQS, llm=bad)


def test_no_llm_means_no_fallback_tags(monkeypatch):
    # llm=None goes to pipeline_llm.complete_json (patched to fail here): error, never keyword tags.
    def fail(system, user):
        raise LLMError("no provider")
    monkeypatch.setattr("vera.pipeline_llm.complete_json", fail)
    with pytest.raises(LLMError, match="no provider"):
        adapt_m3_to_m4(corpus(src(1, [sp(1, text="randomized controlled trial of RCT speed")])), REQS)


def test_unknown_span_label_counted():
    def llm(s, u):
        return res({"tags": [{"span": "S1", "req_ids": ["req_rct"]}, {"span": "S99", "req_ids": ["req_rct"]}]})
    out = adapt_m3_to_m4(corpus(src(1, [sp(1)])), REQS, llm=llm)
    assert out["tagging"]["unknown_labels"] == 1


def test_cap_overflow_batching_and_skipped():
    spans = [sp(i, rel=1 - i / 1000) for i in range(1, 11)]
    t = FakeTagger(lambda l, u: ["req_rct"])
    r = tag_spans(spans, REQS, llm=t, batch_size=3, max_batches=2)
    assert len(t.calls) == 2 and r.batches == 2
    assert r.skipped_spans == [7, 8, 9, 10]
    assert sorted(r.tags) == [1, 2, 3, 4, 5, 6]
    out = adapt_m3_to_m4(corpus(src(1, spans)), REQS, tagger=lambda s, q, llm=None: r)
    assert [s["sub_question_ids"] for s in out["spans"][6:]] == [[]] * 4  # skipped: untagged, not assigned
    assert out["tagging"]["skipped_spans"] == [7, 8, 9, 10]


def test_beyond_m4_max_units_reported(caplog):
    n = MAX_UNITS + 5
    spans = [sp(i, rel=1 - i / 1000) for i in range(1, n + 1)]
    caplog.set_level("WARNING", logger="vera")
    out = adapt_m3_to_m4(corpus(src(1, spans)), REQS, llm=FakeTagger(lambda l, u: ["req_rct"]))
    assert len(out["spans"]) == n  # kept for Gate C
    assert out["tagging"]["beyond_m4_max_units"] == list(range(MAX_UNITS + 1, n + 1))
    assert out["tagging"]["m4_max_units"] == MAX_UNITS
    assert any("MAX_UNITS" in r.message for r in caplog.records)


def test_prompt_has_untrusted_note_truncation_and_requirements():
    long = "x" * 1000
    t = FakeTagger(lambda l, u: ["req_rct"])
    adapt_m3_to_m4(corpus(src(1, [sp(1, text=long)])), REQS, llm=t)
    system, user = t.calls[0]
    assert UNTRUSTED_NOTE in system
    assert "x" * m.TAG_SPAN_CHARS in user and "x" * (m.TAG_SPAN_CHARS + 1) not in user
    assert all(i in user for i in IDS)


def test_appraisal_not_carried_by_default_and_mapped_when_opted_in():
    c = corpus(src(1, [sp(1)], overall_quality=3.0), src(2, [sp(2, 0.4)], overall_quality=None))
    t = FakeTagger(lambda l, u: ["req_rct"])
    out = adapt_m3_to_m4(c, REQS, llm=t)
    assert all("appraisal" not in s for s in out["spans"])
    out = adapt_m3_to_m4(c, REQS, llm=t, carry_appraisal=True)
    by = {s["span_id"]: s for s in out["spans"]}
    assert by[1]["appraisal"] == {"overall": 0.5}
    assert "appraisal" not in by[2]  # None is never invented


def test_coverage_table_lists_all_requirements():
    out = adapt_m3_to_m4(corpus(src(1, [sp(1), sp(2)])), REQS,
                         llm=FakeTagger(lambda l, u: ["req_rct", "req_field"]))
    tab = coverage_table(out)
    assert tab["req_rct"] == 2 and tab["req_field"] == 2 and tab["req_limits"] == 0 and len(tab) == 8


def test_gate_c_sees_tagged_spans_not_missing():
    spans_a = [sp(1, 0.9), sp(2, 0.8)]
    spans_b = [sp(3, 0.7), sp(4, 0.6)]
    c = corpus(src(1, spans_a, year=2025), src(2, spans_b, year=2025))
    # relevance order is span 1,2,3,4 -> labels S1..S4
    t = FakeTagger(lambda lab, u: (["req_rct", "req_field", "req_quality"] if lab in ("S1", "S3")
                                   else ["req_slowdown", "req_perception", "req_limits"]))
    out = adapt_m3_to_m4(c, REQS, llm=t)
    gate = evaluate_adequacy(out, "q", max_searches=0, llm=None)
    cov = gate["coverage"]
    assert cov["req_rct"]["status"] == "covered" and cov["req_rct"]["independent_sources"] == 2
    assert cov["req_moderators"]["status"] == "missing"
    assert gate["decision"] == "adequate" and gate["qualified"] is True
