from tests.m45_support import FakeLLM, corpus, span  # noqa: F401
from vera.m4.m4_runner import run_m4
from vera.m4_m5_runner import run_m4_m5

GOOD = {"response_text": "Faster in lab [E1]. Fabricated 90% gain [E1].", "claims": [
    {"claim_text": "Developers completed the task 55.8% faster in a controlled experiment", "evidence_labels": ["E1"]},
    {"claim_text": "Developers saw a 90% gain in a controlled experiment", "evidence_labels": ["E1"]}]}


def pipeline_llm(verdicts, revised="Faster in lab [E1]."):
    return FakeLLM(
        audit={"gaps": []}, pairwise={"relations": []},
        research_answer=GOOD,
        strict={"verdicts": [{"index": i, "verdict": v, "issue": "x"} for i, v in enumerate(verdicts)]},
        revise={"revised_text": revised, "actions": [{"claim_index": 1, "action": "removed"}]})


class Store:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def f(*a, **k):
            self.calls.append((name, a, k))
            return 7 if name == "save_answer" else None
        return f


def test_research_loop_is_capped_at_two_and_never_runs_away(corpus):
    corpus["spans"] = [s for s in corpus["spans"] if "sq_quality" not in s["sub_question_ids"]]
    calls = []
    out = run_m4("1", "q", corpus, research_fn=lambda m: calls.append(m) or {"spans": []}, llm=FakeLLM(audit={"gaps": []}, pairwise={"relations": []}))
    assert len(calls) == 2 and out["gate_c_trace"] == ["search_again", "search_again", "adequate"]
    assert out["gate_c"]["qualified"] and "UNRESOLVED GAP" in out["reasoning_context"]


def test_research_adds_evidence_and_stops_early(corpus):
    corpus["spans"] = [s for s in corpus["spans"] if "sq_quality" not in s["sub_question_ids"]]
    extra = {"spans": [span("n1", "new a", ["sq_quality"]), span("n2", "new b", ["sq_quality"])]}
    out = run_m4("1", "q", corpus, research_fn=lambda m: extra, llm=FakeLLM(audit={"gaps": []}, pairwise={"relations": []}))
    assert out["gate_c_trace"] == ["search_again", "adequate"] and not out["gate_c"]["qualified"]


def test_failing_research_fn_does_not_loop(corpus):
    corpus["spans"] = corpus["spans"][:2]
    def boom(m):
        raise RuntimeError("search down")
    out = run_m4("1", "q", corpus, research_fn=boom, llm=FakeLLM(audit={"gaps": []}, pairwise={"relations": []}))
    assert out["gate_c_trace"][-1] in ("adequate", "insufficient") and len(out["gate_c_trace"]) == 2


def test_no_research_fn_still_terminates(corpus):
    corpus["spans"] = corpus["spans"][:1]
    out = run_m4("1", "q", corpus, llm=FakeLLM(audit={"gaps": []}, pairwise={"relations": []}))
    assert out["gate_c"]["decision"] == "insufficient"


def test_unsupported_claim_triggers_single_revision_and_reverification(corpus):
    st = Store()
    llm = pipeline_llm(["supported", "supported"])
    out = run_m4_m5("1", "q", corpus, llm=llm, store=st)
    assert out["revised"] and out["final_response"] == "Faster in lab [E1]."
    assert sum(1 for s, _ in llm.calls if "Revise the answer" in s) == 1  # single pass
    assert [c["claim_text"][:10] for c in out["claims"]] == ["Developers"] and out["claims"][0]["verification_status"] == "supported"
    assert out["verification_status"] == "revised_verified"
    names = [c[0] for c in st.calls]
    assert names.count("save_answer") == 2 and names[-1] == "finalize_run"
    assert [c for c in st.calls if c[0] == "save_answer"][1][2]["parent_answer_id"] == 7


def test_clean_draft_skips_revision(corpus):
    good = {"response_text": "ok [E1]", "claims": [GOOD["claims"][0]]}
    llm = FakeLLM(audit={"gaps": []}, pairwise={"relations": []}, research_answer=good,
                  strict={"verdicts": [{"index": 0, "verdict": "supported", "issue": ""}]})
    out = run_m4_m5("1", "q", corpus, llm=llm)
    assert not out["revised"] and out["verification_status"] == "verified"
