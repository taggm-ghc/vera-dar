"""DDFIX-1023 TEVAL: separate discovery/acquisition/admission/answer steps; overall + A1-A9 shown as static tables."""
from vera import research_outcome as ro
from vera import research_record as rr


def _rec(**kw):
    cands = [{"doi": "10.1/a", "outcome": "admitted", "reason": None}, {"doi": "10.1/b", "outcome": "rejected", "reason": "R-ACQ-NOLIC"}]
    for c in cands:
        ro.annotate_diag(c)
    inline = {"status": "ok", "candidates": cands, "portals": [{"portal": "p", "status": "completed", "results": 2}]}
    return rr.summarize(triggers=["insufficient"], inline=kw.pop("inline", inline), fr_before=None, **kw)


def test_steps_generated():
    s = rr.public_fields(_rec(used_new=True, older_available=False))["research_outcome"]["steps"]
    assert s["discovered"] == 2 and s["search"] == "ran" and s["acquisition"] == "attempted" and s["fetch_requests"] == 1
    assert s["admitted"] == 1 and s["answer"] == "generated_new_evidence" and s["acquisition_reason"] is None


def test_steps_abstained_and_not_attempted():
    s = rr.public_fields(_rec(used_new=False, older_available=False, inline={"status": "off"}))["research_outcome"]["steps"]
    assert s["search"] == "not_run" and s["search_reason"] == "disabled" and s["acquisition"] == "not_attempted"
    assert s["discovered"] == 0 and s["admitted"] == 0 and s["answer"] == "abstained"


def test_page_shows_overall_and_checks_as_tables():
    from streamlit.testing.v1 import AppTest

    def app():
        import sys
        sys.path.insert(0, ".")
        from vera.trace_eval.ui import render_trace_eval
        render_trace_eval(results_path="none.json", validation_path="none2.json", public_path="eval_results/trace_eval_public.json")
    at = AppTest.from_function(app, default_timeout=30).run()
    assert not at.exception
    txt = "\n".join(m.value for m in at.markdown) + "\n".join(w.value for w in at.warning)
    assert "Overall: answers that passed every applicable check" in txt and "Per check (A1-A9)" in txt
    assert len(at.table) >= 2 and "HISTORICAL" in txt


import pytest


@pytest.mark.parametrize("st,reason", [("busy", "busy"), ("budget_unknown", "budget_unknown"),
                                       ("rate", "rate_limit"), ("daily_cap", "daily_cap")])
def test_pre_search_statuses_are_not_attempted(st, reason):
    s = rr.summarize(triggers=["insufficient"], inline={"status": st, "candidates": [], "portals": []}, used_new=False,
                     older_available=False, fr_before=None)["steps"]
    assert s["search"] == "not_run" and s["search_reason"] == reason and s["acquisition"] == "not_attempted"


def test_answer_basis_conservative():
    f = lambda **k: rr.summarize(triggers=k.pop("trig", ["insufficient"]), inline={"status": "ok", "candidates": [], "portals": []},
                                 fr_before=None, **k)["steps"]["answer"]
    # used_new flagged but nothing admitted: not proven new evidence
    assert f(used_new=True, older_available=True) == "generated_existing_only"
    assert f(used_new=False, older_available=True, answer_failed=True) == "generated_existing_only"
    assert f(used_new=False, older_available=True, trig=[]) == "generated_existing_only"
    assert f(used_new=False, older_available=False) == "abstained"
    assert f(used_new=True, older_available=True, answer_basis="generated_new_evidence") == "generated_existing_only"
    assert f(used_new=False, older_available=False, answer_basis="not_attempted") == "not_attempted"


def test_caption_text_uses_admitted_not_used_as_evidence():
    import pathlib
    src = pathlib.Path("streamlit_app.py").read_text()
    assert "passed the initial relevance check; answer" in src and "retrieval" in src and "used as evidence" not in src


def _steps(status, cands, portals=()):
    for x in cands:
        ro.annotate_diag(x)
    return rr.summarize(triggers=["insufficient"], inline={"status": status, "candidates": cands, "portals": list(portals)},
                        used_new=False, older_available=False, fr_before=None)["steps"]


@pytest.mark.parametrize("st", ["deadline", "daily_cap"])
def test_found_but_never_fetched_is_not_attempted(st):
    c = [{"doi": f"10.1/{i}", "outcome": "not_evaluated", "reason": None} for i in range(3 if st == "deadline" else 2)]
    s = _steps(st, c)
    assert s["search"] == "ran" and s["acquisition"] == "not_attempted" and s["fetch_requests"] == 0
    assert s["discovered"] == len(c)


def test_search_failed_zero_found_search_ran_no_acquisition():
    s = _steps("search_failed", [])
    assert s["search"] == "ran" and s["acquisition"] == "not_attempted" and s["discovered"] == 0


def test_fetch_timeout_is_attempted():
    s = _steps("ok", [{"doi": "10.1/a", "outcome": "rejected", "reason": "R-ACQ-FETCH_TIMEOUT"}])
    assert s["acquisition"] == "attempted" and s["fetch_requests"] == 1


def test_reused_and_unselected_do_not_count_as_fetch():
    s = _steps("ok", [{"doi": "10.1/a", "outcome": "reused", "reason": None},
                      {"doi": "10.1/b", "outcome": "dropped", "reason": "R-ACQ-NOTOA"},
                      {"doi": "10.1/c", "outcome": "rejected", "reason": "R-ACQ-NODOI"}])
    assert s["acquisition"] == "not_attempted"


def test_used_new_but_nothing_admitted_no_older_is_abstained():
    s = rr.summarize(triggers=["insufficient"], inline={"status": "ok", "candidates": [], "portals": []},
                     used_new=True, older_available=False, fr_before=None)["steps"]
    assert s["answer"] == "abstained"


@pytest.mark.parametrize("code", sorted(rr._FETCH_YES))
def test_each_fetch_code_counts_as_attempted(code):
    s = _steps("ok", [{"doi": "10.1/a", "outcome": "rejected", "reason": code}])
    assert s["acquisition"] == "attempted", code


@pytest.mark.parametrize("code", ["R-ACQ-NODOI", "R-ACQ-NOTOA", "R-ACQ-DUPLICATE", "R-ACQ-KNOWN", "R-ACQ-DEFER_CAP",
                                  "R-ACQ-RATELIMITED", "R-ACQ-DEADLINE", "R-ACQ-DOC_BUDGET", "R-ACQ-CRAWL_DELAY"])
def test_no_request_codes_not_attempted(code):
    s = _steps("ok", [{"doi": "10.1/a", "outcome": "rejected", "reason": code}])
    assert s["acquisition"] == "not_attempted", code


def test_guard_deadline_without_evidence_is_search_not_run():
    s = _steps("deadline", [])
    assert s["search"] == "not_run" and s["search_reason"] == "deadline" and s["acquisition"] == "not_attempted"
