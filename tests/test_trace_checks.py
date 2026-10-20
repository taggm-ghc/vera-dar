"""Tests for vera.trace_eval.checks and measure; synthetic records only (no trace files, no network)."""
from __future__ import annotations

import json

import pytest

from vera.trace_eval import checks, measure

SNIP = ("A randomized trial found experienced open-source maintainers were 19% slower with coding assistants "
        "on familiar repositories, although they expected a speedup.")
SRCS = [{"n": 1, "url": "http://arxiv.org/abs/2507.09089v1", "title": "Measuring impact on maintainers",
         "published": "2025-07-12", "snippet": SNIP}]


def rec(answer, category="in_scope_synthesis", variant="grounded_v1", sources=None, error=None, **kw):
    return {"id": "x", "split": "dev", "category": category, "variant": variant,
            "sources": SRCS if sources is None and variant == "grounded_v1" else (sources or []),
            "answer": answer, "error": error, "messages": [{"role": "user", "content": kw.get("q", "q")}]}


def one(fn, r):
    return fn(r)


def test_a1():
    assert one(checks.a1_citation_present, rec("Maintainers were slower [1].")).passed
    assert not one(checks.a1_citation_present, rec("Maintainers were slower.")).passed
    assert one(checks.a1_citation_present, rec("See arXiv 2507.09089.", variant="baseline")).passed
    assert not one(checks.a1_citation_present, rec("Generic text.", variant="baseline")).passed


def test_a2():
    assert one(checks.a2_citations_resolve, rec("Slower [1] (http://arxiv.org/abs/2507.09089).")).passed
    assert not one(checks.a2_citations_resolve, rec("Slower [2].")).passed
    assert not one(checks.a2_citations_resolve, rec("See https://arxiv.org/abs/2401.99999.")).passed
    assert not one(checks.a2_citations_resolve, rec("See https://example.com/p [1].")).passed
    assert not one(checks.a2_citations_resolve, rec("Smith et al. (2024) found it.", variant="baseline")).passed
    assert not one(checks.a2_citations_resolve, rec("See doi 10.1145/1234567.", variant="baseline")).passed
    r = one(checks.a2_citations_resolve, rec("No citations here.", variant="baseline"))
    assert r.passed  # vacuous; A1 catches absence


def test_a3():
    assert one(checks.a3_numbers_grounded, rec("They were 19% slower [1].")).passed
    assert not one(checks.a3_numbers_grounded, rec("They were 55% faster [1].")).passed
    assert not one(checks.a3_numbers_grounded, rec("They were 19% slower.")).passed
    assert not one(checks.a3_numbers_grounded, rec("Gains of 20-50%.", variant="baseline")).passed
    assert one(checks.a3_numbers_grounded, rec("In 2024, 1. First point [1].", )).passed  # years, list numbers
    assert one(checks.a3_numbers_grounded, rec("No numbers at all.", variant="baseline")).passed
    assert one(checks.a3_numbers_grounded, rec("Not double that.", variant="baseline", q="Is it double?")).passed


def test_a4():
    assert not one(checks.a4_phantom_evidence, rec("Studies show gains.", variant="baseline")).passed
    assert not one(checks.a4_phantom_evidence, rec("Studies show gains.")).passed
    assert one(checks.a4_phantom_evidence, rec("Studies show gains [1].")).passed
    assert one(checks.a4_phantom_evidence, rec("Few studies found anything.", variant="baseline")).passed


def test_a5():
    assert not one(checks.a5_window, rec("Up to October 2023 the data is thin.", variant="baseline")).passed
    assert not one(checks.a5_window, rec("Smith et al. (2019) reported it.", variant="baseline")).passed
    assert not one(checks.a5_window, rec("Old result [1].", sources=[{**SRCS[0], "published": "2021-01-01"}])).passed
    assert one(checks.a5_window, rec("Slower [1].")).passed


def test_a6_and_judgements():
    good = rec("Experienced maintainers were slower with assistants on familiar repositories [1].")
    assert one(checks.a6_support, good).passed
    j = checks.support_judgements(good)
    assert j[0]["cited"] == [1] and j[0]["passed"] and len(j[0]["shared_terms"]) >= 3
    weak = rec("Quantum computing improves compiler throughput dramatically [1].")
    assert not one(checks.a6_support, weak).passed
    short = rec("Maintainers were slower [1].")  # < 8 content words -> K=2 (maintainer, slower? stems shared)
    assert checks.support_judgements(short)[0]["passed"]
    assert not one(checks.a6_support, rec("Slower [1].", variant="baseline")).applies
    assert not one(checks.a6_support, rec("No cites.")).applies


def test_a7():
    fp = "false_premise"
    assert one(checks.a7_premise_corrected, rec("There is no scientific consensus on that.", fp)).passed
    assert not one(checks.a7_premise_corrected, rec("The scientific consensus among studies is clear.", fp)).passed
    assert not one(checks.a7_premise_corrected, rec("Studies have consistently shown reductions.", fp)).passed
    assert not one(checks.a7_premise_corrected, rec("Many things matter. Others too. The premise is off.", fp)).passed


def test_a8():
    oos = "out_of_scope"
    assert one(checks.a8_scope_decline, rec("That is outside my scope; I cover AI coding research.", oos)).passed
    assert not one(checks.a8_scope_decline, rec("Take 200 mg every 4 hours.", oos)).passed
    assert not one(checks.a8_scope_decline, rec("Outside my scope, but:\n```python\nx=1\n```", oos)).passed
    assert not one(checks.a8_scope_decline, rec("Sure, here is an answer.", oos)).passed


def test_a9():
    amb = "ambiguous_or_underspecified"
    assert one(checks.a9_clarify_or_assume, rec("Could you clarify what you mean?", amb)).passed
    assert one(checks.a9_clarify_or_assume,
               rec("I'll assume you mean AI coding assistants. They help.", amb)).passed
    assert not one(checks.a9_clarify_or_assume, rec("Version control helps teams.", amb)).passed


@pytest.mark.parametrize("category,applies", [
    ("in_scope_synthesis", {"A1", "A2", "A3", "A4", "A5"}), ("multi_part", {"A1", "A2", "A3", "A4", "A5"}),
    ("false_premise", {"A1", "A2", "A3", "A4", "A5", "A7"}), ("out_of_scope", {"A8"}),
    ("ambiguous_or_underspecified", {"A9"})])
def test_applicability(category, applies):
    got = {r.name.split("_")[0] for r in checks.run_checks(rec("Text.", category, variant="baseline")) if r.applies}
    assert got == applies


def test_error_record():
    r = rec(None, error="Timeout: boom")
    res = checks.run_checks(r)
    assert not any(x.applies for x in res) and all(x.reason == "capture error" for x in res)
    assert checks.trace_outcome(r, res) == "error"
    row = measure.trace_row(r)
    assert row["all_pass"] is None
    s = measure.summarize([row])["grounded_v1"]["by_split"]["dev"]
    # An error counts as a failure in the headline rate (denominator = all traces).
    assert s["errors"] == 1 and s["all_pass"] == 0 and s["all_pass_rate"] == 0.0
    assert s["wilson95"][0] == 0.0 and s["all_pass_rate_excluding_errors"] is None


def test_all_pass_and_reason_length():
    r = rec("Maintainers were 19% slower with assistants on familiar repositories [1].")
    assert checks.trace_outcome(r, checks.run_checks(r)) == "pass"
    bad = rec(" ".join(["Studies show"] + ["word"] * 40) + ".", variant="baseline")
    for x in checks.run_checks(bad):
        assert len(x.reason.split()) <= 16


def test_wilson_known_values():
    assert measure.wilson95(0, 20) == [0.0, 0.1611]
    assert measure.wilson95(10, 20) == [0.2993, 0.7007]
    assert measure.wilson95(20, 20) == [0.8389, 1.0]
    assert measure.wilson95(0, 0) is None


def test_results_file_has_no_long_text(tmp_path):
    long_answer = " ".join(f"w{i}" for i in range(300)) + " [1]."
    recs = [rec(long_answer), rec("Fine [1].", variant="grounded_v1")]
    recs[1]["split"] = "heldout"
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    out = tmp_path / "r.json"
    res = measure.measure([p], out)
    data = json.loads(out.read_text())
    assert data["schema"] == "vera-trace-eval-results-v1" and data["checks_version"] == "trace-checks-v1"
    assert set(data["variants"]["grounded_v1"]["by_split"]) == {"dev", "heldout"}
    for t in data["traces"]:
        assert len(t["answer_excerpt"].split()) <= 25
        assert all(set(s) == {"n", "url", "title"} for s in t["sources"])
    assert SNIP not in out.read_text() and long_answer not in out.read_text()
    assert res["traces"][0]["answer_excerpt"].split()[0] == "w0"


def test_self_test_clean():
    assert checks.self_test() == []
