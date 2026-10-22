"""Streamlit AppTest for the Trace Eval page, from synthetic JSON files only."""
import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.public_terms import find_terms, public_terms
from vera.trace_eval.ui import (CAPTURE_ERROR_TEXT, GUIDANCE_UNAVAILABLE_TEXT, LABELLER_TEXT, MISSING_RESULTS_TEXT,
                                RENDER_ERROR_TEXT)

ROOT = Path(__file__).resolve().parent.parent

# arXiv is the public source name and the only link host (ui.py ARXIV_PREFIX); not a back-end disclosure on this page.
# Named exemption (K3.P2 plan r2 F1): nothing else is removed from the shared term list.
TRACE_EVAL_ALLOWED = frozenset({"arxiv"})
TERMS = tuple(t for t in public_terms() if t not in TRACE_EVAL_ALLOWED)
RAW_ID = re.compile(r"\bA\d_|grounded_v1|checks_version|questions_sha256")


def _chk(name, applies=True, passed=True):
    return {"name": name, "applies": applies, "passed": passed if applies else None, "reason": "r"}


def _results():
    split = {"n": 2, "errors": 0, "all_pass": 1, "all_pass_rate": 0.5, "wilson95": [0.1, 0.9]}
    pc = {"applies": 2, "passed": 1, "rate": 0.5, "wilson95": [0.1, 0.9]}
    var = {"by_split": {"dev": split}, "per_check": {"A1_citation_present": pc}}
    trace = lambda v: {"id": "q01", "split": "dev", "category": "in_scope_synthesis", "variant": v,
                       "all_pass": True, "error": None, "checks": [_chk("A1_citation_present")],
                       "answer_excerpt": "some words", "sources": [
                           {"n": 1, "url": "https://arxiv.org/abs/1234.5678", "title": "Paper"}]}
    return {"schema": "vera-trace-eval-results-v1", "checks_version": "v", "questions_sha256": "a" * 64,
            "variants": {"baseline": var, "grounded_v1": var}, "traces": [trace("baseline"), trace("grounded_v1")]}


def _validation():
    return {"schema": "vera-trace-check-validation-v1", "positive_class": "FAIL", "labeller": "x",
            "per_check": {"A1_citation_present": {"n": 4, "tp": 2, "tn": 1, "fp": 0, "fn": 1,
                                                  "tpr": 0.67, "tnr": 1.0, "note": ""}},
            "support": {"n": 3, "supported": 1, "partially": 1, "unsupported": 1,
                        "heuristic_tpr": None, "heuristic_tnr": 0.5, "note": ""}}


def _app(results_path, validation_path, unlocked=True):
    import streamlit as st
    from vera import trace_viz
    from vera.trace_eval.ui import render_trace_eval
    if unlocked and "trace_viz_status" not in st.session_state:  # TRACEVIZ-1022: simulate an accepted key (API tested elsewhere)
        st.session_state["trace_viz_status"] = 200
        st.session_state["trace_viz_detail"] = trace_viz.private_view(results_path)
    render_trace_eval(results_path, validation_path)


@pytest.fixture
def files(tmp_path):
    r, v = tmp_path / "r.json", tmp_path / "v.json"
    r.write_text(json.dumps(_results()))
    v.write_text(json.dumps(_validation()))
    return str(r), str(v), tmp_path


def _run(r, v):
    return AppTest.from_function(_app, kwargs={"results_path": r, "validation_path": v}, default_timeout=30).run()


def test_renders_with_headline_and_validation(files):
    r, v, _ = files
    at = _run(r, v)
    assert not at.exception
    assert len(at.dataframe) >= 3
    assert not any("not yet available" in i.value for i in at.info)
    assert len(at.selectbox) == 2


def test_validation_missing_shows_info(files):
    r, _, tmp = files
    at = _run(r, str(tmp / "absent.json"))
    assert not at.exception
    assert any("Check validation not yet available" in i.value for i in at.info)


def test_missing_results_warns(files):
    _, v, tmp = files
    at = _run(str(tmp / "absent.json"), v)
    assert not at.exception
    assert [w.value for w in at.warning] == [MISSING_RESULTS_TEXT]
    for bad in ("venv", "scripts/", ".py", "eval_results"):
        assert bad not in at.warning[0].value


def test_sources_have_no_forbidden_literals():
    for rel in ("vera/trace_eval/ui.py", "pages/trace_eval.py"):
        src = (ROOT / rel).read_text()
        for bad in ("onrender", "localhost", "http://"):
            assert bad not in src, (rel, bad)


# ---- K3.P2: public text remediation (sentinels first, then the shared term list) ----
def _page_text(at) -> str:
    """Every element of the tree (captions, errors, markdown, widget labels/options, dataframes); bytes fields skipped."""
    import pandas as pd
    parts = []

    def walk(n):
        proto = getattr(n, "proto", None)
        if proto is not None and hasattr(proto, "ListFields"):
            parts.extend(str(v) for _, v in proto.ListFields() if not isinstance(v, bytes))
        if type(n).__name__ == "Dataframe":  # other nodes' .value can need widget state; their protos are walked above
            parts.append(pd.DataFrame(n.value).to_string())
        kids = getattr(n, "children", None)
        for ch in (kids.values() if isinstance(kids, dict) else []):
            walk(ch)
    walk(at._tree)
    return "\n".join(parts)


def _sweep(r, v, check=None):
    """Run the page over every (variant, question id) selectbox state; yield (state, text) for each."""
    at = _run(r, v)
    assert not at.exception
    yield at, _page_text(at)
    for variant in at.selectbox[0].options:
        at.selectbox[0].select(variant)
        at.run()
        for qid in at.selectbox[1].options:
            at.selectbox[1].select(qid)
            at.run()
            assert not at.exception
            yield at, _page_text(at)


def _write(tmp, res, val=None):
    r, v = tmp / "r.json", tmp / "v.json"
    r.write_text(json.dumps(res))
    v.write_text(json.dumps(val if val is not None else _validation()))
    return str(r), str(v)


def _two_question_results(error="SearchError: SENTINEL_ERR_7f3 across providers: SENTINEL_PROV_a1: HTTP 429 '' "
                                 "https://SENTINEL-HOST.invalid"):
    res = _results()
    extra = json.loads(json.dumps(res["traces"][1]))
    extra.update({"id": "q02", "error": error, "all_pass": False})
    res["traces"].append(extra)
    return res


def test_capture_error_is_fixed_text(tmp_path):
    r, v = _write(tmp_path, _two_question_results())
    seen = 0
    for at, text in _sweep(r, v):
        if at.selectbox[1].value == "q02" and at.selectbox[0].value:
            assert CAPTURE_ERROR_TEXT in text
            seen += 1
        for bad in ("SENTINEL_ERR_7f3", "SENTINEL_PROV_a1", "SENTINEL-HOST", "SearchError", "HTTP 429", "providers"):
            assert bad not in text, bad
    assert seen


def test_context_has_no_model_identity(tmp_path):
    r, v = _write(tmp_path, _results())
    text = _page_text(_run(r, v))
    assert "third-party hosted language model" in text
    for bad in ("gpt", "nano", "/ask", "keyless"):
        assert bad not in text.lower(), bad
    assert "checks_version" not in text and "questions_sha256" not in text and "Snapshot captured" in text


def test_labeller_not_rendered(tmp_path):
    val = _validation()
    val["labeller"] = "SENTINEL_LABELLER_MODEL_x9 opus"
    r, v = _write(tmp_path, _results(), val)
    text = _page_text(_run(r, v))
    assert "SENTINEL_LABELLER_MODEL_x9" not in text and "opus" not in text.lower() and LABELLER_TEXT in text


def test_public_terms_absent_from_page_chrome(tmp_path):
    r, v = _write(tmp_path, _two_question_results())
    for at, text in _sweep(r, v):
        assert not find_terms(text, TERMS), find_terms(text, TERMS)
        assert not RAW_ID.search(text), RAW_ID.search(text)


def test_exemption_is_only_arxiv():
    assert TRACE_EVAL_ALLOWED == {"arxiv"} and "arxiv" in public_terms() and "arxiv" not in TERMS
    assert "qwen" in TERMS and "duckduckgo" in TERMS


def test_real_committed_results_render_without_known_leaks():
    r, v = ROOT / "eval_results" / "trace_eval_v1.json", ROOT / "eval_results" / "trace_eval_v1_validation.json"
    if not (r.is_file() and v.is_file()):
        pytest.skip("committed evidence files absent")
    n = 0
    for at, text in _sweep(str(r), str(v)):
        n += 1
        for bad in ("gpt-4.1-nano", "SearchError", "HTTP 429", "providers:", "opus subagent", "trace_measure.py",
                    "config/trace_eval_guidance.json", "checks_version", "questions_sha256", "grounded_v1",
                    "arXiv rate limits"):
            assert bad not in text, bad
        assert not re.search(r"\bA\d_", text), re.search(r"\bA\d_", text)
    assert n > 10
    assert (ROOT / "eval_results" / "trace_eval_v1.json").read_text().count("grounded_v1") > 0  # evidence unchanged


def test_missing_results_and_guidance_texts_fixed(files, monkeypatch):
    r, v, tmp = files
    from vera.trace_eval import guidance

    def boom(*a, **k):
        raise ValueError("config/trace_eval_guidance.json bad")
    monkeypatch.setattr(guidance, "load_config", boom)
    at = _run(r, v)
    assert not at.exception
    assert [i.value for i in at.info][:1] == [GUIDANCE_UNAVAILABLE_TEXT]
    text = _page_text(at)
    assert "trace_eval_guidance" not in text and "config/" not in text and "ValueError" not in text


def test_malformed_results_show_fixed_text(tmp_path):
    res = _results()
    del res["traces"][0]["category"]
    del res["traces"][1]["checks"][0]["reason"]
    r, v = _write(tmp_path, res)
    at = _run(r, v)
    assert not at.exception
    # TRACEVIZ-1022: the browser moved to the key-gated tier and reads with .get, so missing fields no longer raise
    assert [e.value for e in at.error] == []  # no error now; the first malformed case is tolerated by the .get readers
    text = _page_text(at)
    for bad in ("KeyError", "Traceback", ".py"):
        assert bad not in text, bad
    # a non-object results file and a failing guidance computation take the same fixed route
    (tmp_path / "r.json").write_text("[1, 2]")
    at = _run(str(tmp_path / "r.json"), v)
    assert not at.exception and [e.value for e in at.error] == [RENDER_ERROR_TEXT]


def test_unknown_variant_and_check_names_not_rendered(tmp_path):
    res = _results()
    for t in res["traces"]:
        t["checks"][0]["name"] = "SENTINEL_CHECK_c4"
        if t["variant"] == "grounded_v1":
            t["variant"] = "SENTINEL_VARIANT_v9"
    res["variants"] = {"baseline": {**res["variants"]["baseline"], "per_check": {"SENTINEL_CHECK_c4": res["variants"]["baseline"]["per_check"]["A1_citation_present"]}},
                       "SENTINEL_VARIANT_v9": {**res["variants"]["baseline"], "per_check": {"SENTINEL_CHECK_c4": res["variants"]["baseline"]["per_check"]["A1_citation_present"]}}}
    val = _validation()
    val["per_check"] = {"SENTINEL_CHECK_c4": val["per_check"]["A1_citation_present"]}
    r, v = _write(tmp_path, res, val)
    for at, text in _sweep(r, v):
        assert "SENTINEL_VARIANT_v9" not in text and "SENTINEL_CHECK_c4" not in text
    assert "Variant 1" in text and "Check 1" in text
    assert len(at.dataframe) >= 5  # guidance tables still render with display names


def test_known_names_are_relabelled_everywhere(tmp_path):
    val = _validation()
    val["support"].update({"tp": 2, "tn": 3, "fp": 1, "fn": 1})  # makes guidance emit its "A6_support (heuristic)" row
    r, v = _write(tmp_path, _results(), val)
    text = _page_text(_run(r, v))
    for shown in ("Baseline", "Grounded", "Citation present", "Support (heuristic)"):
        assert shown in text, shown
    assert "Support heuristic" in text and "(A6)" not in text


def test_public_labels_do_not_modify_inputs():
    from vera.trace_eval.ui import _public_labels
    res, val = _results(), _validation()
    before = json.dumps([res, val], sort_keys=True)
    _public_labels(res, val, {"baseline_variant": "baseline"})
    assert json.dumps([res, val], sort_keys=True) == before


def test_error_value_only_used_as_truth_test():
    src = (ROOT / "vera" / "trace_eval" / "ui.py").read_text()
    uses = [m.group(0) for m in re.finditer(r"""t\[['"]error['"]\]|t\.get\(['"]error['"]\)""", src)]
    # TRACEVIZ-1022: the browser moved to the private tier, which only receives a boolean capture_error (never the text)
    assert uses == [] and 'if t.get("capture_error"):' in src
    assert "labeller')" not in src and "val['labeller']" not in src
