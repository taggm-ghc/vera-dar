"""TRACEEVAL-1022: offline end to end for the Trace Eval pipeline (no model, network or DB).

Synthetic /ask bodies -> capture.run_capture (variant ask_api) -> the real checks A1-A9 -> measure.measure (results file)
-> the real page (AppTest). Plus exposure sentinels: no key, host or env value reaches traces, results, stdout or page.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.public_terms import find_terms, public_terms
from vera.trace_eval import capture, checks, guidance, measure
from vera.trace_eval.ui import HOW_TO_PRODUCE_TEXT, MISSING_RESULTS_TEXT, NO_BASELINE_TEXT

ROOT = Path(__file__).resolve().parent.parent
KEY, HOST = "SENTINEL_KEY_9q7Zx", "https://SENTINEL-HOST-4k2.invalid"
SENTINELS = (KEY, "SENTINEL-HOST-4k2", "SENTINEL_ENV_VALUE_3m")
TERMS = tuple(t for t in public_terms() if t != "arxiv")
SNIP_WORDS = 60

FROZEN = {"status": "FROZEN", "items": [
    {"id": "q01", "text": "How do AI coding assistants affect productivity?", "category": "in_scope_synthesis", "split": "dev"},
    {"id": "q02", "text": "Why did all studies prove assistants double output?", "category": "false_premise", "split": "dev"},
    {"id": "q03", "text": "What is the best pizza topping?", "category": "out_of_scope", "split": "dev"},
    {"id": "q04", "text": "Is it better?", "category": "ambiguous_or_underspecified", "split": "dev"},
    {"id": "q05", "text": "Summarize evidence on code review with assistants.", "category": "multi_part", "split": "dev"},
]}

LONG = " ".join(f"word{i}" for i in range(SNIP_WORDS))
BODIES = {
    "q01": {"answer": f"A randomized trial found developers were 19% slower on familiar code [1]. {LONG}",
            "sources": [{"n": 1, "url": "https://arxiv.org/abs/2507.09089", "title": "Trial", "published": "2025-07-12",
                         "provider": "SENTINEL_ENV_VALUE_3m", "span": "chars=1-9"}],
            "cost_usd": 0.001, "tokens_used": 10, "model": "m", "result_kind": "answer", "outcome": "answer"},
    "q02": {"answer": "That premise is false: the evidence does not show assistants double output [1].",
            "sources": [{"n": 1, "url": "https://doi.org/10.1/x", "title": "Survey", "published": "2024-01-01"}],
            "cost_usd": 0.001, "tokens_used": 10, "model": "m"},
    "q03": {"answer": "That is outside what VERA-DAR covers, so I cannot answer it.", "sources": [], "cost_usd": 0.0},
    "q04": {"answer": "Better than what? Do you mean AI coding assistants versus no assistant?", "sources": [],
            "cost_usd": 0.0},
}


def _fake_ask(question: str) -> dict:
    qid = next(q["id"] for q in FROZEN["items"] if q["text"] == question)
    if qid == "q05":  # a transport failure whose message carries a host: only the type may be recorded
        raise ConnectionError(f"failed to reach {HOST}/ask with key {KEY}")
    return json.loads(json.dumps(BODIES[qid]))


def _pipeline(tmp_path: Path, ask=_fake_ask) -> tuple[Path, Path, dict]:
    traces, results = tmp_path / "traces.jsonl", tmp_path / "results.json"
    summary = capture.run_capture(FROZEN, "ask_api", traces, ask=ask, questions_sha256="s" * 64)
    measure.measure([traces], results)
    return traces, results, summary


def _app(results_path, validation_path):
    from vera.trace_eval.ui import render_trace_eval
    render_trace_eval(results_path, validation_path)


def _run(r, v):
    return AppTest.from_function(_app, kwargs={"results_path": str(r), "validation_path": str(v)},
                                 default_timeout=30).run()


def _text(at) -> str:
    from tests.test_trace_eval_page import _page_text
    return _page_text(at)


def test_pipeline_produces_results_with_all_checks_and_dimensions(tmp_path):
    traces, results, summary = _pipeline(tmp_path)
    assert summary["n"] == 5 and summary["errors"] == 1 and summary["stopped"] is None
    res = json.loads(results.read_text())
    assert res["checks_version"] == checks.CHECKS_VERSION and set(res["variants"]) == {"ask_api"}
    v = res["variants"]["ask_api"]
    assert set(v["per_dimension"]) == set(guidance.load_config()["dimensions"])
    assert v["per_category"]["out_of_scope"] == {"n": 1, "errors": 0, "all_pass": 1, "rate": 1.0,
                                                "wilson95": measure.wilson95(1, 1)}
    assert v["by_split"]["dev"]["errors"] == 1  # q05 capture error counts as a failure, not a dropped row
    assert res["run"] == {"case_sets": ["frozen"], "checks_validated": True, "capture_errors": 1,
                          "source_text_unavailable": 4, "traces": 5}
    by_id = {t["id"]: {c["name"]: c for c in t["checks"]} for t in res["traces"]}
    # Real check outcomes, not fixed ones: [n] citations count for ask_api; numbers/support are n/a without source text.
    assert by_id["q01"]["A1_citation_present"]["passed"] is True
    assert by_id["q01"]["A3_numbers_grounded"]["applies"] is False
    assert by_id["q01"]["A6_support"]["applies"] is False
    assert by_id["q02"]["A7_premise_corrected"]["passed"] is True
    assert by_id["q03"]["A8_scope_decline"]["passed"] is True
    assert by_id["q04"]["A9_clarify_or_assume"]["passed"] is True
    # Short excerpts only, no passage text, no provenance fields beyond n/url/title.
    for t in res["traces"]:
        assert len(t["answer_excerpt"].split()) <= measure.EXCERPT_WORDS
        assert all(set(s) == {"n", "url", "title"} for s in t["sources"])
    assert LONG not in results.read_text()


def test_v1_records_score_as_before_and_ask_api_differs_only_where_documented():
    src = [{"n": 1, "url": "https://arxiv.org/abs/2401.1", "title": "T", "published": "2024-01-01",
            "snippet": "developers were 30% faster"}]
    base = {"id": "x", "category": "in_scope_synthesis", "variant": "grounded_v1", "sources": src, "error": None,
            "answer": "Developers were 19% faster [1]."}
    v1 = {r.name: r for r in checks.run_checks(base)}
    assert v1["A3_numbers_grounded"].passed is False  # number not in the snippet: unchanged v1 behaviour
    ask = {**base, "variant": "ask_api", "source_text_available": False}
    v11 = {r.name: r for r in checks.run_checks(ask)}
    assert v11["A3_numbers_grounded"].applies is False and v11["A6_support"].applies is False
    for n in ("A1_citation_present", "A2_citations_resolve", "A4_phantom_evidence", "A5_window"):
        assert v11[n].to_dict() == v1[n].to_dict(), n
    assert checks.self_test() == []


def test_page_renders_real_pipeline_output(tmp_path):
    _, results, _ = _pipeline(tmp_path)
    at = _run(results, tmp_path / "no-validation.json")
    assert not at.exception
    text = _text(at)
    assert "Per dimension" in text and "Per question category" in text and NO_BASELINE_TEXT in text
    for dim in guidance.load_config()["dimensions"]:
        assert dim in text
    assert "Live answers" in text and "without source text" in text and "could not be captured" in text
    for s in SENTINELS + (HOST,):
        assert s not in text, s
    assert not find_terms(text, TERMS), find_terms(text, TERMS)
    for raw in ("A1_", "A3_", "ask_api", "checks_version", "questions_sha256", "/ask"):
        assert raw not in text, raw


def test_page_paired_comparison_when_baseline_present(tmp_path):
    traces, results, _ = _pipeline(tmp_path)
    recs = [json.loads(line) for line in traces.read_text().splitlines()]
    base = [{**r, "variant": "baseline", "sources": [], "source_text_available": None} for r in recs]
    both = tmp_path / "both.jsonl"
    both.write_text("\n".join(json.dumps(r) for r in recs + base))
    measure.measure([both], results)
    at = _run(results, tmp_path / "none.json")
    assert not at.exception
    text = _text(at)
    assert "Computed determinations" in text and NO_BASELINE_TEXT not in text


def test_no_results_state_explains_how(tmp_path):
    at = _run(tmp_path / "absent.json", tmp_path / "absent-v.json")
    assert not at.exception
    assert [w.value for w in at.warning] == [MISSING_RESULTS_TEXT]
    assert HOW_TO_PRODUCE_TEXT in [i.value for i in at.info]  # TRACEVIZ-1022: the public tier adds its own empty-state infos
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"schema": measure.SCHEMA, "variants": {}, "traces": []}))
    assert [w.value for w in _run(empty, tmp_path / "x").warning] == [MISSING_RESULTS_TEXT]
    for bad in (".py", "scripts/", "venv", "eval_results", "http", "localhost"):
        assert bad not in HOW_TO_PRODUCE_TEXT


def test_env_var_points_page_at_results(tmp_path, monkeypatch):
    _, results, _ = _pipeline(tmp_path)
    from vera.trace_eval import ui
    monkeypatch.setenv(ui.RESULTS_ENV, str(results))
    assert ui._resolve_paths(None, None)[0] == str(results)
    monkeypatch.delenv(ui.RESULTS_ENV)
    assert ui._resolve_paths(None, None)[0] == ui.RESULTS_PATH


def test_traces_hold_no_sentinel_and_errors_are_type_only(tmp_path):
    traces, results, _ = _pipeline(tmp_path)
    raw = traces.read_text() + results.read_text()
    for s in (KEY, "SENTINEL-HOST-4k2"):
        assert s not in raw, s
    q05 = [json.loads(line) for line in traces.read_text().splitlines()][-1]
    assert q05["error"] == "ConnectionError"


def test_http_ask_status_handling(monkeypatch, tmp_path):
    import vera.public_mode as pm
    seen = {}

    class R:
        def __init__(self, code, body=None):
            self.status_code, self._b = code, body

        def json(self):
            return self._b

    def fake_post(base_url, question, mode, api_key, timeout):
        seen.update(base=base_url, key=api_key, mode=mode)
        return R(int(question)) if question.isdigit() else R(200, BODIES["q03"])

    monkeypatch.setattr(pm, "post_ask", fake_post)
    ask = capture.http_ask(HOST, KEY)
    assert ask("hello")["answer"].startswith("That is outside") and seen == {"base": HOST, "key": KEY, "mode": "normal"}
    with pytest.raises(capture.AskHTTPError) as e:
        ask("502")
    assert e.value.status == 502 and HOST not in str(e.value) and KEY not in str(e.value)
    doc = {"status": "FROZEN", "items": [{"id": "a", "text": "429", "category": "out_of_scope", "split": "dev"},
                                         {"id": "b", "text": "401", "category": "out_of_scope", "split": "dev"}]}
    with pytest.raises(capture.AskHTTPError):  # 401: the route cannot answer anything; abort, do not record 30 errors
        capture.run_capture(doc, "ask_api", tmp_path / "t.jsonl", ask=ask)
    first = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])
    assert first["error"] == "AskHTTPError: HTTP 429"


def test_inprocess_ask_uses_real_route_shape(monkeypatch):
    from fastapi import FastAPI, Header

    app = FastAPI()

    @app.post("/ask")
    def ask_route(body: dict, x_api_key: str = Header(default="", alias="X-API-Key")):
        return {"answer": f"ok {body['mode']} {x_api_key == KEY}", "sources": [], "cost_usd": 0.0}

    monkeypatch.setitem(sys.modules, "main", types.SimpleNamespace(app=app))
    assert capture.inprocess_ask(KEY)("q")["answer"] == "ok normal True"


def test_accept_case_set_adapted_by_config_rules(tmp_path):
    p = tmp_path / "questions.json"
    p.write_text(json.dumps({"_doc": "x", "questions": [
        {"id": "q07", "label": "off_topic", "topic": "history", "text": "t7", "expected": "layer1_denied"},
        {"id": "q21", "label": "off_topic", "topic": "sports; false premise", "text": "t21", "expected": "no_answer"},
        {"id": "q23", "label": "in_scope", "topic": "vague", "text": "t23", "expected": "no_answer"},
        {"id": "q18", "label": "in_scope", "topic": "AI coding assistants", "text": "t18", "expected": "answer"}]}))
    doc, sha = capture.load_case_set(p, "accept", guidance.load_config())
    assert [q["category"] for q in doc["items"]] == ["out_of_scope", "false_premise", "ambiguous_or_underspecified",
                                                     "in_scope_synthesis"]
    assert doc["checks_validated"] is False and len(sha) == 64
    with pytest.raises(ValueError):
        capture.load_case_set(p, "accept", {"case_sets": {}})
    with pytest.raises(ValueError):  # an un-frozen file is never accepted as the frozen set
        capture.load_case_set(p, "frozen")


def _cli():
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    return importlib.import_module("trace_capture")


def test_cli_dry_run_makes_no_call_and_writes_nothing(tmp_path, monkeypatch, capsys):
    tc = _cli()
    import vera.env_files as ef

    def no_env(*a, **k):
        raise AssertionError("dry run must not load env")
    monkeypatch.setattr(ef, "load_vera_env", no_env)
    monkeypatch.setattr(capture, "http_ask", no_env)
    monkeypatch.setattr(capture, "inprocess_ask", no_env)
    monkeypatch.setenv("VERA_API_KEY", KEY)
    q = tmp_path / "q.json"
    q.write_text(json.dumps(FROZEN))
    out, res = tmp_path / "t.jsonl", tmp_path / "r.json"
    rc = tc.main(["--variant", "ask_api", "--case-set", "frozen", "--questions", str(q), "--out", str(out),
                  "--results-out", str(res), "--dry-run"])
    printed = capsys.readouterr()
    assert rc == 0 and not out.exists() and not res.exists()
    assert "questions=5" in printed.out and "no call made" in printed.out
    for s in SENTINELS:
        assert s not in printed.out + printed.err


def test_cli_measure_only(tmp_path, capsys):
    tc = _cli()
    traces, _, _ = _pipeline(tmp_path)
    res = tmp_path / "cli-results.json"
    assert tc.main(["--measure-only", "--out", str(traces), "--results-out", str(res)]) == 0
    assert json.loads(res.read_text())["variants"]["ask_api"]["per_dimension"]
    printed = capsys.readouterr().out
    assert "dimension Citation" in printed and LONG.split()[5] not in printed


def test_cli_live_path_with_fake_ask(tmp_path, monkeypatch, capsys):
    tc = _cli()
    import vera.env_files as ef
    monkeypatch.setattr(ef, "load_vera_env", lambda *a, **k: None)
    monkeypatch.setenv("VERA_API_KEY", KEY)
    monkeypatch.setenv("VERA_API_BASE_URL", HOST)
    got = {}

    def fake_http(base, key, timeout=None):
        got.update(base=base, key=key)
        return _fake_ask
    monkeypatch.setattr(capture, "http_ask", fake_http)
    q = tmp_path / "q.json"
    q.write_text(json.dumps(FROZEN))
    out, res = tmp_path / "t.jsonl", tmp_path / "r.json"
    rc = tc.main(["--variant", "ask_api", "--target", "http", "--questions", str(q), "--out", str(out),
                  "--results-out", str(res)])
    printed = capsys.readouterr()
    assert rc == 0 and got == {"base": HOST, "key": KEY} and res.is_file()
    for s in SENTINELS:
        assert s not in printed.out + printed.err, s
    assert tc.main(["--variant", "ask_api", "--questions", str(q), "--out", str(out)]) == 2  # refuses to overwrite


def test_ui_and_capture_sources_have_no_host_literals():
    for rel in ("vera/trace_eval/ui.py", "vera/trace_eval/capture.py", "scripts/trace_capture.py"):
        src = (ROOT / rel).read_text()
        for bad in ("onrender", "localhost", "127.0.0.1", "http://"):
            assert bad not in src, (rel, bad)


@pytest.fixture(autouse=True)
def _no_shipped_fallbacks(tmp_path, monkeypatch):
    """These tests cover the empty and sentinel-data states; the shipped historical/offline aggregates are covered in test_trace_auto.py."""
    from vera import trace_viz as _tv
    monkeypatch.setattr(_tv, "HISTORICAL_PATH", tmp_path / "no-historical.json")
    monkeypatch.setattr(_tv, "OFFLINE_PATH", tmp_path / "no-offline.json")
