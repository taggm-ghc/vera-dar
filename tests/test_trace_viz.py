"""TRACEVIZ-1022: two-tier Trace Eval view. Public aggregates always render; detail needs X-Viz-Key (server side)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from vera import trace_viz
from vera.trace_eval import ui as tui

ROOT = Path(__file__).resolve().parent.parent
KEY = "VIZ-KEY-correct-0123456789abcdef"
SENT_Q, SENT_EXC, SENT_HOST = "SENTINEL_QUESTION_text_q1", "SENTINEL_EXCERPT_third_party_words", "sentinel-host.invalid"
SENT_SECRET, SENT_TITLE = "SENTINEL_SECRET_apikey_zz91", "SENTINEL_TITLE_of_paper"


def _results():
    pc = {"applies": 2, "passed": 1, "rate": 0.5, "wilson95": [0.1, 0.9]}
    split = {"n": 2, "errors": 1, "all_pass": 1, "all_pass_rate": 0.5, "wilson95": [0.1, 0.9]}
    tr = {"id": "q01", "split": "dev", "category": "in_scope_synthesis", "variant": "baseline", "all_pass": True,
          "error": f"Err {SENT_HOST} {SENT_SECRET}", "question": SENT_Q,
          "checks": [{"name": "A1_citation_present", "applies": True, "passed": True, "reason": "r1"}],
          "answer_excerpt": SENT_EXC, "sources": [{"n": 1, "title": SENT_TITLE, "url": f"https://{SENT_HOST}/x"}]}
    return {"schema": "r", "generated_at": "2026-10-04T00:00:00Z", "variants": {"baseline": {
        "by_split": {"dev": split}, "per_check": {"A1_citation_present": pc}}}, "traces": [tr]}


def _val():
    return {"per_check": {"A1_citation_present": {"n": 4, "tp": 2, "tn": 1, "fp": 0, "fn": 1, "tpr": 0.67, "tnr": 1.0}},
            "labeller": SENT_SECRET}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    r, v = tmp_path / "r.json", tmp_path / "v.json"
    r.write_text(json.dumps(_results()))
    v.write_text(json.dumps(_val()))
    monkeypatch.setenv("VERA_TRACE_EVAL_RESULTS", str(r))
    monkeypatch.setattr(trace_viz, "VALIDATION_PATH", v)
    monkeypatch.setattr(trace_viz, "PUBLIC_PATH", tmp_path / "none.json")
    monkeypatch.setenv("VERA_API_KEY", "api-key-different-0123456789")
    trace_viz.reset_for_tests()
    return r, v, tmp_path


@pytest.fixture
def client(paths, monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    import main
    return TestClient(main.app)


def _no_leak(text: str):
    for bad in (SENT_Q, SENT_EXC, SENT_HOST, SENT_SECRET, SENT_TITLE, KEY, "A1_citation_present", "baseline"):
        assert bad not in text, bad


def test_public_endpoint_has_aggregates_and_no_sentinels(client):
    r = client.get("/trace/public")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["run"]["n_questions"] == 1 and body["run"]["n_errors"] == 1
    assert body["run"]["variants"]["Baseline"]["per_check"]["Citation present"]["rate"] == 0.5
    assert body["check_validation"]["checks"]["Citation present"]["tpr"] == 0.67
    assert len(body["stages"]) == 9
    _no_leak(r.text)


def test_public_endpoint_no_data_is_useful_not_blank(client, tmp_path, monkeypatch):
    monkeypatch.setenv("VERA_TRACE_EVAL_RESULTS", str(tmp_path / "missing.json"))
    monkeypatch.setattr(trace_viz, "VALIDATION_PATH", tmp_path / "missing2.json")
    body = client.get("/trace/public").json()
    assert body["status"] == "no_run_results" and body["run"] is None and len(body["stages"]) == 9


def test_private_unset_key_is_disabled_even_with_a_header(client, monkeypatch):
    monkeypatch.delenv("VERA_TRACE_VIZ_KEY", raising=False)
    for hdr in ({}, {"X-Viz-Key": KEY}, {"X-Viz-Key": ""}):
        r = client.get("/trace/private", headers=hdr)
        assert r.status_code == 403 and r.json() == {"detail": "Not available"}


def test_private_weak_or_api_key_equal_is_disabled(client, monkeypatch):
    monkeypatch.setenv("VERA_TRACE_VIZ_KEY", "short")
    assert client.get("/trace/private", headers={"X-Viz-Key": "short"}).status_code == 403
    monkeypatch.setenv("VERA_TRACE_VIZ_KEY", "api-key-different-0123456789")
    assert client.get("/trace/private", headers={"X-Viz-Key": "api-key-different-0123456789"}).status_code == 403


def test_private_missing_and_wrong_key_identical_403(client, monkeypatch):
    monkeypatch.setenv("VERA_TRACE_VIZ_KEY", KEY)
    a = client.get("/trace/private")
    b = client.get("/trace/private", headers={"X-Viz-Key": "wrong-key-0123456789abcdef"})
    assert a.status_code == b.status_code == 403 and a.text == b.text
    _no_leak(a.text + b.text)


def test_private_correct_key_returns_detail(client, monkeypatch):
    monkeypatch.setenv("VERA_TRACE_VIZ_KEY", KEY)
    r = client.get("/trace/private", headers={"X-Viz-Key": KEY})
    assert r.status_code == 200
    t = r.json()["traces"][0]
    assert t["question"] == SENT_Q and t["answer_excerpt"] == SENT_EXC and t["capture_error"] is True
    assert t["variant"] == "Baseline" and t["checks"][0]["name"] == "Citation present"
    assert SENT_SECRET not in r.text and "Err " not in r.text  # the raw capture-error text is never sent


def test_private_correct_key_no_data_is_empty_not_error(client, monkeypatch, tmp_path):
    monkeypatch.setenv("VERA_TRACE_VIZ_KEY", KEY)
    monkeypatch.setenv("VERA_TRACE_EVAL_RESULTS", str(tmp_path / "missing.json"))
    r = client.get("/trace/private", headers={"X-Viz-Key": KEY})
    assert r.status_code == 200 and r.json() == {"available": False, "traces": []}


def test_correct_key_still_200_after_many_failures_and_overlimit_wrong_is_identical_403(client, monkeypatch):
    monkeypatch.setenv("VERA_TRACE_VIZ_KEY", KEY)
    first = client.get("/trace/private", headers={"X-Viz-Key": "bad-0-0123456789abcdef"})
    codes = [client.get("/trace/private", headers={"X-Viz-Key": f"bad-{i}-0123456789abcdef"}) for i in range(12)]
    assert {c.status_code for c in codes} == {403} and {c.text for c in codes} == {first.text}
    assert client.get("/trace/private", headers={"X-Viz-Key": KEY}).status_code == 200


def test_public_file_projection_is_closed(client, paths, monkeypatch):
    _r, _v, tmp = paths
    pub = {"schema": trace_viz.SCHEMA, "generated_at": "2026-10-01", "n_questions": "[x](http://evil)", "n_traces": 3,
           "n_errors": 0, "SENTINEL_EXTRA_FIELD": SENT_Q, "question": SENT_Q,
           "variants": {"baseline": {"per_check": {"A1_citation_present": {"passed": 1, "applies": 2, "rate": 0.5, "url": SENT_HOST}},
                                     "per_category": {"Bad `cat` ![i](x)": {"n": 1, "all_pass": 1}}, "extra": SENT_EXC},
                        "SENTINEL_RAW_VARIANT": {}}}
    f = tmp / "pub.json"
    f.write_text(json.dumps(pub))
    monkeypatch.setattr(trace_viz, "PUBLIC_PATH", f)
    r = client.get("/trace/public")
    for bad in ("SENTINEL_EXTRA_FIELD", SENT_Q, SENT_HOST, SENT_EXC, "SENTINEL_RAW_VARIANT", "baseline", "A1_citation_present", "evil", "`"):
        assert bad not in r.text, bad
    run = r.json()["run"]
    assert run["n_questions"] is None and run["n_traces"] == 3 and "Baseline" in run["variants"]
    at = _at(_r, _v, f)
    page = _page(at)
    for bad in ("SENTINEL_EXTRA_FIELD", SENT_Q, SENT_HOST, SENT_EXC, "SENTINEL_RAW_VARIANT", "evil"):
        assert bad not in page, bad


def test_env_vars_are_in_examples_note_and_not_hardcoded():
    src = (ROOT / "vera" / "trace_viz.py").read_text()
    assert 'VIZ_KEY_ENV = "VERA_TRACE_VIZ_KEY"' in src and "compare_digest" in src


# --- page ----------------------------------------------------------------------------------------------------------

def _app(results_path, validation_path, public_path):
    from vera.trace_eval.ui import render_trace_eval
    render_trace_eval(results_path, validation_path, public_path, base_url="http://upstream.invalid:9")


def _page(at) -> str:
    return "\n".join(repr(el.proto) for el in list(at.main) + list(at.sidebar)) + repr(at.session_state.filtered_state)


def _at(r, v, p):
    return AppTest.from_function(_app, kwargs={"results_path": str(r), "validation_path": str(v), "public_path": str(p)},
                                 default_timeout=30).run()


def test_page_public_with_data_shows_aggregates_no_key_no_leak(paths):
    r, v, tmp = paths
    at = _at(r, v, tmp / "none.json")
    assert not at.exception
    text = _page(at)
    assert "Evaluation flow" in text and "Latest run" in text and "Check validation" in text
    assert len(at.dataframe) >= 2
    for bad in (SENT_Q, SENT_EXC, SENT_HOST, SENT_SECRET, SENT_TITLE):
        assert bad not in text, bad
    assert not at.selectbox  # per-trace browser is locked


def test_page_public_no_data_is_not_blank(tmp_path):
    at = _at(tmp_path / "x.json", tmp_path / "y.json", tmp_path / "z.json")
    assert not at.exception
    text = _page(at)
    assert "Evaluation flow" in text and "No evaluation run has been published" in text
    assert [i for i in at.info] and at.text_input[0].proto.type == 1  # PASSWORD


def test_page_key_flow_sends_header_only_and_denial_shows_status_text_only(paths, monkeypatch):
    r, v, tmp = paths
    seen = {}

    def fake_get(url, headers=None, timeout=None, **kw):
        seen.update(url=url, headers=headers, kw=kw)

        class R:
            status_code = 403
            def json(self): return {}
        return R()
    monkeypatch.setattr("requests.get", fake_get)
    at = _at(r, v, tmp / "none.json")
    at.text_input[0].set_value("typed-wrong-key-0123456789").run()
    at.button[0].click().run()
    assert not at.exception
    assert seen["headers"] == {"X-Viz-Key": "typed-wrong-key-0123456789"} and "typed-wrong" not in seen["url"]
    assert tui.PRIVATE_DENIED_TEXT in _page(at) and "typed-wrong" not in "".join(repr(el.proto) for el in at.main)


def test_page_unlock_success_shows_detail(paths, monkeypatch):
    r, v, tmp = paths
    monkeypatch.setattr("vera.trace_eval.ui.fetch_private", lambda b, k: (200, trace_viz.private_view(str(r))))
    at = _at(r, v, tmp / "none.json")
    at.text_input[0].set_value(KEY).run()
    at.button[0].click().run()
    assert not at.exception and len(at.selectbox) == 2 and SENT_Q in _page(at)
