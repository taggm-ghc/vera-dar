"""PLAN-FDA D8 / owner R-027 req. 6 condition: the partial status and the list of unanswered parts must survive into
every user-visible surface. Stubs only (stubbed LLM/checker, requests.post, router); no TestClient, no app launch, no DB.

Surfaces: (a) the /ask response body built by main.ask (called in-process); (b) the Streamlit page (AppTest, requests
stubbed); (c) status/headline text (Answer heading, success box, claim-check caption, disclosure/sidebar)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from tests import test_fulldoc_d8_parts as d8
from tests.test_item78_sources_sidebar import _tree_text
from vera import sources_sidebar as sb

ROOT = Path(__file__).resolve().parent.parent
STATEMENT = d8.PP["partial_statement"]
UNANSWERED = "Part 2 (What tolerates failures?): the documents VERA-DAR holds do not support an answer to this part."


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("FD_TEST_KEY", "x")


def partial_result():
    o = d8.run(d8.reply(d8.TWO, [d8.A1, d8.A2]), {"Gamma": "unsupported"})
    assert o.result is not None and o.result.claim_check["answer_completeness"] == "partial"
    return o.result


def _body_via_main_ask(monkeypatch, result):
    import main
    from fastapi import BackgroundTasks

    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.setattr(main, "GROUNDING", None)
    monkeypatch.setattr(main, "answer_via_chain", lambda q, entries, g: (result, "groq:m"))
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)
    monkeypatch.setattr(main, "admit_to_corpus", lambda *a: None)
    monkeypatch.setattr(main, "llm_call_for", lambda e: None)
    monkeypatch.setattr(main, "memory_from_env", lambda block: None)
    req = main.AskRequest(question="What reduces latency and what tolerates failures?")
    resp = main.ask(req, SimpleNamespace(state=SimpleNamespace()), BackgroundTasks())
    return json.loads(resp.model_dump_json())


# ---- (a) API response body ------------------------------------------------------------------------------------

def test_a_api_body_keeps_partial_statement_unanswered_lines_and_completeness(monkeypatch):
    body = _body_via_main_ask(monkeypatch, partial_result())
    assert body["answer"].startswith(STATEMENT) and UNANSWERED in body["answer"]
    assert body["claim_check"]["answer_completeness"] == "partial"
    assert body["claim_check"]["parts"] == {"total": 2, "answered": [1], "unanswered": [2]}
    assert body["result_kind"] == "answer"  # OWNER GATE fallback design: no distinct kind passes the guarded seams
    # GAP (guarded main.py EvidenceCitation has no `part` field): the public citation drops the internal part id;
    # citations are still only for the answered part (their claim ids belong to part 1) and keep span/attribution.
    assert body["citations"] and all("part" not in c and c["span_id"] and c["attribution"] for c in body["citations"])
    assert "Gamma replication" not in body["answer"]  # the unsupported part's claim text is never published


def test_a_api_body_complete_answer_has_no_partial_markers(monkeypatch):
    o = d8.run(d8.reply(d8.TWO, [d8.A1, d8.A2]))
    body = _body_via_main_ask(monkeypatch, o.result)
    assert body["claim_check"]["answer_completeness"] == "complete" and STATEMENT not in body["answer"]
    assert body["claim_check"]["parts"]["unanswered"] == []


# ---- (b) Streamlit page ---------------------------------------------------------------------------------------

def _render(monkeypatch, body):
    import requests

    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_KEY", "k")
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: body))
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(json=lambda: {"commit": "abcdef1234"}))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    at.text_input[0].set_value("What reduces latency and what tolerates failures?")
    at.button[0].click()
    at.run()
    assert not at.exception
    return at


def _ui_body(monkeypatch):
    b = _body_via_main_ask(monkeypatch, partial_result())
    b.update(sources=[], traced_sources=[])  # as in a real /ask answer the sources list is rendered separately
    return b


def test_b_streamlit_renders_partial_statement_and_unanswered_part(monkeypatch):
    at = _render(monkeypatch, _ui_body(monkeypatch))
    text = _tree_text(at)
    assert STATEMENT in text and "do not support an answer to this part" in text and "Part 2" in text


def test_b_streamlit_complete_answer_has_no_partial_statement(monkeypatch):
    o = d8.run(d8.reply(d8.TWO, [d8.A1, d8.A2]))
    b = _body_via_main_ask(monkeypatch, o.result)
    b.update(sources=[], traced_sources=[])
    assert STATEMENT not in _tree_text(_render(monkeypatch, b))


# ---- (c) status / headline text -------------------------------------------------------------------------------

def test_c_answer_heading_does_not_present_partial_as_complete(monkeypatch):
    at = _render(monkeypatch, _ui_body(monkeypatch))
    assert not at.success, "partial answer shown in a green success box"
    assert "### ✅ Answer" not in [m.value for m in at.markdown]
    assert "### ⚠️ Partial answer" in [m.value for m in at.markdown]
    assert len(at.warning) == 1 and STATEMENT in at.warning[0].value


def test_c_complete_answer_still_renders_answer_heading_in_success_box(monkeypatch):
    o = d8.run(d8.reply(d8.TWO, [d8.A1, d8.A2]))
    b = _body_via_main_ask(monkeypatch, o.result)
    b.update(sources=[], traced_sources=[])
    at = _render(monkeypatch, b)
    assert "### ✅ Answer" in [m.value for m in at.markdown] and len(at.success) == 1 and not at.warning


def test_c_no_verified_or_complete_badge_text_for_partial(monkeypatch):
    at = _render(monkeypatch, _ui_body(monkeypatch))
    outside = _tree_text(at).replace(STATEMENT, "")
    for word in ("fully answered", "complete answer", "all parts", "verified answer", "fully supported"):
        assert word not in outside.lower(), word


def test_c_claim_check_caption_does_not_hide_partial_state(monkeypatch):
    """The caption 'Claim check: N cited claims checked ... removed.' (vera/sources_sidebar.py:183-186) reads only
    checked/supported/partial/removed and is silent on unanswered parts; recorded as a gap, not a completeness claim."""
    at = _render(monkeypatch, _ui_body(monkeypatch))
    caps = [c.value for c in at.caption]
    cc = [c for c in caps if c.startswith("Claim check:")]
    assert len(cc) == 1 and "complete" not in cc[0].lower() and "all " not in cc[0].lower()
