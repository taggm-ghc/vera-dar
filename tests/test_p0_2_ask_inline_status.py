"""P0-2 fix 1: the Ask no-answer branch shows the inline-acquisition status (live_report wording); nothing when absent."""
import json
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_discovery_panel import ROOT, body
from vera import sources_sidebar as sb
from vera.m7.live_report import INLINE_STATUS


def _run(monkeypatch, data):
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_KEY", "k")
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: data))
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(json=lambda: {"commit": "abcdef1234"}))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    at.text_input[0].set_value("What is VERA-DAR?")
    at.button[0].click()
    at.run()
    assert not at.exception
    return "\n".join(str(c.value) for c in at.caption)


@pytest.mark.parametrize("status", sorted(INLINE_STATUS))
def test_status_shown_with_report_wording(monkeypatch, status):
    ia = {"status": status, "documents": [{"title": "SENT_TITLE", "url": "https://SENT_HOST.example/x", "reason": "SENT_CODE"}]}
    caps = _run(monkeypatch, body("reading_list", contract={"inline_acquisition": ia}))
    assert "**New search for this question:** " + INLINE_STATUS[status] in caps
    for bad in ("SENT_TITLE", "SENT_HOST", "SENT_CODE"):
        assert bad not in caps


def test_unknown_status_uses_report_fallback(monkeypatch):
    caps = _run(monkeypatch, body("reading_list", contract={"inline_acquisition": {"status": "weird"}}))
    assert "A new search was considered (outcome: other)." in caps


@pytest.mark.parametrize("contract", [None, {}, {"inline_acquisition": None}, {"inline_acquisition": {}}, "bad"])
def test_nothing_new_when_absent(monkeypatch, contract):
    d = body("reading_list")
    if contract is not None:
        d["contract"] = contract
    assert "New search for this question" not in _run(monkeypatch, d)


def test_metrics_note_attributes_zero_cost_to_free_tier():
    src = (ROOT / "streamlit_app.py").read_text()
    lr = (ROOT / "vera" / "m7" / "live_report.py").read_text()
    assert "Calls to the free-tier provider are recorded at $0, and the answer checker is not priced." in src
    assert "0 tokens means no counted model call ran." in src
    assert "0 means no counted model call ran" not in src
    assert "calls to the free-tier provider are recorded at $0" in lr
