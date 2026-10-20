"""Streamlit AppTest: the page renders with no exceptions, from fixture and from a failed case."""
import copy
import os

from streamlit.testing.v1 import AppTest

from vera.m7.fixture import build_fixture_report


def _app(report):
    from vera.m7.demo_ui import render_demo
    render_demo(report, "test")


def _at(report):
    at = AppTest.from_function(_app, kwargs={"report": report}, default_timeout=30)
    return at.run()


def test_fixture_renders_without_exception():
    at = _at(build_fixture_report())
    assert not at.exception
    assert any("EXAMPLE REPORT" in w.value for w in at.warning)
    assert any("SUCCESS" in s.value for s in at.success)


def test_failed_case_banner():
    rep = build_fixture_report()
    ev = rep["evaluation"]
    ev["outcome"], ev["failed_dimensions"] = "failed_case", ["grounding"]
    at = _at(rep)
    assert not at.exception
    assert any("FAILED CASE" in e.value and "Grounding" in e.value for e in at.error)


def test_fabricated_citation_displays_error_not_crash():
    rep = build_fixture_report()
    rep["engineered"]["claims"][0]["citations"] = ["GHOST"]
    at = _at(rep)
    assert not at.exception
    assert any("not in the evidence corpus" in e.value for e in at.error)


def test_page_file_renders_with_fixture_env(monkeypatch):
    monkeypatch.setenv("VERA_DEMO_SOURCE", "fixture")
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "pages", "vera_demo.py"),
                           default_timeout=30).run()
    assert not at.exception
    assert at.title and "VERA-DAR" in at.title[0].value


def test_scholar_link_out_button_and_caption_are_present_and_nothing_is_requested(monkeypatch):
    import requests
    def boom(*a, **k):
        raise AssertionError("network used by the demo page")
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    at = _at(build_fixture_report())
    assert not at.exception
    btns = at.get("link_button")
    assert len(btns) == 1 and btns[0].proto.label == "Search Google Scholar yourself"
    assert btns[0].proto.url.startswith("https://scholar.google.com/scholar?q=")
    assert any("VERA-DAR does not query Google Scholar" in c.value for c in at.caption)


def test_demo_page_is_neutral_and_isolated(monkeypatch):
    """K3.I: pages/vera_demo.py with the DB stubbed to fail and no API key/network; no back-end or stage terms,
    no timestamp, no loader note. 'fixture' is not scanned: the fixture content legitimately shows on this page."""
    import re
    import socket

    import requests

    from tests.public_terms import find_terms, public_terms
    from tests.test_item78_sources_sidebar import _tree_text

    def boom(*a, **k):
        raise AssertionError("network or database used by the demo page")
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr("vera.db.get_session", lambda: (_ for _ in ()).throw(RuntimeError("EXTERNAL_DB_URL is not set")))
    monkeypatch.delenv("VERA_API_KEY", raising=False)
    path = os.path.join(os.path.dirname(__file__), "..", "pages", "vera_demo.py")
    # Stage names still appear inside fixture CONTENT (vera/m7/fixture.py rationales, audit step text), which is not a
    # K3.I file: recorded as a hand-off. Chrome (tabs, headers, expander labels) is scanned for them separately below.
    stage = {"gate a", "gate b", "gate c"}
    terms = [t for t in public_terms() if t not in stage]
    for env in ("fixture", None):
        if env:
            monkeypatch.setenv("VERA_DEMO_SOURCE", env)
        else:
            monkeypatch.delenv("VERA_DEMO_SOURCE", raising=False)
        at = AppTest.from_file(path, default_timeout=30).run()
        assert not at.exception
        text = _tree_text(at)
        assert "Example report" in text and "EXAMPLE REPORT" in text
        assert [t.label for t in at.tabs[:3]] == ["Source selection", "Evidence quality", "Answer adequacy"]
        assert not find_terms(text, terms), find_terms(text, terms)
        chrome = " ".join([t.label for t in at.tabs] + [h.value for h in at.header] + [h.value for h in at.subheader]
                          + [e.label for e in at.expander if "rationale" not in e.label.lower() or "Gate" in e.label])
        assert not find_terms(chrome), find_terms(chrome)
        for bad in ("VERA_DEMO_SOURCE", "FIXTURE fallback", "vera_vjay", "EXTERNAL_DB_URL", "RuntimeError", "M1-M5"):
            assert bad not in text, bad
        assert not re.search(r"generated \d{4}-\d{2}-\d{2}|\d{4}-\d{2}-\d{2}T\d", text)
