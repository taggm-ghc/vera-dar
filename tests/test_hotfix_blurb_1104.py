"""HOTFIX-BLURB-1104: owner 2026-11-04, "only the About page should have About-related content" (repeat regression in
build 2054f12: the lane, retention and corpus text spread to every page). Every page is rendered offline with every
disclosure-widening flag on; About-only strings must render on About and nowhere else, and the per-page notice must be
exactly the approved short line (R-031 / N1)."""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from vera import disclosure, sources_sidebar

ROOT = Path(__file__).resolve().parent.parent
OTHER_PAGES = ("pages/how_to_use.py", "pages/trace_eval.py", "pages/vera_demo.py", "pages/corpus_description.py")
_NORM = lambda t: re.sub(r"\s+", " ", t).strip()  # noqa: E731
APPROVED_SHORT = ("VERA-DAR is an AI system and its answers may be wrong. Your question is sent to third-party AI and "
                  "search services.")


@pytest.fixture(autouse=True)
def _all_widening_flags_on(monkeypatch):
    monkeypatch.setattr(disclosure, "reporting_lane_on", lambda: True)
    monkeypatch.setattr(disclosure, "_official_on", lambda: True)
    monkeypatch.setattr(disclosure, "web_read_on", lambda: True)
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")  # the deployed mode
    import requests

    def _offline(*_a, **_k):
        raise requests.exceptions.ConnectionError("offline test")
    monkeypatch.setattr(requests, "get", _offline)  # Trace Eval Live tab: API unreachable offline (handled)
    monkeypatch.setattr(requests, "post", _offline)


def _about_only_strings():
    s = [disclosure._LANE_SEARCH, disclosure._LANE_SEARCH_HN, disclosure._LANE_PAGES, disclosure._LANE_LICENCE,
         disclosure._LANE_GDELT_CREDIT, disclosure._NO_SAVE, disclosure._THIRD_PARTY, "30 days", "GDELT",
         sources_sidebar.LICENCE_SIDEBAR, sources_sidebar.MODEL_LINE, "How VERA-DAR answers:", "Trace Eval sources:",
         sources_sidebar.CORPUS_NOTES_HEADING]  # CORPUS-DESC-1107: the corpus lines moved to the Corpus Description page
    s += [t for _h, t in disclosure.disclosure_sections()]
    return [_NORM(x).strip() for x in s if _NORM(x).strip()]


def _all_text(at):
    out = []
    for root in (at.main, at.sidebar):
        for kind in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "error", "success"):
            out += [str(e.value) for e in getattr(root, kind)]
    return _NORM(" ".join(out)), [_NORM(str(c.value)) for c in at.main.caption]


def _run(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_short_notice_is_the_approved_line():
    assert disclosure.short_notice_text() == disclosure.SHORT_NOTICE == APPROVED_SHORT


@pytest.mark.parametrize("page", OTHER_PAGES)
def test_no_about_content_outside_about(page):
    text, caps = _all_text(_run(page))
    leaked = [s for s in _about_only_strings() if s in text]
    assert not leaked, (page, leaked)
    notices = [c for c in caps if c.startswith("VERA-DAR is an AI system")]
    if page == "streamlit_app.py":  # owner 2026-11-05: the AI and third-party notice is on the Ask page only
        assert notices == [APPROVED_SHORT], (page, notices)  # exactly one notice, exactly the approved short text
    else:
        assert APPROVED_SHORT not in caps and notices == [], (page, notices)


def test_about_holds_the_about_content():
    text, _caps = _all_text(_run("pages/about.py"))
    for s in (disclosure._LANE_GDELT_CREDIT, sources_sidebar.LICENCE_SIDEBAR, sources_sidebar.CORPUS_NOTES_HEADING):
        assert _NORM(s).strip() in text, s
    # PRIVACY-1104 (owner 2026-11-04: terms content migrates off About and How): retention and lane-privacy text is on Terms
    for s in (disclosure._LANE_PAGES, disclosure._NO_SAVE, "30 days"):
        assert _NORM(s).strip() not in text, s


def test_terms_holds_the_terms_content():
    text, caps = _all_text(_run("pages/terms.py"))
    assert APPROVED_SHORT not in caps  # owner 2026-11-05: the notice is on the Ask page only, not on Terms
    for s in (disclosure._LANE_PAGES, disclosure._LANE_GDELT_CREDIT, disclosure._NO_SAVE, "30 days"):
        assert _NORM(s).strip() in text, s
    assert sources_sidebar.CORPUS_NOTES_HEADING not in text  # About content stays on About
