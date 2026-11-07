"""SIDEBAR-RC-1107 (owner report 2026-11-07): About described a sidebar entry "What VERA-DAR draws on" that has not
rendered since 35f21fb (HOTFIX-BLURB-1104 moved the corpus description to About only). These tests render the real
pages and check that what About says about the corpus section matches what is actually rendered, in the
deterministic-fallback state the owner saw."""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from vera import help_pages, sources_sidebar

ROOT = Path(__file__).resolve().parent.parent
PAGES = ("streamlit_app.py", "pages/about.py", "pages/how_to_use.py", "pages/trace_eval.py", "pages/vera_demo.py")
FALLBACK = {"summary_source": "deterministic_fallback", "source_count": 275, "corpus_updated_at": "2026-11-07T00:00:00+00:00",
            "summary": "275 sources. Publication years: 2005 to 2026. A written description is temporarily unavailable.",
            "topics": [], "coverage": None, "topic_shares": None, "summary_generated_at": None, "summary_model": None}
_NORM = lambda t: re.sub(r"\s+", " ", re.sub(r"[*\\]", "", str(t))).strip()  # noqa: E731  (markdown bold/escapes off)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    import requests

    def _no_net(*_a, **_k):
        raise requests.exceptions.ConnectionError("offline test")
    monkeypatch.setattr(requests, "get", _no_net)
    monkeypatch.setattr(requests, "post", _no_net)
    monkeypatch.setattr(sources_sidebar, "_corpus", lambda st: dict(FALLBACK))


def _texts(root):
    out = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "error", "success"):
        out += [_NORM(e.value) for e in getattr(root, kind)]
    out += [_NORM(getattr(e, "label", "")) for e in getattr(root, "expander", [])]
    return " ".join(out)


def _run(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_corpus_heading_is_not_in_any_sidebar(page):
    assert help_pages.CORPUS_HEADING not in _texts(_run(page).sidebar), page


def test_about_text_matches_the_rendered_corpus_section():
    """CORPUS-DESC-1107 (owner 2026-11-07): the corpus description moved from About to its own Corpus Description page.
    About keeps one link line and no corpus section; the page shows the count line once and describes itself correctly."""
    about = _texts(_run("pages/about.py").main)
    assert help_pages.CORPUS_HEADING not in about and "VERA-DAR's corpus: 275 sources" not in about
    assert "Corpus Description" in about  # the one link line (page_link label, or its text outside a multipage run)
    paras = " ".join(p for s in help_pages.load()["about"]["after_disclosure"] for p in s["paragraphs"])
    assert "sidebar" not in paras.lower() and help_pages.CORPUS_HEADING not in paras
    at = _run("pages/corpus_description.py")
    main = _texts(at.main)
    assert main.count("VERA-DAR's corpus: 275 sources, newest source 2026-11-07.") == 1
    assert "Main topics" not in main and "Description written by a language model" not in main  # fallback state
    page = " ".join(help_pages.load()["corpus_page"]["paragraphs"])
    assert "sidebar" not in page.lower() and "When no checked description is available" in page
