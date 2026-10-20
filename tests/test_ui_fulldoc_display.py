"""AE-1020/WP-D: the Ask page shows a full-document answer truthfully (offline; stubbed requests, no DB/network)."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_item78_sources_sidebar import _tree_text
from vera import sources_sidebar as sb
from vera import ui_evidence as ue

ROOT = Path(__file__).resolve().parent.parent
TITLE, URL = "Secret Title of Paper", "https://example.org/paper1"
QUOTE = "latency falls when requests are batched"


def _body(**kw):
    b = {"answer": "Batching reduces latency [1].", "model": "m", "tokens_used": 10, "cost_usd": 0.0,
         "result_kind": "answer", "evidence_status": "admitted",
         "sources": [{"n": 1, "title": TITLE, "url": URL, "published": "2024", "identifier": "doi:10/x",
                      "acquisition": "full_document_acquired", "span": "chars=10-90"}],
         "traced_sources": [], "recalled": [],
         "examined_full_documents": [{"document_id": "d1"}, {"document_id": "d2"}],
         "admitted_evidence": [{"span_id": "s1", "document_id": "d1", "span": "chars=10-90"}],
         "claim_check": {"outcome": "checked", "checked": 1, "supported": 1, "partial": 0, "removed": 0,
                         "uncited_dropped": 2, "unchecked_dropped": 0, "verbatim_dropped": 0,
                         "answer_completeness": "partial", "parts": {"total": 2, "answered": [1], "unanswered": [2]}},
         "citations": [{"claim_id": 1, "n": 1, "span_id": "s1", "span": "chars=10-90", "verdict": "supported",
                        "quote": QUOTE, "quote_span": "chars=20-60",
                        "attribution": {"creator": "A. Author", "title": TITLE, "source": URL, "licence": "CC BY 4.0",
                                        "licence_url": "https://creativecommons.org/licenses/by/4.0/", "changes": "none"}}]}
    b.update(kw)
    return b


def _render(monkeypatch, body, page="streamlit_app.py"):
    import requests
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_KEY", "k")
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: body))
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(json=lambda: {"commit": "abcdef1234"}))
    at = AppTest.from_file(str(ROOT / page), default_timeout=30).run()
    at.text_input[0].set_value("q")
    at.button[0].click()
    at.run()
    assert not at.exception
    return at


def _main_text(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info])


def test_answer_shows_quote_span_attribution_scope_and_no_abstract_wording(monkeypatch):
    at = _render(monkeypatch, _body())
    t = _main_text(at)
    assert QUOTE in t and "chars" in t and "20" in t
    assert "2 documents inspected, 1 passage admitted" in t
    assert "A\\. Author" in t and "CC BY 4\\.0" in t
    assert "parts answered: 1 of 2" in t and "2 left out" in t
    scoped = " ".join([m.value for m in at.markdown if m.value.startswith("**Sources**")]
                      + [c.value for c in at.caption if c.value.startswith("Claim check")])
    assert scoped and "abstract" not in scoped.lower()
    assert "Automated checks" in t


def test_quote_none_still_shows_span_reference(monkeypatch):
    b = _body()
    b["citations"][0].update(quote=None, quote_span=None)
    t = _main_text(_render(monkeypatch, b))
    assert "excerpt not shown" in t and "chars" in t


def test_insufficient_fallback_reason_no_raw_token_no_full_doc_claim(monkeypatch):
    b = {"answer": "x", "model": "m", "tokens_used": 1, "cost_usd": 0.0, "result_kind": "reading_list",
         "evidence_status": "insufficient", "examined_full_documents": [{"document_id": "d1"}], "admitted_evidence": [],
         "contract": {"layers": [], "fallback": "insufficient_full_document_evidence"},
         "candidates": [{"title": "T", "url": "https://example.org/t", "acquisition": "not_acquired"}]}
    at = _render(monkeypatch, b)
    t = _main_text(at)
    assert ue.fallback_text("insufficient_full_document_evidence", n_inspected=1) in [i.value for i in at.info]
    assert "insufficient_full_document_evidence" not in _tree_text(at)
    assert "1 document inspected, 0 passages admitted" in t
    assert "full document acquired" not in t.lower()


def test_unknown_and_missing_fallback_tokens_are_generic():
    assert ue.fallback_text("weird_internal_code") == ue.GENERIC_FALLBACK == ue.fallback_text(None) == ue.fallback_text(5)
    for tok in ue._TOKEN_DEFAULT:
        assert tok not in ue.fallback_text(tok) and ue.fallback_text(tok)


@pytest.mark.parametrize("mut", [
    {"citations": "bad"}, {"citations": [None, 3, {"n": "x", "attribution": "z", "quote": 5}]},
    {"claim_check": {"outcome": "checked", "checked": "x", "parts": "q"}}, {"claim_check": "junk"},
    {"sources": [{"n": "zz", "title": None}, 7]}, {"examined_full_documents": "x", "admitted_evidence": None},
    {"tokens_used": None, "cost_usd": None}])
def test_malformed_data_does_not_crash(monkeypatch, mut):
    _render(monkeypatch, _body(**mut))


def test_non_checked_outcome_says_not_checked():
    assert ue.claim_check_text({"outcome": "checker_unavailable"}) == ue.NOT_CHECKED_NOTE
    assert ue.claim_check_text(None) is None


def test_legacy_abstract_wording_kept_for_abstract_rows(monkeypatch):
    b = _body(evidence_status=None, citations=[])
    b["sources"][0].pop("acquisition")
    b.pop("evidence_status")
    t = _main_text(_render(monkeypatch, b))
    assert "abstracts" in t and "against the cited abstract text" in t


def test_sidebar_has_no_titles_quotes_or_urls(monkeypatch):
    at = _render(monkeypatch, _body())
    side = " ".join(c.value for c in at.sidebar.caption) + " ".join(m.value for m in at.sidebar.markdown)
    for forbidden in (TITLE, URL, QUOTE, "A. Author", "A\\. Author", "inspected, 1 passage"):
        assert forbidden not in side
    assert not ue._TOKEN_DEFAULT.keys() & set(side.split())


@pytest.mark.parametrize("page", ["streamlit_app.py", "pages/vera_demo.py", "pages/trace_eval.py"])
def test_disclosure_on_every_page(monkeypatch, page):
    from vera.disclosure import _BASE
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    import requests
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(json=lambda: {"commit": "abcdef1234"}))
    at = AppTest.from_file(str(ROOT / page), default_timeout=30).run()
    assert not at.exception
    assert any(_BASE[:60] in c.value for c in at.caption)


def test_static_guard_references():
    app = (ROOT / "streamlit_app.py").read_text()
    assert "render_citations" in app and "render_evidence_scope" in app
    assert "examined_full_documents" in (ROOT / "vera" / "ui_evidence.py").read_text()


@pytest.mark.parametrize("token", ["insufficient_full_document_evidence", "model_insufficient"])
def test_fallback_text_has_no_unfilled_placeholder(monkeypatch, token):
    b = {"answer": "x", "model": "m", "tokens_used": 1, "cost_usd": 0.0, "result_kind": "reading_list",
         "evidence_status": "insufficient", "examined_full_documents": [{"document_id": "d1"}, {"document_id": "d2"}],
         "admitted_evidence": [], "contract": {"layers": [], "fallback": token}, "candidates": []}
    at = _render(monkeypatch, b)
    infos = " ".join(i.value for i in at.info)
    assert "{" not in infos and "}" not in infos and "inspected 2 full document" in infos


def test_long_quote_trimmed_to_config_cap(monkeypatch):
    b = _body()
    b["citations"][0]["quote"] = "w" * 900
    t = _main_text(_render(monkeypatch, b))
    assert "w" * ue._quote_cap() in t and "w" * (ue._quote_cap() + 1) not in t
