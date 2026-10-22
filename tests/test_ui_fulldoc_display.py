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


@pytest.mark.parametrize("fulldoc", [False, True])
@pytest.mark.parametrize("page", ["streamlit_app.py", "pages/vera_demo.py", "pages/trace_eval.py"])
def test_disclosure_on_every_page(monkeypatch, page, fulldoc):
    from vera import disclosure
    from vera.disclosure import _BASE, _BASE_FULLDOC
    monkeypatch.setattr(disclosure, "fulldoc_enabled", lambda *a, **k: fulldoc)  # off / P1.6 on
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    import requests
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(json=lambda: {"commit": "abcdef1234"}))
    at = AppTest.from_file(str(ROOT / page), default_timeout=30).run()
    assert not at.exception
    if not fulldoc:
        assert any(_BASE[:60] in c.value for c in at.caption)
    else:  # state 2: the full state-2 base text, including the passage-transmission sentence
        assert any(_BASE_FULLDOC in c.value for c in at.caption)


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
    assert "{" not in infos and "}" not in infos and "searched 2 full document" in infos


def test_long_quote_trimmed_to_config_cap(monkeypatch):
    b = _body()
    b["citations"][0]["quote"] = "w" * 900
    t = _main_text(_render(monkeypatch, b))
    assert "w" * ue._quote_cap() in t and "w" * (ue._quote_cap() + 1) not in t


def test_db_source_answer_renders_answer_sources_and_evidence(monkeypatch):
    """A full-document answer whose corpus came from the DB reader (source=db) renders like any other (UI is source-agnostic)."""
    at = _render(monkeypatch, _body(source="db", corpus_source={"kind": "db"}))
    t = _main_text(at)
    assert "Partial answer" in t and TITLE in t  # answer heading (completeness is partial in _body) and the Sources block
    assert QUOTE in t and "2 documents inspected, 1 passage admitted" in t
    assert any(c.value == "Evidence source: stored database." for c in at.caption)
    assert "A\\. Author" in t and "CC BY 4\\.0" in t
    assert not any("abstract" in c.value.lower() for c in at.caption if c.value.startswith("Claim check"))


@pytest.mark.parametrize("cs,expect", [({"kind": "bundle"}, "Evidence source: local files."), ({"kind": "x"}, None),
                                       (None, None), ("db", None)])
def test_evidence_source_caption(monkeypatch, cs, expect):
    at = _render(monkeypatch, _body(corpus_source=cs))
    caps = [c.value for c in at.caption if c.value.startswith("Evidence source")]
    assert caps == ([expect] if expect else [])


# ---- DAR-UI-1022 ----
DOI = "https://doi.org/10.4103/ijem.ijem_546_20"


def test_doi_with_dots_and_underscores_is_an_unescaped_link(monkeypatch):
    b = _body()
    b["sources"][0].update(title=DOI, url=DOI, identifier=DOI)
    at = _render(monkeypatch, b)
    t = _main_text(at)
    assert f"[{DOI}]({DOI})" in t and "doi\\.org" not in t and "ijem\\_546" not in t
    b["sources"][0].update(url=None)  # URL only as the title: still a clickable link, not escaped text
    assert f"[{DOI}]({DOI})" in _main_text(_render(monkeypatch, b))


def _multi():
    srcs = [{"n": i, "title": TITLE, "url": URL, "published": "2024", "document_id": "d1", "span": f"chars={i}0-{i}9",
             "acquisition": "full_document_acquired"} for i in range(1, 7)]
    return _body(sources=srcs, examined_full_documents=[{"document_id": "d1", "canonical_source_id": "c1"},
                                                        {"document_id": "d9", "canonical_source_id": "c1"}],
                 admitted_evidence=[{"span_id": f"s{i}", "document_id": "d1", "span": f"chars={i}0-{i}9"} for i in range(1, 7)])


def test_sources_grouped_one_entry_per_document_cited_first(monkeypatch):
    at = _render(monkeypatch, _multi())
    rows = [m.value for m in at.markdown if TITLE in m.value]
    assert len(rows) == 1 and rows[0].startswith("[1] ")
    caps = " ".join(c.value for c in at.caption)
    assert "passages used: chars=10\\-19" in caps and "also admitted, not cited: chars=20\\-29" in caps


def test_scope_counts_distinct_documents_inspected_and_used(monkeypatch):
    t = _main_text(_render(monkeypatch, _multi()))
    assert "1 document inspected, 6 passages admitted as evidence (from 1 document)" in t


@pytest.mark.parametrize("ia,expect", [
    ({"documents": [{"reason": "below_persist_min"}], "reused": 0, "n_persisted": 0}, "documents used for this answer only (not saved)"),
    ({"documents": [{"reason": "below_persist_min"}], "reused": 1, "n_persisted": 0}, "stored database and documents used for this answer only"),
    ({"documents": [], "reused": 1, "n_persisted": 0}, "stored database."),
])
def test_evidence_source_mix_from_inline_acquisition(monkeypatch, ia, expect):
    at = _render(monkeypatch, _body(corpus_source={"kind": "db"}, contract={"inline_acquisition": ia}))
    caps = [c.value for c in at.caption if c.value.startswith("Evidence source")]
    assert len(caps) == 1 and expect in caps[0]


def test_uncited_document_listed_under_also_admitted_heading(monkeypatch):
    b = _multi()
    b["sources"].append({"n": 7, "title": "Herpes paper", "url": "https://example.org/h", "document_id": "d2",
                         "span": "chars=1-9", "acquisition": "full_document_acquired"})
    at = _render(monkeypatch, b)
    md = [m.value for m in at.markdown]
    i = next(k for k, v in enumerate(md) if v.startswith("**Also admitted, not cited**"))
    assert any("Herpes paper" in v for v in md[i + 1:]) and not any("Herpes paper" in v for v in md[:i])


# --- SEC-1022-A fixes U1-U4 ---------------------------------------------------------------------------------------
def test_u1_cited_set_comes_from_answer_markers_not_citation_rows(monkeypatch):
    assert sb.cited_numbers("A [1] and B [2, 3].") == {1, 2, 3} and sb.cited_numbers("none") == set()
    assert sb.cited_numbers(None) is None
    b = _multi()
    b["sources"].append({"n": 7, "title": "Seventh paper", "url": "https://example.org/s", "document_id": "d7",
                         "span": "chars=1-9", "acquisition": "full_document_acquired"})
    b["answer"] = "Batching reduces latency [1]. Another claim [7]."  # [7] has no citation row (no attribution)
    b["citations"] = [c for c in b["citations"] if c["n"] == 1]
    md = [m.value for m in _render(monkeypatch, b).markdown]
    assert not any(v.startswith("**Also admitted, not cited**") for v in md)
    b["answer"] = "Batching reduces latency [1]."
    md = [m.value for m in _render(monkeypatch, b).markdown]
    assert any(v.startswith("**Also admitted, not cited**") for v in md)


@pytest.mark.parametrize("title,url,must_not", [
    ("javascript:alert(1)", None, "]("),
    ("data:text/html;base64,AAAA", None, "]("),
    ("https://a.test/x)[y](javascript:z)", None, "javascript"),
    ("[click](javascript:alert(1))", "javascript:alert(1)", "](javascript"),
    ("<img src=x onerror=alert(1)>", None, "<img"),
    ("https://x.test/a\n", None, "\n)"),
    ("Plain title", "https://x.test/a\nb", "\n"),
])
def test_u2_u4_hostile_titles_and_urls_never_become_unsafe_links(title, url, must_not):
    import re
    out = sb._link(title, url)
    m = re.fullmatch(r"\[(?:[^\]\\]|\\.)*\]\((https?://[^\s()<>\"'\x00-\x1f]+)\)", out)
    if m:  # a link: its single target is a clean http(s) URL and the visible text has no unescaped markup
        assert m.group(1) in (url, title)
    else:  # plain text: nothing that Markdown or HTML would act on
        assert not re.search(r"(?<!\\)(\]\(|<|`)", out)
    assert not re.search(r"(?<!\\)\]\((javascript|data):", out) and "\n" not in out
    assert not sb._URLISH.fullmatch("https://x\n") and sb._URLISH.fullmatch("https://x.test/a")
    assert sb.plain_url("https://x.test/a\n").startswith("https") and "\\." in sb.plain_url("https://x.test/a\n")


def test_u4_render_sources_with_hostile_and_sentinel_host_titles(monkeypatch):
    b = _body(sources=[{"n": 1, "title": "javascript:alert(1)", "url": "javascript:alert(1)", "document_id": "d1",
                        "span": "chars=1-9", "acquisition": "full_document_acquired"},
                       {"n": 2, "title": "https://sentinel-host.invalid/x\n", "url": None, "document_id": "d2",
                        "span": "chars=1-9", "acquisition": "full_document_acquired"}],
              answer="One [1]. Two [2].")
    at = _render(monkeypatch, b)
    md = " ".join(m.value for m in at.markdown)
    assert "](javascript" not in md and "](data:" not in md and "<img" not in md
    assert "\n)" not in md and ".invalid/x\n]" not in md
    for host_var in ("upstream.invalid", "API_BASE_URL"):  # the API base never appears in the rendered tree
        assert host_var not in _tree_text(at)


@pytest.mark.parametrize("ia,adm,expect", [
    ({"documents": [{"reason": "below_persist_min"}], "reused": 0, "n_persisted": 0}, [], "none were admitted as evidence"),
    ({"documents": [{"reason": "below_persist_min"}], "reused": 1, "n_persisted": 0}, [], "none were admitted as evidence"),
    ({"documents": [{"reason": "below_persist_min"}], "reused": 0, "n_persisted": 0},
     [{"span_id": "s1", "document_id": "d1", "span": "chars=1-9"}], "documents used for this answer only (not saved)"),
])
def test_u3_this_answer_only_requires_admitted_evidence(monkeypatch, ia, adm, expect):
    at = _render(monkeypatch, _body(corpus_source={"kind": "db"}, contract={"inline_acquisition": ia}, admitted_evidence=adm))
    caps = [c.value for c in at.caption if c.value.startswith("Evidence source")]
    assert len(caps) == 1 and expect in caps[0]
