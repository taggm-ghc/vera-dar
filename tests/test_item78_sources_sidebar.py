"""Item #78: 'What VERA-DAR draws on' sidebar, read only from shipped files."""
import json
import random
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from vera import sources_sidebar as sb

ROOT = Path(__file__).resolve().parent.parent


def test_scope_and_providers_come_from_config():
    assert sb.scope_text() == json.loads((ROOT / "config" / "scope_router.json").read_text())["scope"]
    names = sb.answer_providers()
    assert names[0].startswith("groq ") and names[-1].startswith("openai ")
    assert not any("http" in n or "api_key" in n.lower() for n in names)


def test_snapshot_summary_from_results(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"generated_at": "2030-01-02T00:00:00+00:00", "traces": [
        {"id": "q1", "sources": [{"url": "https://arxiv.org/abs/2401.00001", "title": "A"},
                                 {"url": "https://arxiv.org/abs/2603.12345", "title": "B"}]},
        {"id": "q2", "sources": [{"url": "https://arxiv.org/abs/2401.00001", "title": "A"}]}]}))
    s = sb.snapshot_summary(p)
    assert s["count"] == 2 and s["questions"] == 2 and s["years"] == (2024, 2026) and s["captured"] == "2030-01-02"
    assert sb.snapshot_summary(tmp_path / "missing.json") is None


def test_titles_render_as_plain_text():
    assert sb.plain("[click](javascript:x) *bold*") == r"\[click\]\(javascript:x\) \*bold\*"


def _page():
    import streamlit as st
    from vera.sources_sidebar import render_sources_sidebar
    render_sources_sidebar(st)


@pytest.mark.parametrize("fulldoc", [False, True])
def test_sidebar_renders_honest_answer_source_without_hosts_or_dates_in_code(monkeypatch, fulldoc):
    from vera import disclosure
    monkeypatch.setattr(disclosure, "fulldoc_enabled", lambda *a, **k: fulldoc)
    at = AppTest.from_function(_page)
    at.run(timeout=30)
    assert not at.exception
    text = " ".join(c.value for c in at.sidebar.caption)
    if fulldoc:  # state 2 (shipped after P1.6)
        assert sb._HOW_FULLDOC in text and "lists documents it has not read" in text
    else:
        assert "searches published scholarly sources and the web live" in text and "does not use abstracts or snippets as evidence" in text
        assert "instead of answering from memory" in text
    assert "onrender.com" not in text and "127.0.0.1" not in text
    src = (ROOT / "vera" / "sources_sidebar.py").read_text()
    assert not re.search(r"20\d\d-\d\d-\d\d", src)


def test_every_page_calls_the_sidebar():
    for rel in ("streamlit_app.py", "pages/trace_eval.py", "pages/vera_demo.py"):
        assert "render_sources_sidebar(st)" in (ROOT / rel).read_text(), rel


def test_source_labels_do_not_overclaim_inspection():
    """DEC-EI1: no 'followed back to the original papers', no 'Evidence snapshot'; licences are intermediary-reported."""
    at = AppTest.from_function(_page)
    at.run(timeout=30)
    assert not at.exception
    text = " ".join(c.value for c in at.sidebar.caption)
    assert "followed back" not in text and "original papers" not in text and "Evidence snapshot" not in text
    assert "published scholarly sources and the web" in text and "search snippets" not in text and "Trace Eval sources" in text
    cap = []

    class St:
        def markdown(self, *a, **k): pass
        def caption(self, t): cap.append(t)

    sb.render_sources(St(), [{"n": 1, "title": "T", "url": "https://x.org/a", "licence_decision": "allow"}],
                      [{"title": "O", "url": "https://x.org/o"}])
    joined = " ".join(cap)
    assert "not verified with the publisher" in joined
    assert "traced from web results to the original source" not in joined and "matched to a published record" in joined
    assert "free abstract" not in joined


def test_licence_caption_absent_without_licence_decision():
    cap = []

    class St:
        def markdown(self, *a, **k): pass
        def caption(self, t): cap.append(t)

    sb.render_sources(St(), [{"n": 1, "title": "T", "url": "https://x.org/a"}], [])
    assert not any("Licence decisions" in c for c in cap)


def test_cap_comes_from_config(tmp_path):
    """The cap in public text is read from config (no literal in code); unreadable config shows no number."""
    cfg = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())
    assert sb.grounding_cap() == cfg["grounding"]["snippet_max_chars"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"grounding": {"snippet_max_chars": 2000}}))
    assert sb.grounding_cap(p) == 2000
    assert sb.grounding_cap(tmp_path / "missing.json") is None
    p.write_text(json.dumps({"grounding": {}}))
    assert sb.grounding_cap(p) is None
    src = (ROOT / "vera" / "sources_sidebar.py").read_text()
    assert not re.search(r"1,500|\b1500\b", src)


def _capture():
    cap = []

    class St:
        def markdown(self, t, *a, **k): cap.append(t)
        def caption(self, t): cap.append(t)

    return St(), cap


def test_cap_text_follows_config_and_is_omitted_when_unreadable(monkeypatch):
    st, cap = _capture()
    monkeypatch.setattr(sb, "grounding_cap", lambda path=None: 2000)
    sb.render_sources(st, [{"n": 1, "title": "T", "url": "https://x.org/a"}], [],
                      {"outcome": "checked", "checked": 1, "supported": 1, "partial": 0, "removed": 0})
    assert "2,000 characters" in " ".join(cap) and "1,500" not in " ".join(cap)
    st, cap = _capture()
    monkeypatch.setattr(sb, "grounding_cap", lambda path=None: None)
    sb.render_sources(st, [{"n": 1, "title": "T", "url": "https://x.org/a"}], [],
                      {"outcome": "checked", "checked": 1, "supported": 1, "partial": 0, "removed": 0})
    joined = " ".join(cap)
    assert "characters" not in joined and not re.search(r"\d,\d{3}", joined)


# ---- Owner requirement 2026-10-15: the sidebar NEVER discloses a specific individual source/doc/quote ----
_TITLES = ["Zebra Quokka Transformers Revisited", "Obscure Lemur Attention Study", "Plover Gradient Folklore"]
_URLS = ["https://arxiv.org/abs/2401.00001", "https://arxiv.org/abs/2603.12345", "https://example.org/doc/zz-77"]
_QUOTES = ["unmistakable verbatim quote alpha", "another verbatim quote beta"]


def _fixture_snapshot(tmp_path):
    p = tmp_path / "snap.json"
    p.write_text(json.dumps({"generated_at": "2030-01-02T00:00:00+00:00", "traces": [
        {"id": "q1", "sources": [{"url": u, "title": t, "source_id": f"SID-{i}", "snippet": _QUOTES[i % 2],
                                  "quote": _QUOTES[i % 2]} for i, (u, t) in enumerate(zip(_URLS, _TITLES))]}]}))
    return p


def _sidebar_text(monkeypatch, snap_path):
    monkeypatch.setattr(sb, "SNAPSHOT_PATH", snap_path)
    monkeypatch.setattr(sb, "snapshot_summary", lambda path=snap_path, _f=sb.snapshot_summary: _f(path))
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    cap = []

    class Exp:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class Side:
        def expander(self, *a, **k): return Exp()

    class St:
        sidebar = Side()
        def caption(self, t): cap.append(t)
        def markdown(self, t): cap.append(t)

    sb.render_sources_sidebar(St())
    return " ".join(cap)


def test_sidebar_never_discloses_individual_sources(tmp_path, monkeypatch):
    text = _sidebar_text(monkeypatch, _fixture_snapshot(tmp_path))
    assert "3 distinct sources" in text  # aggregate still shown
    for needle in _TITLES + _URLS + _QUOTES + ["SID-0", "SID-1", "SID-2", "random sample", "sample of titles", "•"]:
        assert needle not in text, needle


def test_snapshot_summary_has_only_aggregate_fields(tmp_path):
    s = sb.snapshot_summary(_fixture_snapshot(tmp_path))
    assert set(s) == sb.SNAPSHOT_FIELDS
    blob = json.dumps(s)
    for needle in _TITLES + _URLS + _QUOTES:
        assert needle not in blob


def test_sidebar_builder_takes_no_sampling_argument_and_source_has_no_title_sampling():
    import inspect
    assert list(inspect.signature(sb.render_sources_sidebar).parameters) == ["st"]
    body = inspect.getsource(sb.render_sources_sidebar) + inspect.getsource(sb.snapshot_summary)
    assert "sample(" not in body and '"titles"' not in body and "random" not in body


_INTERNAL_TERMS = ["Gate A", "Gate B", "Gate C", "licence gate", "write gate", "admitted by", "DEC-", "vera_vjay",
                   "milestone", "confirm_required"]


def _no_internal_terms(text):
    import re
    assert not re.search(r"#\d{2,3}\b|\bM[1-7]\b|\bitem \d", text), text
    for term in _INTERNAL_TERMS:
        assert term.lower() not in text.lower(), term


def test_sidebar_and_disclosure_show_no_internal_pipeline_terms(tmp_path, monkeypatch):
    from vera import disclosure
    text = _sidebar_text(monkeypatch, _fixture_snapshot(tmp_path))
    _no_internal_terms(text)
    _no_internal_terms(disclosure.DISCLOSURE_TEXT + " " + disclosure.DISCLOSURE_TEXT_UNCONFIRMED)
    cap = []

    class St:
        def caption(self, t): cap.append(t)

    sb.render_corpus(St(), {"summary_source": "agentic_model", "source_count": 19, "corpus_updated_at": "2026-10-13"})
    assert cap[0].startswith("**VERA-DAR's corpus:** 19 sources, newest source 2026\\-10\\-13")
    _no_internal_terms(" ".join(cap))


# ---- K3.I: model/provider/gate names never reach public text (behavioural: sentinels, plus the shared term list) ----
from tests.public_terms import find_terms, public_terms  # noqa: E402

SENT_CHECKER = "SENTCHECKER9"
SENT_MODEL = "SENTMODEL2"


def test_public_terms_are_config_derived_and_catch_the_live_checker():
    terms = public_terms()
    for needed in ("qwen/qwen3.8-27b", "qwen3.8-27b", "duckduckgo", "gpt-4.1-nano", "openai/gpt-oss-120b", "arxiv"):
        assert needed in terms, needed


def test_no_model_names_in_any_public_render(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "answer_providers", lambda *a, **k: (_ for _ in ()).throw(AssertionError("not used")))
    text = _sidebar_text(monkeypatch, _fixture_snapshot(tmp_path))
    assert not find_terms(text), find_terms(text)
    assert sb.MODEL_LINE in text and "in the order tried" not in text.lower()
    st, cap = _capture()
    sb.render_sources(st, [{"n": 1, "title": "T", "url": "https://x.org/a", "licence_decision": "LICDEC9",
                            "provider": SENT_MODEL, "retrieved_at": "2030-01-01T00:00:00", "identifier": "id1"}],
                      [], {"outcome": "checked", "checker": SENT_CHECKER, "checked": 2, "supported": 2, "partial": 0,
                           "removed": 0})
    joined = " ".join(cap)
    assert SENT_CHECKER not in joined and SENT_MODEL not in joined and "2030-01-01" not in joined
    assert "LICDEC9" not in joined and "id1" in joined
    assert not find_terms(joined), find_terms(joined)


def test_description_line_ignores_summary_model_and_unknown_fields():
    st, cap = _capture()
    base = {"summary_source": "agentic_model", "source_count": 4, "summary_generated_at": "2026-10-13T00:00:00",
            "summary": "A short plain description.", "topics": ["a", "b"]}
    sb.render_corpus(st, {**base, "summary_model": "groq:openai/gpt-oss-20b", "extra": SENT_MODEL})
    st2, cap2 = _capture()
    sb.render_corpus(st2, base)
    assert cap == cap2
    joined = " ".join(cap)
    assert "written by a language model on 2026" in joined and "groq" not in joined and SENT_MODEL not in joined
    assert sb.CORPUS_FIELDS == {"summary_source", "source_count", "corpus_updated_at", "summary", "topics",
                                "summary_generated_at"}


def test_model_written_description_naming_a_backend_is_replaced():
    st, cap = _capture()
    sb.render_corpus(st, {"summary_source": "agentic_model", "source_count": 4, "summary": "Written by gpt-4.1-nano.",
                          "topics": ["groq"]})
    joined = " ".join(cap)
    assert "gpt" not in joined and "groq" not in joined and "description not available" in joined


def test_real_fallback_summary_is_public_safe_and_renders():
    from vera.corpus_description import fallback_summary
    st, cap = _capture()
    for ov, prof in (({"source_count": 0}, {}),
                     ({"source_count": 7}, {"source_types": {"arxiv_abs": 5, "web_page": 2}, "published_years": ["2020", "2025"]}),
                     ({"source_count": 7}, {"source_types": {"x": 1}, "published_years": [None, None]})):
        text = fallback_summary(ov, prof)
        assert not find_terms(text) and "Types" not in text and "arxiv_abs" not in text, text
        sb.render_corpus(st, {"summary_source": "deterministic_fallback", "source_count": 7, "summary": text})
    assert not find_terms(" ".join(cap))


def test_licence_and_heading_reworded(tmp_path, monkeypatch):
    text = _sidebar_text(monkeypatch, _fixture_snapshot(tmp_path))
    assert "**Trace Eval sources:**" in text and "first posted 2024 to 2026" in text
    for old in ("index records", "metadata terms", "arXiv abstract records", "distinct arXiv abstracts", "arXiv identifiers"):
        assert old not in text, old
    st, cap = _capture()
    sb.render_corpus(st, {"summary_source": "agentic_model", "source_count": 1})
    assert sb.LICENCE_SIDEBAR in " ".join(cap) and "not verified with the publisher" in sb.LICENCE_SIDEBAR


def test_nav_labels_and_streamlit_config():
    assert tuple(label for _, label in sb.NAV) == ("Ask VERA-DAR", "Example report", "Trace Eval")
    assert not any("streamlit" in label.lower() for _, label in sb.NAV)
    cfg = (ROOT / ".streamlit" / "config.toml").read_text()
    assert re.search(r"^showSidebarNavigation\s*=\s*false\s*$", cfg, re.M)
    from streamlit.errors import StreamlitPageNotFoundError
    calls = []

    class Side:
        def page_link(self, path, label=None):
            calls.append((path, label))
            if path.startswith("pages/trace"):
                raise StreamlitPageNotFoundError("x", "y", False)  # tolerated: logged, not raised

    class St:
        sidebar = Side()

    sb.render_nav(St())
    assert [c[1] for c in calls] == ["Ask VERA-DAR", "Example report", "Trace Eval"]


def test_main_page_nav_renders_via_apptest_with_labels(monkeypatch):
    import requests

    def _offline(*a, **k):
        raise requests.exceptions.ConnectionError("offline in tests")
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    # the footer's /version lookup (streamlit_app._footer_commit) is best-effort; keep it off the network
    monkeypatch.setattr(requests, "get", _offline)
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception
    labels = [e.proto.label for e in at.get("page_link")]
    assert labels == ["Ask VERA-DAR", "Example report", "Trace Eval"]


def _no_network(monkeypatch):
    import requests
    import socket

    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.delenv("VERA_API_KEY", raising=False)


def _tree_text(at):
    import pandas as pd
    parts = []
    def walk(n):
        proto = getattr(n, "proto", None)
        if proto is not None and hasattr(proto, "ListFields"):
            for f, v in proto.ListFields():
                if isinstance(v, bytes):
                    continue
                parts.append(str(v))
        val = getattr(n, "value", None)
        if isinstance(val, pd.DataFrame):
            parts.append(val.to_string())
        for ch in getattr(n, "children", {}).values() if isinstance(getattr(n, "children", None), dict) else []:
            walk(ch)
    walk(at._tree)
    return "\n".join(parts)


def test_main_page_public_scan_with_sentinel_response(monkeypatch):
    """Whole main page (public mode) with a rich sentinel /ask response: no model/provider/gate terms anywhere."""
    from types import SimpleNamespace
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_KEY", "k")
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    body = {"answer": "An answer [1].", "result_kind": "answer", "tokens_used": 12, "cost_usd": 0.0001,
            "model": SENT_MODEL, "sources": [{"n": 1, "title": "T", "url": "https://x.org/a", "provider": SENT_MODEL,
                         "retrieved_at": "2030-01-01", "licence_decision": "allow", "identifier": "id1"}],
            "traced_sources": [{"title": "O", "url": "https://x.org/o"}],
            "claim_check": {"outcome": "checked", "checker": SENT_CHECKER, "checked": 1, "supported": 1, "partial": 0,
                            "removed": 0}}
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: body))
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(json=lambda: {"commit": "abcdef1234"}))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    at.text_input[0].set_value("What is VERA-DAR?")
    at.button[0].click()
    at.run()
    assert not at.exception and at.success
    text = _tree_text(at)
    assert SENT_MODEL not in text and SENT_CHECKER not in text and "2030-01-01" not in text
    assert not find_terms(text), find_terms(text)
    assert "force_bad" not in text.lower() and "Ask VERA-DAR" in text


EXPECTED_CAPTION = (
    "**How VERA-DAR answers:** for each in-scope question the Ask page searches published scholarly sources and the web "
    "live and lists the ones that may be relevant, with links. VERA-DAR has not read the full documents, so it does not "
    "use abstracts or snippets as evidence and does not write an answer from them. If no usable source is found it "
    "says so instead of answering from memory. Open the sources and read them yourself before relying on any claim. "
    "A third-party hosted language model may be used to decide whether a question is in scope.")


def test_sidebar_caption_exact_a0(monkeypatch):
    from vera import disclosure
    monkeypatch.setattr(disclosure, "fulldoc_enabled", lambda *a, **k: False)  # state 1
    at = AppTest.from_function(_page)
    at.run(timeout=30)
    assert not at.exception
    assert EXPECTED_CAPTION in [c.value for c in at.sidebar.caption]


def test_sidebar_caption_exact_shipped_state():
    at = AppTest.from_function(_page)  # shipped config = state 2 (P1.6)
    at.run(timeout=30)
    assert not at.exception
    assert sb._HOW_FULLDOC + " " + sb.MODEL_LINE in [c.value for c in at.sidebar.caption]


def test_sidebar_has_no_abstract_grounding_claims(monkeypatch):
    monkeypatch.setattr(sb, "grounding_cap", lambda *a, **k: 1500)
    at = AppTest.from_function(_page)
    at.run(timeout=30)
    assert not at.exception
    text = " ".join(c.value for c in at.sidebar.caption)
    for bad in ("answers only from the abstracts", "cites them as [n]", "Answers are drafted", "retrieval-augmented",
                "characters each", "1,500", "1500"):
        assert bad not in text, bad


def test_model_line_exact():
    assert sb.MODEL_LINE == "A third-party hosted language model may be used to decide whether a question is in scope."


def test_licence_sidebar_a0_exact():
    assert sb.LICENCE_SIDEBAR == ("Licence terms are as reported by the indexing services for the listed records and "
                                  "were not verified with the publisher.")
    assert "abstract" not in sb.LICENCE_SIDEBAR and sb.LICENCE_UNDER_ANSWER == sb.LICENCE_SIDEBAR


def test_sidebar_and_disclosure_share_a0_sentences(tmp_path):
    from vera import disclosure
    fd = tmp_path / "fd.json"
    fd.write_text(json.dumps({"fulldoc": {"enabled": False}}))
    t = disclosure.disclosure_text(fulldoc_path=fd)  # state 1 shares the A0 sentences
    for sentence in (sb.LICENCE_SIDEBAR, "does not use abstracts or snippets as evidence"):
        assert sentence in t, sentence
    on = disclosure.disclosure_text()  # shipped = state 2 (P1.6): same statements, state-2 wording
    for phrase in ("as reported by the indexing services and were not verified with the publisher",
                   "does not use their abstracts or snippets as evidence"):
        assert phrase in on, phrase
    # R-028 (2026-10-19): the disclosure replaced the scope-model sentence with the owner-approved third-party sentence
    for x in (t, on):
        assert "sent to third-party AI and search services" in x and sb.MODEL_LINE not in x
