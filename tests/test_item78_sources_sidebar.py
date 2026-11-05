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
    # Deny-All audit G1: no fixed Scope text helper remains in the sidebar module
    assert not hasattr(sb, "scope_text") and not hasattr(sb, "SCOPE_PATH")
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


def _notes_text(data=None):
    """What the About page shows under the corpus-notes heading (render_corpus_notes), as one string."""
    st, cap = _capture()
    sb.render_corpus_notes(st, data)
    return " ".join(cap)


@pytest.mark.parametrize("fulldoc", [False, True])
def test_sidebar_renders_honest_answer_source_without_hosts_or_dates_in_code(monkeypatch, fulldoc):
    from vera import disclosure
    monkeypatch.setattr(disclosure, "fulldoc_enabled", lambda *a, **k: fulldoc)
    at = AppTest.from_function(_page)
    at.run(timeout=30)
    assert not at.exception
    side = " ".join(c.value for c in at.sidebar.caption)
    assert "How VERA-DAR answers:" not in side and sb.MODEL_LINE not in side  # CORPUS-DIGEST-1104: moved to About
    text = _notes_text()  # the About page's corpus notes
    if fulldoc:  # state 2 (shipped after P1.6)
        assert sb._HOW_FULLDOC in text and "lists documents it has not read" in text
    else:
        assert "searches published scholarly sources and the web live" in text and "does not use abstracts or snippets as evidence" in text
        assert "instead of answering from memory" in text
    assert "onrender.com" not in text and "127.0.0.1" not in text
    assert "onrender.com" not in side and "127.0.0.1" not in side
    assert not re.search(r"20\d\d-\d\d-\d\d", (ROOT / "vera" / "sources_sidebar.py").read_text())


def test_every_page_calls_the_sidebar():
    for rel in ("streamlit_app.py", "pages/trace_eval.py", "pages/vera_demo.py"):
        assert "render_sources_sidebar(st)" in (ROOT / rel).read_text(), rel


def test_source_labels_do_not_overclaim_inspection():
    """DEC-EI1: no 'followed back to the original papers', no 'Evidence snapshot'; licences are intermediary-reported."""
    at = AppTest.from_function(_page)
    at.run(timeout=30)
    assert not at.exception
    side = " ".join(c.value for c in at.sidebar.caption)
    text = side + " " + _notes_text()  # sidebar plus the About page's corpus notes
    assert "followed back" not in text and "original papers" not in text and "Evidence snapshot" not in text
    assert "published scholarly sources and the web" in text and "search snippets" not in text and "Trace Eval sources" in text
    assert "Trace Eval sources" not in side and "published scholarly sources and the web" not in side
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
        def subheader(self, t, *a, **k): pass

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


def _sidebar_text(monkeypatch, snap_path, notes=False):
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
        def subheader(self, t, *a, **k): pass

    sb.render_sources_sidebar(St())
    if notes:  # the About page's corpus notes, under the same patched snapshot
        sb.render_corpus_notes(St(), None)
    return " ".join(cap)


def test_sidebar_never_discloses_individual_sources(tmp_path, monkeypatch):
    text = _sidebar_text(monkeypatch, _fixture_snapshot(tmp_path), notes=True)
    assert "3 distinct sources" in text  # aggregate still shown (About page notes)
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
    text = _sidebar_text(monkeypatch, _fixture_snapshot(tmp_path), notes=True)
    _no_internal_terms(text)
    _no_internal_terms(disclosure.DISCLOSURE_TEXT + " " + disclosure.DISCLOSURE_TEXT_UNCONFIRMED)
    cap = []

    class St:
        def caption(self, t): cap.append(t)

    sb.render_corpus(St(), {"summary_source": "agentic_checked", "source_count": 19, "corpus_updated_at": "2026-10-13"})
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
    text = _sidebar_text(monkeypatch, _fixture_snapshot(tmp_path), notes=True)
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
    base = {"summary_source": "agentic_checked", "source_count": 4, "summary_generated_at": "2026-10-13T00:00:00",
            "summary": "A short plain description.", "topics": ["a", "b"]}
    sb.render_corpus(st, {**base, "summary_model": "groq:openai/gpt-oss-20b", "extra": SENT_MODEL})
    st2, cap2 = _capture()
    sb.render_corpus(st2, base)
    assert cap == cap2
    joined = " ".join(cap)
    assert "written by a language model on 2026" in joined and "groq" not in joined and SENT_MODEL not in joined
    assert sb.CORPUS_FIELDS == {"summary_source", "source_count", "corpus_updated_at", "summary", "topics",
                                "summary_generated_at", "coverage", "topic_shares"}


def test_model_written_description_naming_a_backend_is_replaced():
    st, cap = _capture()
    sb.render_corpus(st, {"summary_source": "agentic_checked", "source_count": 4, "summary": "Written by gpt-4.1-nano.",
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
    text = _sidebar_text(monkeypatch, _fixture_snapshot(tmp_path), notes=True)
    assert "**Trace Eval sources:**" in text and "first posted 2024 to 2026" in text
    for old in ("index records", "metadata terms", "arXiv abstract records", "distinct arXiv abstracts", "arXiv identifiers"):
        assert old not in text, old
    notes = _notes_text({"summary_source": "agentic_checked", "source_count": 1})  # licence note now on the About page
    assert sb.LICENCE_SIDEBAR in notes and "not verified with the publisher" in sb.LICENCE_SIDEBAR
    st, cap = _capture()
    sb.render_corpus(st, {"summary_source": "agentic_checked", "source_count": 1})
    assert sb.LICENCE_SIDEBAR not in " ".join(cap)


def test_nav_labels_and_streamlit_config():
    assert tuple(label for _, label in sb.NAV) == ("Ask VERA-DAR", "Report", "Trace Eval", "How to use", "About")
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
    assert [c[1] for c in calls] == ["Ask VERA-DAR", "Report", "Trace Eval", "How to use", "About"]


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
    assert [l for l in labels if l in [n for _, n in sb.NAV]] == [n for _, n in sb.NAV]  # sidebar nav, in order
    assert labels[:2] == [l for _, l in __import__("vera.disclosure", fromlist=["x"]).ABOUT_LINKS]  # OD-1029-UI notice links


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
    "**How VERA-DAR answers:** for each question the Ask page searches published scholarly sources and the web "
    "live and lists the ones that may be relevant, with links. VERA-DAR has not read the full documents, so it does not "
    "use abstracts or snippets as evidence and does not write an answer from them. If no usable source is found it "
    "says so instead of answering from memory. Open the sources and read them yourself before relying on any claim. "
    "A third-party hosted language model may be used to check whether a request is safe to handle.")


def test_sidebar_caption_exact_a0(monkeypatch):
    from vera import disclosure
    monkeypatch.setattr(disclosure, "fulldoc_enabled", lambda *a, **k: False)  # state 1
    at = AppTest.from_function(_page)
    at.run(timeout=30)
    assert not at.exception
    expected = EXPECTED_CAPTION.replace(" " + sb.MODEL_LINE, __import__("vera.disclosure").disclosure.offtopic_sentence() + " " + sb.MODEL_LINE)
    assert expected not in [c.value for c in at.sidebar.caption]  # CORPUS-DIGEST-1104: no longer in the sidebar
    st, cap = _capture()
    sb.render_corpus_notes(st, None)
    # now a line on the About page, without the off-topic sentence, which has its own About disclosure section
    assert EXPECTED_CAPTION in cap and expected not in cap


def test_sidebar_caption_exact_shipped_state():
    at = AppTest.from_function(_page)  # shipped config = state 2 (P1.6)
    at.run(timeout=30)
    assert not at.exception
    line = sb._HOW_FULLDOC + __import__("vera.disclosure").disclosure.offtopic_sentence() + " " + sb.MODEL_LINE
    assert line not in [c.value for c in at.sidebar.caption]  # CORPUS-DIGEST-1104: moved to the About page
    st, cap = _capture()
    sb.render_corpus_notes(st, None)
    assert sb._HOW_FULLDOC + " " + sb.MODEL_LINE in cap and line not in cap  # off-topic sentence: own About section


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
    # INTEGRATE-1104 round 2: no fixed Scope wording; the classifier is a safety check only
    assert sb.MODEL_LINE == "A third-party hosted language model may be used to check whether a request is safe to handle."


def test_licence_sidebar_a0_exact():
    assert sb.LICENCE_SIDEBAR == ("For listed records, only the indexing services' metadata licence was checked; the "
                                  "articles' own licences were not verified with the publisher.")
    assert "abstract" not in sb.LICENCE_SIDEBAR and sb.LICENCE_UNDER_ANSWER == sb.LICENCE_SIDEBAR


def test_sidebar_and_disclosure_share_a0_sentences(tmp_path):
    from vera import disclosure
    fd = tmp_path / "fd.json"
    fd.write_text(json.dumps({"fulldoc": {"enabled": False}}))
    t = disclosure.disclosure_text(fulldoc_path=fd)  # state 1 shares the A0 sentences
    for sentence in (sb.LICENCE_SIDEBAR, "does not use abstracts or snippets as evidence"):
        assert sentence in t, sentence
    on = disclosure.disclosure_text()  # shipped = state 2 (P1.6): same statements, state-2 wording
    for phrase in ("only the indexing services' metadata licence was checked; the articles' own licences were not verified with the publisher",
                   "does not use their abstracts or snippets as evidence"):
        assert phrase in on, phrase
    # R-028 (2026-10-19): the disclosure replaced the scope-model sentence with the owner-approved third-party sentence
    for x in (t, on):
        assert "sent to third-party AI and search services" in x and sb.MODEL_LINE not in x


def test_corpus_failed_fetch_short_cache_success_long_cache(monkeypatch):
    from types import SimpleNamespace
    from vera import public_mode, sources_sidebar as sb
    clock, calls, results = [1000.0], [], [None, {"summary_source": "agentic_checked", "source_count": 3}]

    def fake(*a, **k):
        calls.append(1)
        return results[min(len(calls) - 1, 1)]
    monkeypatch.setattr(public_mode, "get_corpus_summary", fake)
    monkeypatch.setattr(public_mode, "upstream_base_url", lambda: "http://x.invalid")
    import time
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    st = SimpleNamespace(session_state={})
    assert sb._corpus(st) is None and len(calls) == 1
    clock[0] += sb.CORPUS_RETRY_S - 1
    assert sb._corpus(st) is None and len(calls) == 1  # failure remembered within the retry window
    clock[0] += 2
    good = sb._corpus(st)  # retried after the window; real description shown
    assert good and good["source_count"] == 3 and len(calls) == 2
    clock[0] += sb.CORPUS_REFRESH_S - 1
    assert sb._corpus(st) == good and len(calls) == 2  # success cached for the full refresh period
    clock[0] += 2
    sb._corpus(st)
    assert len(calls) == 3


def test_trace_eval_line_is_marked_historical(tmp_path, monkeypatch):
    import json
    from unittest import mock
    from vera import sources_sidebar as sb
    snap = tmp_path / "trace_eval_v1.json"
    snap.write_text(json.dumps({"traces": [{"id": "q1", "sources": [{"url": "https://arxiv.org/abs/2401.00001"}]}],
                                "generated_at": "2026-01-02T00:00:00Z"}), encoding="utf-8")
    monkeypatch.setattr(sb, "_public_snapshot", lambda: None)
    monkeypatch.setattr(sb, "SNAPSHOT_PATH", snap)
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    monkeypatch.setattr(sb, "how_answers_base", lambda: "how")
    st = mock.MagicMock()
    sb.render_sources_sidebar(st)
    assert not any("Trace Eval" in str(c.args[0]) for c in st.caption.call_args_list)  # not in the sidebar
    line = next(c for c in sb.corpus_notes_lines(None) if "Trace Eval sources" in c)  # on the About page
    assert "historical" in line and "1 distinct sources" in line


def test_trace_eval_aggregate_only_line_is_marked_historical(monkeypatch):
    """WP6 (issue c): the aggregate-only (no per-source count) branch is marked historical too."""
    from unittest import mock
    from vera import sources_sidebar as sb
    monkeypatch.setattr(sb, "snapshot_summary", lambda: {"count": None, "questions": 26, "captured": "2026-10-23",
                                                         "years": None})
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    monkeypatch.setattr(sb, "how_answers_base", lambda: "how")
    st = mock.MagicMock()
    sb.render_sources_sidebar(st)
    assert not any("Trace Eval" in str(c.args[0]) for c in st.caption.call_args_list)  # not in the sidebar
    line = next(c for c in sb.corpus_notes_lines(None) if "Trace Eval sources" in c)  # on the About page
    assert "historical" in line and "per-source counts are not published" in line and "26 test questions" in line


# ---- CORPUS-DIGEST-1104: sidebar slimmed; licence/how-answers/Trace Eval/coverage on the About page ----
_SHARES = {"counts": "documents", "basis_documents": 48,
           "main": [{"label": "Epidemiology", "documents": 15, "pct": 31}, {"label": "Public health", "documents": 12, "pct": 25},
                    {"label": "Vaccines", "documents": 11, "pct": 23}],
           "main_pct": 79, "target_reached": True, "other": {"topics": 4, "documents": 8, "pct": 17},
           "ungrouped": {"documents": 2, "pct": 4}}
_AGENTIC = {"summary_source": "agentic_checked", "source_count": 48, "corpus_updated_at": "2026-10-30T00:00:00",
            "summary": "A plain description.", "summary_generated_at": "2026-11-01T00:00:00",
            "coverage": "Mostly epidemiology and public health.", "topic_shares": _SHARES}


def test_sidebar_has_no_scope_licence_howanswers_or_trace_eval(tmp_path, monkeypatch):
    side = _sidebar_text(monkeypatch, _fixture_snapshot(tmp_path))
    st, cap = _capture()
    sb.render_corpus(st, _AGENTIC)
    side += " " + " ".join(cap)
    for needle in ("Scope", sb.LICENCE_SIDEBAR, "not verified with the publisher", "How VERA-DAR answers:",
                   sb.MODEL_LINE, "Trace Eval", "Current coverage", FIXED_SCOPE):
        assert needle not in side, needle
    assert "What VERA-DAR" not in side and sb.ABOUT_LINK_FALLBACK in side  # About link (fallback caption here)


FIXED_SCOPE = "Published evidence (2023-2026) on AI coding assistants and sentinel fixed scope text"


def test_sidebar_renders_about_page_link_when_available():
    from unittest import mock
    st = mock.MagicMock()
    with mock.patch.object(sb, "render_corpus"), mock.patch.object(sb, "_corpus"):
        sb.render_sources_sidebar(st)
    st.page_link.assert_called_once_with("pages/about.py", label=sb.ABOUT_LINK_LABEL)


@pytest.mark.parametrize("data", [None, {"summary_source": "deterministic_fallback", "source_count": 3, "summary": "S.",
                                         "coverage": "Ignored fallback coverage."},
                                  {"summary_source": "unavailable"}])
def test_no_fixed_scope_text_without_generated_description(data, tmp_path, monkeypatch):
    # a configured scope string is present (sentinel), so the test holds whatever the shipped config holds
    scope = FIXED_SCOPE  # sentinel: nothing on any page may render a fixed scope string
    from vera import help_pages
    monkeypatch.setattr(sb, "_corpus", lambda st: data)
    pages = []
    for render in (help_pages.render_about, help_pages.render_how_to_use):
        st, cap = _capture()
        st.title = st.info = st.subheader = lambda *a, **k: None
        render(st)
        pages.append(" ".join(cap))
    st, cap = _capture()
    sb.render_corpus(st, data)
    pages.append(" ".join(cap))
    pages.append(_sidebar_text(monkeypatch, _fixture_snapshot(tmp_path)))
    for text in pages:
        assert FIXED_SCOPE not in text and sb.plain(FIXED_SCOPE) not in text  # INTEGRATE-1104: the sentinel itself
        assert "**Scope:**" not in text and "Scope:" not in text
        assert "Current coverage" not in text and "Ignored fallback coverage" not in text
    assert sb.coverage_text(data) is None
    assert not any("coverage" in line.lower() for line in sb.corpus_notes_lines(data))


def test_generated_coverage_renders_on_about_and_how_to_use(monkeypatch):
    from vera import help_pages
    monkeypatch.setattr(sb, "_corpus", lambda st: _AGENTIC)
    for render in (help_pages.render_about, help_pages.render_how_to_use):
        st, cap = _capture()
        st.title = st.info = st.subheader = lambda *a, **k: None
        render(st)
        assert any(sb.plain("Mostly epidemiology and public health.") in c and "overage" in c for c in cap), render.__name__
    assert sb.corpus_notes_lines(_AGENTIC)[0] == r"**Current coverage:** Mostly epidemiology and public health\."
    # a coverage naming a backend is dropped, never shown
    assert sb.coverage_text({**_AGENTIC, "coverage": "Written by gpt-4.1-nano."}) is None


def test_main_topic_share_lines_render():
    st, cap = _capture()
    sb.render_corpus(st, _AGENTIC)
    joined = "\n".join(cap)
    assert "Epidemiology 31%" in joined and "Public health 25%" in joined and "Vaccines 23%" in joined
    assert "Other topics: 17% across 4 smaller topics." in cap
    assert "Not grouped under a topic: 4%." in cap
    assert any(c.startswith("**Main topics** (79% of documents") and "48 sampled sources" in c for c in cap)
    assert not any(c.startswith("Topics:") for c in cap)
    # a single smaller topic is singular; shares ignored unless the description is model-written
    st, cap = _capture()
    sb.render_corpus(st, {**_AGENTIC, "topic_shares": {**_SHARES, "other": {"topics": 1, "pct": 5}, "ungrouped": {"documents": 0, "pct": 0}}})
    assert "Other topics: 5% across 1 smaller topic." in cap and not any("Not grouped" in c for c in cap)
    st, cap = _capture()
    sb.render_corpus(st, {**_AGENTIC, "summary_source": "deterministic_fallback", "topics": ["a", "b"]})
    assert not any("Topics" in c for c in cap)  # G1: model-written topics need the checker-pass marker



# --- Deny-All audit G1: model-written text shows only with the checker-pass marker ---------------------------------
@pytest.mark.parametrize("source", ["agentic_model", "deterministic_fallback", "something_else", None])
def test_model_text_needs_the_checker_pass_marker(source):
    data = {**_AGENTIC, "summary_source": source, "summary": "Model summary text.", "topics": ["Model topic"],
            "coverage": "Model coverage text."}
    st, cap = _capture()
    sb.render_corpus(st, data)
    joined = " ".join(cap)
    assert "Model topic" not in joined and "Main topics" not in joined and "Epidemiology" not in joined
    assert "Description written by a language model" not in joined and sb.coverage_text(data) is None
    if source == "deterministic_fallback":
        assert sb.plain("Model summary text.") in joined  # code-built summary text (fallback) still shows
    else:
        assert sb.plain("Model summary text.") not in joined
    if source is not None:
        assert cap[0].startswith("**VERA-DAR's corpus:** 48 sources")  # code-built count line always shows


def test_checked_marker_shows_model_text_and_matches_the_pipeline():
    from vera import corpus_description as cd
    assert sb.CHECKED == cd.CHECKED
    st, cap = _capture()
    sb.render_corpus(st, {**_AGENTIC, "summary": "Model summary text."})
    joined = " ".join(cap)
    assert sb.plain("Model summary text.") in joined and "Main topics" in joined and "Description written by a language model" in joined
