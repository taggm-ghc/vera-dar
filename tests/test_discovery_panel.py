"""K2.I5: fail-closed Ask-page branch. Only result_kind == "answer" shows the green Answer block."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from streamlit.testing.v1 import AppTest

from vera import answer_contract as ac
from vera import discovery_panel as dp
from vera import sources_sidebar as sb

ROOT = Path(dp.__file__).resolve().parent.parent
TEXTS = json.loads((ROOT / "config" / "discovery_contract.json").read_text())["texts"]
KINDS = [None, "MISSING", "unspecified", "reading_list", "discovery", "decline", "bogus"]
FORBIDDEN = ("saved", "stored", "verified", "gate", "pending", "queue", "owner key", "d6")


def body(kind="reading_list", **over):
    d = {"answer": ac.text("not_read_line") or "x", "tokens_used": 1, "cost_usd": 0.0,
         "candidates": [{"title": "Paper One", "url": "https://x.org/a", "identifier": "10.1/abc",
                         "published": "2025", "acquisition": "not_acquired", "provider": "SENTPROV",
                         "retrieved_at": "2030-01-01", "licence_decision": "SENTLIC"}]}
    if kind != "MISSING":
        d["result_kind"] = kind
    d.update(over)
    return d


def texts(st):
    out = []
    for name in ("markdown", "caption", "info", "warning", "success"):
        out += [str(c.args[0]) for c in getattr(st, name).call_args_list]
    return out


def render(data):
    st = mock.MagicMock()
    dp.render_discovery(st, data)
    return st


@pytest.mark.parametrize("kind", KINDS)
def test_non_answer_kinds_never_answer_heading_or_success(kind):
    st = render(body(kind))
    assert not st.success.called
    assert not any("✅ Answer" in t for t in texts(st))


def _app(monkeypatch, data):
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_KEY", "k")
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: data))
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(json=lambda: {"commit": "abcdef1234"}))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    at.text_input[0].set_value("What is VERA-DAR?")
    at.button[0].click()
    return at.run()


@pytest.mark.parametrize("kind", KINDS)
def test_app_non_answer_kinds_show_no_success(monkeypatch, kind):
    at = _app(monkeypatch, body(kind))
    assert not at.exception and not at.success
    assert not any("✅ Answer" in m.value for m in at.markdown)


def test_not_read_line_verbatim_first_element():
    st = render(body())
    assert st.warning.call_args_list[0].args[0] == TEXTS["not_read_line"]
    names = [c[0] for c in st.method_calls]
    assert names.index("warning") < names.index("markdown", names.index("markdown") + 1)


def test_partial_variant_on_mixed_acquisition():
    d = body()
    d["candidates"].append(dict(d["candidates"][0], acquisition="full_document_acquired", url="https://x.org/b"))
    assert render(d).warning.call_args_list[0].args[0] == TEXTS["not_read_line_partial"]


def test_unknown_acquisition_shown_as_not_acquired():
    d = body()
    d["candidates"][0]["acquisition"] = "totally_acquired"
    d["candidates"][0]["representation"] = "SENTREP"
    st = render(d)
    t = " ".join(texts(st))
    assert TEXTS["representation_not_acquired"] in t and "SENTREP" not in t
    assert st.warning.call_args_list[0].args[0] == TEXTS["not_read_line"]


@pytest.mark.parametrize("kind", [None, "MISSING", "unspecified"])
def test_non_answer_ignores_sources_traced_claim_check_recalled_and_answer_text(kind):
    d = body(kind, answer="SENTANSWER", sources=[{"title": "SENTSRC", "url": "https://s.org"}],
             traced_sources=[{"title": "SENTTRACED"}], claim_check={"outcome": "SENTCHECK"},
             recalled=[{"claim_text": "SENTRECALLED"}], memory_submitted=3, candidates=[])
    t = " ".join(texts(render(d)))
    for s in ("SENTANSWER", "SENTSRC", "SENTTRACED", "SENTCHECK", "SENTRECALLED"):
        assert s not in t
    assert TEXTS["not_read_line"] in t
    d2 = body(kind, sources=[{"title": "SENTSRC"}], claim_check={"outcome": "SENTCHECK"}, recalled=[{"claim_text": "SENTRECALLED"}])
    t2 = " ".join(texts(render(d2)))
    assert "SENTSRC" not in t2 and "SENTCHECK" not in t2 and "SENTRECALLED" not in t2


def test_candidate_line_hides_provider_retrieved_at_licence_decision():
    t = " ".join(texts(render(body())))
    for s in ("SENTPROV", "2030-01-01", "SENTLIC"):
        assert s not in t
    assert "1. [Paper One](https://x.org/a) (2025)" in t and sb.plain("10.1/abc") in t
    assert sb.LICENCE_SIDEBAR in t
    d = body()
    del d["candidates"][0]["licence_decision"]
    assert sb.LICENCE_SIDEBAR not in " ".join(texts(render(d)))


def test_empty_reading_list_and_decline_show_server_text_escaped():
    st = render(body("reading_list", candidates=[], answer="<script>alert(1)</script> [a](javascript:x)"))
    t = " ".join(texts(st))
    assert "<script>" not in t and "&lt;script&gt;" in t and "(javascript:x)" not in t
    st = render(body("decline", answer="<b>no</b> ![x](http://evil/p.png)"))
    t = " ".join(texts(st))
    assert "<b>" not in t and "![x]" not in t and st.info.called and not st.markdown.called
    st = render(body("discovery", candidates=[], answer="SERVERTEXT"))
    assert "SERVERTEXT" not in " ".join(texts(st)) and st.warning.call_args.args[0] == TEXTS["not_read_line"]


def test_injection_in_titles_and_urls_neutralised():
    d = body(candidates=[{"title": "<img src=x onerror=1> [x](javascript:y)", "url": "javascript:alert(1)"},
                         {"title": "Ok", "url": "https://x.org/a) [evil](http://e.com"},
                         {"title": 5, "url": "https://x.org/c"}, "str", {"title": "no url"}])
    t = " ".join(texts(render(d)))
    assert "](javascript:" not in t and "](http://e.com" not in t and "<img" not in t.replace("\\<", "")
    assert t.count("1. ") == 1 and "2. Ok" in t and "3. " not in t
    assert render(body("discovery", candidates="notalist")).warning.call_args.args[0] == TEXTS["not_read_line"]


def test_discovery_note_line_verbatim_only_when_positive_int():
    for v in (0, -1, True, "1", None, 1.0, [1]):
        assert TEXTS["discovery_note_submitted"] not in " ".join(texts(render(body(discovery_note_submitted=v))))
    st = render(body(discovery_note_submitted=1))
    assert st.info.call_args.args[0] == TEXTS["discovery_note_submitted"]
    assert not any(ch.isdigit() for ch in st.info.call_args.args[0])


def test_recalled_notes_caption_and_chrome_forbidden_words():
    d = body(recalled_discovery=[{"kind": "discovery_note", "note": "Discovery note (not evidence). For topic x.",
                                  "urls": ["https://a.org/p", "javascript:x"], "saved_at": "SAVEDSENT"}])
    st = render(d)
    assert st.caption.call_args_list[-1].args[0] == TEXTS["recalled_note_caption"]
    t = " ".join(texts(st))
    assert "**Earlier reading-list notes**" in t and "SAVEDSENT" not in t and sb._link("https://a.org/p", "https://a.org/p") in t
    assert "](javascript:" not in t
    chrome = [TEXTS["recalled_note_caption"], TEXTS["discovery_note_submitted"], "Earlier reading-list notes", "Reading list"]
    for c in chrome:
        assert not any(w in c.lower() for w in FORBIDDEN), c
    assert render(body(recalled_discovery=[])).caption.call_count == 2  # only candidate + licence captions


def test_answer_kind_keeps_legacy_block(monkeypatch):
    d = {"result_kind": "answer", "answer": "Plain answer.", "tokens_used": 1, "cost_usd": 0.0,
         "sources": [], "traced_sources": [], "candidates": [{"title": "SENTCAND", "url": "https://x.org/a"}]}
    at = _app(monkeypatch, d)
    assert not at.exception and at.success and any("✅ Answer" in m.value for m in at.markdown)
    assert not any("Reading list" in m.value or "SENTCAND" in m.value for m in at.markdown)
    tree = ast.parse((ROOT / "streamlit_app.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") in
             ("render_memory_panel", "render_discovery")]
    assert sorted(c.func.id for c in calls) == ["render_discovery", "render_memory_panel"]


def test_success_and_answer_heading_only_inside_answer_branch():
    tree = ast.parse((ROOT / "streamlit_app.py").read_text())
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}

    def guarded(node, orelse_ok=False):
        child, p = node, parents.get(node)
        while p is not None:
            if isinstance(p, ast.If) and child in p.body and isinstance(p.test, ast.Compare):
                src = ast.unparse(p.test)
                if "result_kind" in src and "== 'answer'" in src:
                    return True
            child, p = p, parents.get(p)
        return False

    hits = [n for n in ast.walk(tree) if (isinstance(n, ast.Call) and ast.unparse(n.func) == "st.success")
            or (isinstance(n, ast.Constant) and isinstance(n.value, str) and "✅ Answer" in n.value)
            or (isinstance(n, ast.Call) and getattr(n.func, "id", "") in ("render_sources", "render_memory_panel"))]
    assert len(hits) >= 3 and all(guarded(h) for h in hits)


def test_discovery_panel_module_never_touches_answer_fields():
    src = (ROOT / "vera" / "discovery_panel.py").read_text().split('"""', 2)[2]
    for bad in ("st.success", "render_sources", "render_memory_panel(", '"sources"', '"traced_sources"',
                '"claim_check"', '"recalled"', '"memory_submitted"'):
        assert bad not in src, bad
