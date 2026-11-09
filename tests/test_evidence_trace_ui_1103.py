import re

import pytest

from vera import evidence_trace as et
from vera import evidence_trace_ui as ui


class Sidebar:
    def __getattr__(self, k):
        raise AssertionError("st.sidebar must not be used")


class FakeSt:
    def __init__(self):
        self.out = []

    sidebar = Sidebar()

    def markdown(self, t, **kw):
        assert "unsafe_allow_html" not in kw
        self.out.append(t)

    caption = markdown


def trace(**kw):
    t = {"version": 1, "view": "public",
         "passages": [{"n": 1, "cited": True, "supports": ["Beta *varies* across sites [1]."], "verdict": "supported",
                       "excerpt": "Beta systems vary widely.", "reason": None},
                      {"n": 2, "cited": False, "supports": [], "verdict": None, "excerpt": None,
                       "reason": "the answer did not use this passage"}],
         "parts": [{"label": "Second part", "covered": False}],
         "adequacy": {"verdict": "qualify", "all_declared_parts_supported": False},
         "claims": {"supported": 1, "partial": 0, "unsupported": 1}, "not_used": {"source not usable": 2}}
    t.update(kw)
    return t


def test_renders_main_pane_sections_and_returns_true():
    st = FakeSt()
    assert ui.render_trace(st, trace(), public=True) is True
    text = "\n".join(st.out)
    assert "How VERA-DAR chose the evidence" in text and "Parts of your question the evidence did not cover" in text
    assert "Citation check" in text and "selected, not cited" in text and "Second part" in text


def test_false_and_empty_for_none_empty_dev_in_public_and_missing_labels(monkeypatch):
    for t in (None, {}, "x"):
        st = FakeSt()
        assert ui.render_trace(st, t, public=True) is False and st.out == []
    st = FakeSt()
    assert ui.render_trace(st, trace(view="dev"), public=True) is False and st.out == []
    monkeypatch.setattr(et, "LABELS_PATH", et.LABELS_PATH.with_name("absent.json"))
    assert ui.render_trace(FakeSt(), trace(), public=True) is False


def test_public_none_resolves_public_mode(monkeypatch):
    monkeypatch.setattr(et, "_is_public_mode", lambda: True)
    assert ui.render_trace(FakeSt(), trace(view="dev")) is False
    assert ui.render_trace(FakeSt(), trace()) is True


def test_dynamic_strings_escaped_and_backend_words_dropped():
    t = trace()
    t["passages"][0]["supports"] = ["<script>x</script> ![i](http://h/p.png) [l](javascript:1)", "Groq said so"]
    st = FakeSt()
    ui.render_trace(st, t, public=True)
    text = "\n".join(st.out)
    assert "<script>" not in text and "![" not in text and "javascript:" not in text and "Groq" not in text


def test_redact_applied_to_hosts_and_urls_after_escaping():
    red = lambda s: re.sub(r"https?://\S+|secret\.example\.org", "[redacted]", s)
    t = trace()
    t["passages"][0]["supports"] = ["see https://secret.example.org/x and secret.example.org"]
    t["parts"] = [{"label": "host secret.example.org", "covered": False}]
    st = FakeSt()
    assert ui.render_trace(st, t, public=True, redact=red) is True
    text = "\n".join(st.out)
    assert "secret" not in text and "example" not in text and "redacted" in text, text


def test_redact_runs_on_raw_text_for_hosts_with_markdown_specials():
    red = lambda s: re.sub(r"[\w-]+(?:[.][\w-]+)*_[\w-]+(?:[.][\w-]+)+", "[redacted]", s)
    t = trace()
    t["passages"][0]["supports"] = ["see my_host-1.sub_domain.example.com now"]
    st = FakeSt()
    assert ui.render_trace(st, t, public=True, redact=red) is True
    text = "\n".join(st.out)
    assert "example" not in text, text
    assert "sub_domain" not in text and "my_host" not in text and "redacted" in text
