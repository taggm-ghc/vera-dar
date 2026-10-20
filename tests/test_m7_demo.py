"""Mock DB data: every section shows; loader fallbacks are verbose; HTML in data is escaped."""
import json
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from vera.m7 import data
from vera.m7.fixture import build_fixture_report


def _render(rep):
    from vera.m7.demo_ui import render_demo
    render_demo(rep, "mock-db")


def _run(rep):
    return AppTest.from_function(_render, kwargs={"rep": rep}, default_timeout=30).run()


def test_all_five_sections_and_audit_tabs_present():
    at = _run(build_fixture_report())
    heads = [h.value for h in at.header]
    for n in ("1.", "2.", "3.", "4.", "5."):
        assert any(h.startswith(n) for h in heads), n
    assert [t.label for t in at.tabs] == ["Source selection", "Evidence quality", "Answer adequacy", "Evidence relations", "Claim verification"]
    assert len(at.dataframe) >= 3                      # eval table + cost table + gate A table at least
    labels = " ".join(e.label for e in at.expander)
    assert "Source-selection rationale" in labels and "Rationale: Grounding" in labels and "c1 [" in labels


def test_claim_expander_shows_source_info():
    at = _run(build_fixture_report())
    md = " ".join(m.value for m in at.markdown)
    assert "fixture://source-1" in md or "`fixture://source-1`" in md


def test_html_in_span_text_is_escaped():
    rep = build_fixture_report()
    rep["corpus"]["spans"]["sp1"]["text"] = "<script>alert(1)</script>"
    at = _run(rep)
    md = " ".join(m.value for m in at.markdown)
    assert "<script>" not in md and "&lt;script&gt;" in md


def test_javascript_url_not_linked():
    from vera.m7.demo_ui import _safe_link
    assert _safe_link("javascript:alert(1)") == "`javascript:alert(1)`"
    assert _safe_link("https://x.org").startswith("[")


def test_loader_reads_db_row(monkeypatch):
    rep = build_fixture_report()
    rep["is_fixture"] = False

    class S:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a, **k): return SimpleNamespace(first=lambda: ("7", json.dumps(rep)))
    monkeypatch.delenv("VERA_DEMO_SOURCE", raising=False)
    monkeypatch.setattr("vera.db.get_session", lambda: S())
    got, note = data.load_latest_report()
    assert note == "vera_vjay.runs run_id=7" and not got.get("is_fixture")


def test_loader_falls_back_verbosely_on_db_error(monkeypatch):
    monkeypatch.delenv("VERA_DEMO_SOURCE", raising=False)

    def boom():
        raise RuntimeError("EXTERNAL_DB_URL is not set")
    monkeypatch.setattr("vera.db.get_session", boom)
    got, note = data.load_latest_report()
    assert got["is_fixture"] and "FIXTURE fallback" in note and "EXTERNAL_DB_URL" in note


def test_loader_no_rows_says_m6_not_run(monkeypatch):
    class S:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a, **k): return SimpleNamespace(first=lambda: None)
    monkeypatch.delenv("VERA_DEMO_SOURCE", raising=False)
    monkeypatch.setattr("vera.db.get_session", lambda: S())
    got, note = data.load_latest_report()
    assert got["is_fixture"] and "M6 has not been run" in note
